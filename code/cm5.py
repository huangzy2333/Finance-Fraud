import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import cross_val_score, ShuffleSplit
#from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
#from imblearn.combine import SMOTEENN
#from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler, SMOTE
#from imblearn.ensemble import EasyEnsembleClassifier
from tensorflow import random
from tensorflow import keras
#from sklearn.ensemble import StackingClassifier
#from sklearn.metrics import make_scorer
from sklearn.feature_selection import SelectFromModel


def data_processing(X_train, X_test):
    scaler = preprocessing.StandardScaler()
    train_scale = scaler.fit_transform(X_train)
    test_scale = scaler.transform(X_test)
    return train_scale, test_scale


# get a list of models to evaluate

# build cnn_model

def create_model(shape):
    np.random.seed(42)
    random.set_seed(42)
    optimizer = keras.optimizers.SGD()
    cnn_model = keras.models.Sequential()
    cnn_model.add(keras.layers.Conv1D(256,3,activation='tanh',input_shape=(shape,1)))
    cnn_model.add(keras.layers.Dropout(0.2))
    cnn_model.add(keras.layers.Conv1D(128,3,activation='tanh'))
    cnn_model.add(keras.layers.Dropout(0.2))
    cnn_model.add(keras.layers.Flatten())
    cnn_model.add(keras.layers.Dense(64,activation='tanh'))
    cnn_model.add(keras.layers.Dense(1,activation='sigmoid'))    
    cnn_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return cnn_model  

#cnn_model = keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model)

def get_models():
    models = dict()
    models['lr'] = LogisticRegression(solver='newton-cg', max_iter=500)
    #models['svm'] = LinearSVC(probability=True)
    models['svm'] = SVC(C=0.5, kernel='linear', probability=True)
    #models['gbdt'] = GradientBoostingClassifier(n_estimators=1000)
    models['xgb'] = XGBClassifier(booster='gblinear', learning_rate=0.01, use_label_encoder=False, eval_metric='logloss', n_estimators=500)
    models['cnn'] = "models"
    return models


def feature_select(models, X, X_test, y):
    selector = SelectFromModel(estimator=models).fit(X, y)
    X_train_select = selector.transform(X)
    X_test_select = selector.transform(X_test)
    return X_train_select, X_test_select


def kfold_training(name, models, X, y, X_test, y_test, cv):
    train_pred_set = []
    test_pred_set = []
    train_true_set = []
    X, X_test = data_processing(X, X_test)
    if name == 'cnn':
        X_reshape = X.reshape(X.shape[0], X.shape[1], 1)
        X_test_reshape = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        for k, (train, valid) in enumerate(cv.split(X, y)):
            model = create_model(X.shape[1])
            model.fit(X_reshape[train], y[train], batch_size=1, epochs=10)
            train_pred_y = model.predict(X_reshape[valid]).flatten()
            test_pred_y = model.predict(X_test_reshape).flatten()
            train_true_set.extend(y[valid])
            train_pred_set.extend(train_pred_y)
            test_pred_set.append(test_pred_y)
    else:
        X, X_test = feature_select(models, X, X_test, y)
        for k, (train, valid) in enumerate(cv.split(X, y)):
            model = models.fit(X[train], y[train])
            train_pred_y = model.predict_proba(X[valid])[:, 1]
            train_true_set.extend(y[valid])
            test_pred_y = model.predict_proba(X_test)[:, 1]
            train_pred_set.extend(train_pred_y)
            test_pred_set.append(test_pred_y)

    X_train_proba = train_pred_set
    X_test_proba = np.array(test_pred_set).mean(axis=0).tolist()
    y_train_true = train_true_set
    return X_train_proba, X_test_proba, y_train_true


def get_stacking(X_train, y_train, X_test, y_test, random_seed=42):
    X_train_level1 = []
    X_test_level1 = []
    models = get_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    for name, models in models.items():
        X_train_proba, X_test_proba, y_train_true = kfold_training(name=name, models=models, X=X_train, y=y_train,
                                                                   X_test=X_test, y_test=y_test, cv=cv)
        X_train_level1.append(X_train_proba)
        X_test_level1.append(X_test_proba)

    X_train_level1_array = np.array(X_train_level1).T
    X_test_level1_array = np.array(X_test_level1).T
    y_train_true = np.array(y_train_true)
    return X_train_level1_array, X_test_level1_array, y_train_true


def stacking_final_predict(X_train_level1, X_train, y_train_true, X_test_level1, X_test, y_test):
    auc_set = []
    report_set = []
    # X_train, X_test = data_processing(X_train, X_test)
    # X_all_train = np.concatenate([X_train, X_train_level1], axis=1)
    # X_all_test = np.concatenate([X_test, X_test_level1], axis=1)
    X_all_train = X_train_level1
    X_all_test = X_test_level1
    cv_1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    X_all_train, X_all_test = feature_select(LogisticRegression(), X_all_train, X_all_test, y)
    for k, (train, valid) in enumerate(cv_1.split(X_all_train, y_train_true)):
        #final_estimator = LogisticRegression(solver='newton-cg', max_iter=500).fit(X_all_train[train],y_train_true[train])
        final_estimator = XGBClassifier(booster='gblinear', learning_rate=0.001, use_label_encoder=False, eval_metric='logloss', n_estimators=500).fit(X_all_train, y_train_true)

        y_pred = final_estimator.predict(X_all_test)
        y_pred_proba = final_estimator.predict_proba(X_all_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred, output_dict=True)

        report_0 = list(report['0'].values())
        report_1 = list(report['1'].values())
        report_0.extend(report_1)
        auc_set.append(auc)
        report_set.append(report_0)
    auc_mean = np.mean(auc_set)
    report_mean = np.mean(report_set, axis=0)
    auc_report = np.insert(report_mean, 0, auc_mean).tolist()
    return auc_report, y_pred, y_pred_proba


def model_select(X_train_level1, y_train_true):
    model_auc = []
    for i in range(4):
        y_pred_proba = X_train_level1[:, i]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        auc = roc_auc_score(y_train_true, y_pred_proba)
        report = classification_report(y_train_true, y_pred, output_dict=True)
        model_auc.append(auc)
        print("第 %d 个模型" % i)
        print("auc: %f" % auc)
        print(report['0'].values())
        print(report['1'].values())
    model_auc_array = np.array(model_auc)
    greatest_model = np.argmax(model_auc_array)
    return greatest_model


def take_out(X_train, X_test, gm):
    X_train_best = X_train[:, gm].reshape(-1, 1)
    X_test_best = X_test[:, gm].reshape(-1, 1)
    return X_train_best, X_test_best


def random_replicate(X_train, y_train, X_test, y_test):
    results_replicate = []
    pred = []
    pred_proba = []
    for i in range(10):
        X_train_level1, X_test_level1, y_train_true = get_stacking(X_train, y_train, X_test, y_test, random_seed=i)
        auc_report, y_pred, y_pred_proba = stacking_final_predict(X_train_level1, X_train, y_train_true, X_test_level1, X_test, y_test)
        results_replicate.append(auc_report)
    results_df = pd.DataFrame(results_replicate,
                              columns=['auc', 'legit_prec', 'legit_rec', 'legit_f1', 'legit_num', 'fraud_prec',
                                       'fraud_rec', 'fraud_f1', 'fraud_num'])
    pred_df = pd.DataFrame(pred)
    pred_proba_df = pd.DataFrame(pred_proba)
    return results_df, pred_df, pred_proba_df


def stacking_cm5(path, save_path):
    # 加载数据
    X_stf = pd.read_csv(os.path.join(path, 'stf_data.csv')).values
    X_test_stf = pd.read_csv(os.path.join(path, 'stf_test_data.csv')).values
    y = pd.read_csv(os.path.join(path, 'default.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(path, 'default_test.csv')).values.ravel()

    # 执行随机复制实验
    results_df, pred_df, pred_proba_df = random_replicate(X_stf, y, X_test_stf, y_test)

    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 保存结果到CSV文件
    results_df.to_csv(os.path.join(save_path, 'cm5_tfn.csv'), index=False)
    pred_df.to_csv(os.path.join(save_path, 'cm5_tfn_pred.csv'), index=False)
    pred_proba_df.to_csv(os.path.join(save_path, 'cm5_tfn_proba.csv'), index=False)



# example
#path = '../data/TFN'
#save_path = '../results/'
#stacking_cm5(path, save_path)