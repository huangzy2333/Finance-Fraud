
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import cross_val_score, ShuffleSplit
#from sklearn.model_selection import StratifiedShuffleSplit
#from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pylab as plt 
from xgboost import plot_importance
#from imblearn.combine import SMOTEENN
#from imblearn.over_sampling import SMOTE
#from imblearn.over_sampling import RandomOverSampler, SMOTE
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
    cnn_model.add(keras.layers.Conv1D(256, 3, activation='tanh', input_shape=(shape, 1)))
    cnn_model.add(keras.layers.Dropout(0.2))
    cnn_model.add(keras.layers.Conv1D(128, 3, activation='tanh'))
    cnn_model.add(keras.layers.Dropout(0.2))
    cnn_model.add(keras.layers.Flatten())
    cnn_model.add(keras.layers.Dense(64, activation='tanh'))
    cnn_model.add(keras.layers.Dense(1, activation='sigmoid'))
    cnn_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return cnn_model


# cnn_model = keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model)

def get_models():
    models = dict()
    models['lr'] = LogisticRegression(solver='newton-cg', max_iter=500)
    # models['svm'] = LinearSVC(probability=True)
    models['svm'] = SVC(C=0.5, kernel='linear', probability=True)
    # models['gbdt'] = GradientBoostingClassifier(n_estimators=1000)
    models['xgb'] = XGBClassifier(booster='gblinear', learning_rate=0.01, use_label_encoder=False,
                                  eval_metric='logloss', n_estimators=500)
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


def get_stacking(X_train, y_train, X_test, y_test, random_seed):
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


def stacking_final_predict(X_train_level1, y_train_true, X_test_level1, y_test):
    auc_report = []
    # X_train, X_test = data_processing(X_train, X_test)
    # X_all_train = np.concatenate([X_train, X_train_level1], axis=1)
    # X_all_test = np.concatenate([X_test, X_test_level1], axis=1)
    X_all_train = X_train_level1
    X_all_test = X_test_level1
    # cv_1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # X_all_train, X_all_test = feature_select(LogisticRegression(), X_all_train, X_all_test, y)
    # for k, (train, valid) in enumerate(cv_1.split(X_all_train, y_train_true)):
    # final_estimator = LogisticRegression(C=0.5, solver='newton-cg', max_iter=500).fit(X_all_train, y_train_true)
    final_estimator = XGBClassifier(booster='gblinear', learning_rate=0.001, use_label_encoder=False,
                                    eval_metric='logloss', n_estimators=500).fit(X_all_train, y_train_true)
    # final_estimator = XGBClassifier(learning_rate=0.05, max_depth=8, n_estimators=500, use_label_encoder=False, eval_metric='logloss', random_state=3).fit(X_all_train[train], y_train_true[train])
    feature_map = '/Users/hwangsheep/PycharmProjects/fraud/remote/fusion_data/feature.fmap'
    # plot_importance(final_estimator,title='CBP feature importance', fmap=feature_map, importance_type='weight')

    plot_importance(final_estimator, title='Feature importance', max_num_features=4, fmap=feature_map,
                    importance_type='weight')
    plt.show()
    y_pred = final_estimator.predict(X_all_test)
    y_pred_proba = final_estimator.predict_proba(X_all_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    report_0 = list(report['0'].values())
    report_1 = list(report['1'].values())
    # report_0.extend(report_1)

    auc_report.append(auc)
    auc_report.extend(report_0)
    auc_report.extend(report_1)
    print(auc_report)
    return auc_report,y_pred, y_pred_proba


def model_select(X_train_level1, y_train_true):
    model_auc = []
    for i in range(4):
        y_pred_proba = X_train_level1[:, i]
        # y_pred = (y_pred_proba >= 0.5).astype(int)
        auc = roc_auc_score(y_train_true, y_pred_proba)
        # report = classification_report(y_train_true, y_pred, output_dict=True)
        model_auc.append(auc)
        # print("第 %d 个模型" % i)
        # print("auc: %f" % auc)
        # print(report['0'].values())
        # print(report['1'].values())
    model_auc_array = np.array(model_auc)
    greatest_model = np.argmax(model_auc_array)
    return greatest_model


def take_out(X_train, X_test, gm):
    X_train_best = X_train[:, gm].reshape(-1, 1)
    X_test_best = X_test[:, gm].reshape(-1, 1)
    return X_train_best, X_test_best




# data upload

def load_data(path):
    # 定义要加载的数据集名称
    X_names = ['s', 'f', 't', 'sf', 'st', 'ft', 'stf']

    # 加载训练数据集
    datasets = {name: pd.read_csv(f"{path}{name}_data.csv").values for name in X_names}

    # 加载测试数据集
    test_datasets = {f"{name}_test": pd.read_csv(f"{path}{name}_test_data.csv").values for name in X_names}

    # 加载标签数据
    y = pd.read_csv(f"{path}default.csv").values.ravel()
    y_test = pd.read_csv(f"{path}default_test.csv").values.ravel()

    return datasets, test_datasets, y, y_test


def random_replicate(X_f, X_test_f, X_t, X_test_t, X_ft, X_test_ft, X_s, X_test_s, X_sf, X_test_sf, X_st, X_test_st,
                     X_stf, X_test_stf, y, y_test):
    results_replicate = []
    pred = []
    pred_proba = []
    for i in range(10):
        X_train_level1_s, X_test_level1_s, y_train_true_s = get_stacking(X_s, y, X_test_s, y_test, random_seed=i)
        X_train_level1_f, X_test_level1_f, y_train_true_f = get_stacking(X_f, y, X_test_f, y_test, random_seed=i)
        X_train_level1_t, X_test_level1_t, y_train_true_t = get_stacking(X_t, y, X_test_t, y_test, random_seed=i)
        X_train_level1_ft, X_test_level1_ft, y_train_true_ft = get_stacking(X_ft, y, X_test_ft, y_test, random_seed=i)
        X_train_level1_sf, X_test_level1_sf, y_train_true_sf = get_stacking(X_sf, y, X_test_sf, y_test, random_seed=i)
        X_train_level1_st, X_test_level1_st, y_train_true_st = get_stacking(X_st, y, X_test_st, y_test, random_seed=i)
        X_train_level1_stf, X_test_level1_stf, y_train_true_stf = get_stacking(X_stf, y, X_test_stf, y_test,
                                                                               random_seed=i)
        gm_s = model_select(X_test_level1_s, y_test)
        gm_f = model_select(X_test_level1_f, y_test)
        gm_t = model_select(X_test_level1_t, y_test)
        gm_ft = model_select(X_test_level1_ft, y_test)
        gm_sf = model_select(X_test_level1_sf, y_test)
        gm_st = model_select(X_test_level1_st, y_test)
        gm_stf = model_select(X_test_level1_stf, y_test)
        best_train_level_s, best_test_level_s = take_out(X_train_level1_s, X_test_level1_s, gm_s)
        best_train_level_f, best_test_level_f = take_out(X_train_level1_f, X_test_level1_f, gm_f)
        best_train_level_t, best_test_level_t = take_out(X_train_level1_t, X_test_level1_t, gm_t)
        best_train_level_ft, best_test_level_ft = take_out(X_train_level1_ft, X_test_level1_ft, gm_ft)
        best_train_level_sf, best_test_level_sf = take_out(X_train_level1_sf, X_test_level1_sf, gm_sf)
        best_train_level_st, best_test_level_st = take_out(X_train_level1_st, X_test_level1_st, gm_st)
        best_train_level_stf, best_test_level_stf = take_out(X_train_level1_stf, X_test_level1_stf, gm_stf)
        X_train_level1_best = np.concatenate(
            [best_train_level_s, best_train_level_f, best_train_level_t, best_train_level_ft, best_train_level_sf,
             best_train_level_st, best_train_level_stf], axis=1)
        X_test_level1_best = np.concatenate(
            [best_test_level_s, best_test_level_f, best_test_level_t, best_test_level_ft, best_test_level_sf,
             best_test_level_st, best_test_level_stf], axis=1)

        # auc_best, report_best = stacking_final_predict(X_train_level1_best, X_stf, y_train_true_stf, X_test_level1_best, X_test_stf, y_test)
        auc_report_best, y_pred, y_pred_proba  = stacking_final_predict(X_train_level1_best, y_train_true_stf, X_test_level1_best, y_test)
        results_replicate.append(auc_report_best)
        pred.append(y_pred)
        pred_proba.append(y_pred_proba)
    results_df = pd.DataFrame(results_replicate,
                              columns=['auc', 'legit_prec', 'legit_rec', 'legit_f1', 'legit_num', 'fraud_prec',
                                       'fraud_rec', 'fraud_f1', 'fraud_num'])
    pred_df = pd.DataFrame(pred)
    pred_proba_df = pd.DataFrame(pred_proba)
    
    return results_df,pred_df, pred_proba_df

def stacking_cm6(data_path, save_path):

    datasets, test_datasets, y, y_test = load_data(data_path)
    X_s, X_test_s = datasets['s'], test_datasets['s_test']
    X_f, X_test_f = datasets['f'], test_datasets['f_test']
    X_t, X_test_t = datasets['t'], test_datasets['t_test']
    X_sf, X_test_sf = datasets['sf'], test_datasets['sf_test']
    X_st, X_test_st = datasets['st'], test_datasets['st_test']
    X_ft, X_test_ft = datasets['ft'], test_datasets['ft_test']
    X_stf, X_test_stf = datasets['stf'], test_datasets['stf_test']

    results_df,pred_df, pred_proba_df = random_replicate(X_f, X_test_f, X_t, X_test_t, X_ft, X_test_ft, X_s, X_test_s, X_sf, X_test_sf, X_st, X_test_st, X_stf, X_test_stf, y, y_test)

    results_df.to_csv(os.path.join(save_path, 'cm6_TFN_test.csv'), index=False)
    pred_df.to_csv(os.path.join(save_path, 'cm6_TFN_pred.csv'), index=False)
    pred_proba_df.to_csv(os.path.join(save_path, 'cm6_TFN_proba.csv'), index=False)


#data_path = '../data/TFN/'
#save_path = '../results/'
#stacking_cm6(data_path, save_path)