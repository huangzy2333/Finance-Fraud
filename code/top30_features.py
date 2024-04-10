#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 23:52:34 2024

@author: hwangsheep
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
import matplotlib.pylab as plt 
from sklearn.preprocessing import StandardScaler
from xgboost import plot_importance
from imblearn.over_sampling import RandomOverSampler, SMOTE
#from imblearn.ensemble import EasyEnsembleClassifier
from tensorflow import keras

def data_processing(X_train, X_test):
    scaler = preprocessing.StandardScaler()
    train_scale = scaler.fit_transform(X_train)
    test_scale = scaler.transform(X_test)
    return train_scale, test_scale
# get a list of models to evaluate

# build cnn_model
def create_model(shape):
    cnn_model = keras.models.Sequential()
    cnn_model.add(keras.layers.Conv1D(64,3,activation='tanh',input_shape=(shape,1)))
    cnn_model.add(keras.layers.Dropout(0.2))
    cnn_model.add(keras.layers.Conv1D(32,3,activation='tanh'))
    cnn_model.add(keras.layers.Dropout(0.2))
    cnn_model.add(keras.layers.Flatten())
    cnn_model.add(keras.layers.Dense(16,activation='tanh'))
    cnn_model.add(keras.layers.Dense(1,activation='sigmoid'))    
    cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return cnn_model  


#cnn_model = keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model)

def get_models():
    models = dict()
    models['lr'] = LogisticRegression(max_iter=500)
    models['svm'] = SVC(kernel='linear', probability=True)
    #models['gbdt'] = GradientBoostingClassifier(n_estimators=1000)
    models['xgb'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=1000)
    #models['cnn'] = "models"
    return models

def feature_select(models, X, X_test, y):
    selector = SelectFromModel(estimator=models).fit(X, y)
    X_train_select = selector.transform(X)
    X_test_select = selector.transform(X_test)
    return X_train_select, X_test_select

def kfold_train_base(name, models, X, y, X_test, y_test, cv):
    auc_set = []
    report_set = []
    X, X_test = data_processing(X, X_test) 
    if name == 'cnn':       
        for k, (train, valid) in enumerate(cv.split(X, y)):
            X_reshape = X.reshape(X.shape[0],X.shape[1],1)
            X_test_reshape = X_test.reshape(X_test.shape[0],X_test.shape[1],1) 
            model = create_model(X.shape[1])
            model.fit(X_reshape[train], y[train], batch_size=1, epochs=10)
            test_pred_proba_y = model.predict(X_test_reshape).flatten()
            test_pred_y = (test_pred_proba_y >= 0.5).astype(int)
            auc = roc_auc_score(y_test, test_pred_proba_y)
            report = classification_report(y_test, test_pred_y, output_dict=True)
            report_0 = list(report['0'].values())
            report_1 = list(report['1'].values())
            report_0.extend(report_1)
            auc_set.append(auc)
            report_set.append(report_0)
    else:
        #X, X_test = feature_select(models, X, X_test, y)
        for k, (train, valid) in enumerate(cv.split(X, y)):
            model = models.fit(X[train], y[train])
            
            test_pred_proba_y = model.predict_proba(X_test)[:,1]
            test_pred_y = model.predict(X_test)
            auc = roc_auc_score(y_test, test_pred_proba_y)
            report = classification_report(y_test, test_pred_y, output_dict=True)
            report_0 = list(report['0'].values())
            report_1 = list(report['1'].values())
            report_0.extend(report_1)
            auc_set.append(auc)
            report_set.append(report_0)

    auc_mean = np.mean(auc_set)
    report_mean = np.mean(report_set, axis=0)
    auc_report = np.insert(report_mean, 0, auc_mean).tolist()
    return auc_report,model
    
def model_set(X_train, y_train, X_test, y_test):
    
    model_results = []
    features = []
    models = get_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    for name, models in models.items():
        auc_report,model = kfold_train_base(name=name, models=models, X=X_train, y=y_train, X_test=X_test, y_test=y_test, cv=cv)
        model_results.append(auc_report)
        if name == 'lr' or name == 'svm':
            importances = np.abs(model.coef_[0])
        elif name == 'gbdt' or name == 'xgb':
            importances = model.feature_importances_
        features.append(importances)
            
    print(model_results)
    return np.array(model_results),features

path = '/Users/hwangsheep/PycharmProjects/fraud/remote/fusion120/TFN/'


X_stf = pd.read_csv(path + 'stf_data.csv')

X_test_stf = pd.read_csv(path + 'stf_test_data.csv')

y = pd.read_csv(path + 'default.csv')
y_test = pd.read_csv(path + 'default_test.csv')


X_stf, X_test_stf = X_stf.values, X_test_stf.values
y = y.values.ravel()
y_test = y_test.values.ravel()


results_stf,features = model_set(X_stf, y, X_test_stf, y_test)
features = pd.DataFrame(features)
# 对每行进行归一化
normalized_df = features.div(features.sum(axis=1), axis=0)

# 对每列进行平均
column_means = normalized_df.mean(axis=0)

# 根据平均值对列进行排序
sorted_columns = column_means.sort_values(ascending=False)

# 取排完序以后的前30列
top_30_sorted_columns = sorted_columns.head(30)

# 选择排完序以后的前30列
top_30_normalized_df = normalized_df[top_30_sorted_columns.index]
'''
for column_index in top_30_normalized_df.columns:
    new_index = column_index + 1  # 将当前索引加 1
    top_30_normalized_df.rename(columns={column_index: new_index}, inplace=True)  # 修改列索引的值
'''

# 定义一个函数来更改列索引
def change_column_index(i):
    if i < 24:
        return f'S{i+1}'
    elif 24 <= i < 103:
        return f'T{i+1}'
    else:
        return f'F{i+1}'
    
top_30_normalized_df.columns = [change_column_index(int(col)) for col in top_30_normalized_df.columns] 
# 使用建议的颜色以符合Nature期刊风格 
colors = {'S': '#E89DA0', 'T': '#88CEE6', 'F': '#B2D3A4'} 
bar_colors = [colors[col[0]] for col in top_30_normalized_df.columns] 

# 绘制特征重要性图 
plt.figure(figsize=(12, 8)) 
bars = plt.bar(top_30_normalized_df.columns, top_30_normalized_df.mean(), color=bar_colors)
# 创建自定义图例 
from matplotlib.patches import Patch 
legend_elements = [Patch(facecolor=colors['S'], label='Satellite Features'), 
                   Patch(facecolor=colors['T'], label='Textual Features'), 
                   Patch(facecolor=colors['F'], label='Financial Features')] 
plt.legend(handles=legend_elements) 
plt.title('CM4 Top 30 Feature Importance') 
plt.xlabel('Features') 
plt.ylabel('Mean Importance') 
plt.xticks(rotation=45) 
plt.show()
