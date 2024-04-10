#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 23:52:34 2024

@author: hwangsheep
"""
import os
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
from xgboost import plot_importance
from imblearn.over_sampling import RandomOverSampler, SMOTE
#from imblearn.ensemble import EasyEnsembleClassifier
from tensorflow import keras
from tensorflow import random

def data_processing(X_train, X_test):
    scaler = preprocessing.StandardScaler()
    train_scale = scaler.fit_transform(X_train)
    test_scale = scaler.transform(X_test)
    return train_scale, test_scale
# get a list of models to evaluate

# build cnn_model
def data_processing(X_train, X_test):
    scaler = preprocessing.StandardScaler()
    train_scale = scaler.fit_transform(X_train)
    test_scale = scaler.transform(X_test)
    return train_scale, test_scale
# get a list of models to evaluate

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


def kfold_train_base(name, models, X, y, X_test, y_test, cv):
    auc_set = []
    report_set = []
    pred_proba_set = []  # 用于收集预测概率
    pred_set = []  # 用于收集预测值
    X, X_test = data_processing(X, X_test)

    if name == 'cnn':
        for k, (train, valid) in enumerate(cv.split(X, y)):
            X_reshape = X.reshape(X.shape[0], X.shape[1], 1)
            X_test_reshape = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
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
            pred_proba_set.append(test_pred_proba_y)  # 添加预测概率
            pred_set.append(test_pred_y)  # 添加预测值
    else:
        X, X_test = feature_select(models, X, X_test, y)
        for k, (train, valid) in enumerate(cv.split(X, y)):
            model = models.fit(X[train], y[train])
            test_pred_proba_y = model.predict_proba(X_test)[:, 1]
            test_pred_y = model.predict(X_test)
            auc = roc_auc_score(y_test, test_pred_proba_y)
            report = classification_report(y_test, test_pred_y, output_dict=True)
            report_0 = list(report['0'].values())
            report_1 = list(report['1'].values())
            report_0.extend(report_1)
            auc_set.append(auc)
            report_set.append(report_0)
            pred_proba_set.append(test_pred_proba_y)  # 添加预测概率
            pred_set.append(test_pred_y)  # 添加预测值

    auc_mean = np.mean(auc_set)
    report_mean = np.mean(report_set, axis=0)
    auc_report = np.insert(report_mean, 0, auc_mean).tolist()

    # 汇总预测值和预测概率
    final_pred_proba = np.mean(pred_proba_set, axis=0)
    final_pred = np.round(final_pred_proba).astype(int)

    return auc_report, final_pred, final_pred_proba


def model_set(X_train, y_train, X_test, y_test):
    model_results = []
    pred_results = []
    pred_proba_results = []
    models = get_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    for name, model in models.items():
        auc_report, preds, pred_probas = kfold_train_base(name=name, models=model, X=X_train, y=y_train, X_test=X_test,
                                                          y_test=y_test, cv=cv)
        model_results.append(auc_report)
        pred_results.append(preds)
        pred_proba_results.append(pred_probas)

    return model_results, pred_results, pred_proba_results



def load_all_datasets(data_path):
    # 定义数据集名称
    X_names = ['s', 'f', 't', 'sf', 'st', 'ft', 'stf']

    # 加载数据集
    X_datasets = {name: pd.read_csv(f"{data_path}{name}_data.csv").values for name in X_names}
    X_test_datasets = {f"{name}_test": pd.read_csv(f"{data_path}{name}_test_data.csv").values for name in X_names}

    y = pd.read_csv(f"{data_path}default.csv").values.ravel()
    y_test = pd.read_csv(f"{data_path}default_test.csv").values.ravel()

    return X_datasets, X_test_datasets, y, y_test


def run_models_and_collect_results(X_datasets, X_test_datasets, y, y_test):
    all_model_results = []
    all_pred_results = []
    all_pred_proba_results = []

    for name, X_train in X_datasets.items():
        print(f"Processing dataset: {name}")
        X_test = X_test_datasets[f"{name}_test"]

        model_results, pred_results, pred_proba_results = model_set(X_train, y, X_test, y_test)

        all_model_results.extend(model_results)
        all_pred_results.extend(pred_results)
        all_pred_proba_results.extend(pred_proba_results)

    return all_model_results, all_pred_results, all_pred_proba_results


def save_results(all_model_results, all_pred_results, all_pred_proba_results, results_path):
    # 确保保存路径存在
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # 保存所有数据集的模型结果
    results_df = pd.DataFrame(all_model_results,
                              columns=['auc', 'legit_prec', 'legit_rec', 'legit_f1', 'legit_num', 'fraud_prec',
                                       'fraud_rec', 'fraud_f1', 'fraud_num'])
    results_df.to_csv(f"{results_path}baseline_tfn.csv", index=False)

    # 保存所有数据集的预测值
    pred_df = pd.DataFrame(all_pred_results)
    pred_df.to_csv(f"{results_path}baseline_pred.csv", index=False)

    # 保存所有数据集的预测概率
    pred_proba_df = pd.DataFrame(all_pred_proba_results)
    pred_proba_df.to_csv(f"{results_path}baseline_pred_proba.csv", index=False)

def baseline_models(data_path, results_path):

    X_datasets, X_test_datasets, y, y_test = load_all_datasets(data_path)  # 假设这个函数负责加载所有数据
    all_model_results, all_pred_results, all_pred_proba_results = run_models_and_collect_results(X_datasets,
                                                                                             X_test_datasets, y, y_test)
    save_results(all_model_results, all_pred_results, all_pred_proba_results, results_path)

# 示例用法
#data_path = '../data/TFN/'
#results_path = '../results/'
#baseline_models(data_path, results_path)