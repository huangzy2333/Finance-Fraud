#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:37:17 2024

@author: hwangsheep
"""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix, roc_curve,roc_auc_score,auc
from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt
# 假设y_true是真实标签，y_scores是模型预测为正类的概率 
y_pred = pd.read_csv('../results/cm6_tfn_proba.csv')
y_true = pd.read_csv('../data/TFN/default_test.csv')

y_scores = y_pred.iloc[0].values.flatten()  #Choose the index of the best AUC model
y_true = y_true.values.flatten()

pm = 0.5

# 初始化变量来存储每种成本设置下的FPR和TPR值
fpr_list = [0]
tpr_list = [0]

# 遍历不同的成本比例
for cost_fn in np.linspace(1, 100, 991):
    cost_fp = 1  # 假阳性成本固定为1
    min_cost = float('inf')
    best_threshold = 0

    # 遍历不同的阈值以找到最小成本
    for threshold in np.linspace(0, 1, 101):
        y_pred = (y_scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # 计算总成本 
        p1 = fn/(tp+fn)
        p2 = fp/(fp+tn)
        total_cost = pm * p1 * cost_fn + (1-pm) * p2 * cost_fp
        
        if total_cost < min_cost:
            min_cost = total_cost
            best_threshold = threshold
    
    # 使用找到的最优阈值重新计算FPR和TPR
    y_pred_optimal = (y_scores >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_optimal).ravel()
    fpr_optimal = fp / (fp + tn)
    tpr_optimal = tp / (tp + fn)
    
    fpr_list.append(fpr_optimal)
    tpr_list.append(tpr_optimal)

fpr_list.append(1)
tpr_list.append(1)
# 计算AUC
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
roc_auc1 = auc(fpr_list, tpr_list)

print(fpr_list, tpr_list)
# 绘制ROC曲线
plt.figure()
#plt.plot(fpr, tpr, 'black',label='CM6, AUC=%0.2f' % roc_auc)
plt.plot(fpr_list, tpr_list, '#403990',linewidth=1.5,label='CM6, AUC=%0.2f' % roc_auc1)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.03, 1.03])
plt.ylim([-0.03, 1.03])
plt.xlabel('FPR')
plt.ylabel('TPR')
#plt.title('ROC Curves Using Varying Cost Settings')
plt.legend(loc="lower right")
plt.show()
