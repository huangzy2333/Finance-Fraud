import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

y_true = pd.read_csv('../data/TFN/default_test.csv')
y_true = y_true.values.flatten()

y_pred_baseline = pd.read_csv('../results/baseline_pred_proba.csv')
y_scores_lr = y_pred_baseline.iloc[-4].values.flatten()
y_scores_svm = y_pred_baseline.iloc[-3].values.flatten()
y_scores_xgboost = y_pred_baseline.iloc[-2].values.flatten()
y_scores_cnn = y_pred_baseline.iloc[-1].values.flatten()

y_pred_cm5 = pd.read_csv('../results/cm5_tfn_proba.csv')
y_scores_cm5 = y_pred_cm5.iloc[0].values.flatten()

y_pred_cm6 = pd.read_csv('../results/cm6_tfn_proba.csv')
y_scores_cm6 = y_pred_cm6.iloc[0].values.flatten()  #Choose the index of the best AUC model

fpr1, tpr1, thresholds1 = roc_curve(y_true, y_scores_lr)
auc1 = roc_auc_score(y_true, y_scores_lr)
fpr2, tpr2, thresholds2 = roc_curve(y_true, y_scores_svm)
auc2 = roc_auc_score(y_true, y_scores_svm)
fpr3, tpr3, thresholds3 = roc_curve(y_true, y_scores_xgboost)
auc3 = roc_auc_score(y_true, y_scores_xgboost)
fpr4, tpr4, thresholds4 = roc_curve(y_true, y_scores_cnn)
auc4 = roc_auc_score(y_true, y_scores_cnn)
fpr5, tpr5, thresholds5 = roc_curve(y_true, y_scores_cm5)
auc5 = roc_auc_score(y_true, y_scores_cm5)
fpr6, tpr6, thresholds6 = roc_curve(y_true, y_scores_cm6)
auc6 = roc_auc_score(y_true, y_scores_cm6)

plt.figure()
plt.title('ROC')
plt.plot(fpr_stf, tpr_four_models[0],'blue',label='CM4-LR, AUC=%0.2f' % auc1)
plt.plot(fpr_stf, tpr_four_models[1],'red',label='CM4-SVM, AUC=%0.2f' % auc2)
#plt.plot(fpr_stf, tpr_four_models[2],'yellow',label='CM4-GBDT, AUC=0.758')
plt.plot(fpr_stf, tpr_four_models[2],'green',label='CM4-XGBoost, AUC=%0.2f' % auc3)
plt.plot(fpr_stf, tpr_four_models[3],'purple',label='CM4-CNN, AUC=%0.2f' % auc4)
plt.plot(fpr_stf, tpr_stack_1, 'orange', label='CM5, AUC=%0.2f' % auc5)
plt.plot(fpr, tpr, 'black', label='CM6, AUC=%0.2f' % auc6)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.savefig("../results/roc_curve_stack.png")
plt.show()

