from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import numpy as np
# from tabpfn import TabPFNClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import top_k_accuracy_score
import pickle

data_arrays = np.load(
    '../data/data_top_causes_all_features_all_states.npz', allow_pickle=True)
X_test = data_arrays['X_test']
X_train = data_arrays['X_train']
y_test = data_arrays['y_test']
y_train = data_arrays['y_train']

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
k = 3
num_classes = np.unique(y_train)
np.random.seed(12227)

with open('../data/standard_scaler.pkl', 'wb') as file:
    pickle.dump(sc, file)

"""
existem 19 valores únicos de resposta, logo aleatório 
1/19 = 5,26%
"""

#####################################################################################################
##
# CRIAR MODELOS
##
#####################################################################################################

# Set num_class to the number of classes in your problem
xgb_classifier = xgb.XGBClassifier(
    objective='multi:softmax', num_class=num_classes)
xgb_classifier.fit(X_train, y_train)
xgb_accuracy = xgb_classifier.score(X_test, y_test)
xgb_y_pred = xgb_classifier.predict(X_test)
xgb_y_pred_proba = xgb_classifier.predict_proba(X_test)
xgb_top_k_accuracy = top_k_accuracy_score(y_test, xgb_y_pred_proba, k=k)
print(f"XGBoost Accuracy: {xgb_accuracy*100:.2f}%")
print(f"XGBoost Top-{k} Accuracy: {xgb_top_k_accuracy*100:.2f}%")
matriz_confusao = pd.crosstab(index=y_test, columns=xgb_y_pred, rownames=[
    'Real'], colnames=['Predição'], margins=True).to_excel('../03. Planilhas/matriz confusao xgboost all states.xlsx')
model_filename = '../data/xgboost_model_all_states.txt'
xgb_classifier.save_model(model_filename)

print('')
print('')

# Set num_class to the number of classes in your problem
lgb_classifier = lgb.LGBMClassifier(
    objective='multiclass', num_class=num_classes)
lgb_classifier.fit(X_train, y_train)
lgb_accuracy = lgb_classifier.score(X_test, y_test)
lgb_y_pred_proba = lgb_classifier.predict_proba(X_test)
lgb_top_k_accuracy = top_k_accuracy_score(y_test, lgb_y_pred_proba, k=k)
print(f"LightGBM Accuracy: {lgb_accuracy*100:.2f}%")
print(f"LightGBM Top-{k} Accuracy: {lgb_top_k_accuracy*100:.2f}%")

print('')
print('')

catboost_classifier = CatBoostClassifier(
    iterations=1000, depth=6, learning_rate=0.1, loss_function='MultiClass', custom_metric=['Accuracy'])
catboost_classifier.fit(X_train, y_train)
catboost_accuracy = catboost_classifier.score(X_test, y_test)
catboost_y_pred_proba = catboost_classifier.predict_proba(X_test)
catboost_top_k_accuracy = top_k_accuracy_score(
    y_test, catboost_y_pred_proba, k=k)
print(f"CatBoost Accuracy: {catboost_accuracy*100:.2f}%")
print(f"CatBoost Top-{k} Accuracy: {catboost_top_k_accuracy*100:.2f}%")

print('')
print('')
