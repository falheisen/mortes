import pandas as pd
import xgboost as xgb
import lightgbm as lgb
# from tabpfn import TabPFNClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import top_k_accuracy_score

X_train_all_reasons_all_features = pd.read_pickle(
    '../data/X_train_all_reasons_all_features.pkl')
X_test_all_reasons_all_features = pd.read_pickle(
    '../data/X_test_all_reasons_all_features.pkl')
y_train_all_reasons_all_features = pd.read_pickle(
    '../data/y_train_all_reasons_all_features.pkl')
y_test_all_reasons_all_features = pd.read_pickle(
    '../data/y_test_all_reasons_all_features.pkl')

X_train_all_reasons_all_features.shape
X_test_all_reasons_all_features.shape
num_classes = len(y_train_all_reasons_all_features.unique())
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_all_reasons_all_features)
y_test_encoded = label_encoder.transform(y_test_all_reasons_all_features)

sc = StandardScaler()
X_train = sc.fit_transform(X_train_all_reasons_all_features)
X_test = sc.transform(X_test_all_reasons_all_features)
k = 5

"""
existem 19 valores únicos de resposta, logo aleatório 
1/19 = 5,26%
"""

# Set num_class to the number of classes in your problem
xgb_classifier = xgb.XGBClassifier(
    objective='multi:softmax', num_class=num_classes)
xgb_classifier.fit(X_train, y_train_encoded)
accuracy = xgb_classifier.score(X_test, y_test_encoded)
y_pred = xgb_classifier.predict(X_test)
y_pred_proba = xgb_classifier.predict_proba(X_test)
top_k_accuracy = top_k_accuracy_score(y_test_encoded, y_pred_proba, k=k)
print(f"XGBoost Accuracy: {accuracy:.2f}")
print(f"XGBoost Top-{k} Accuracy: {top_k_accuracy:.2f}")
truth_table = pd.crosstab(index=y_test_encoded, columns=y_pred, rownames=[
                          'Real'], colnames=['Predição'], margins=True).to_excel('../03. Planilhas/tabela verdade xgboost.xlsx')
del xgb_classifier

print('')
print('')

# Set num_class to the number of classes in your problem
lgb_classifier = lgb.LGBMClassifier(
    objective='multiclass', num_class=num_classes)
lgb_classifier.fit(X_train, y_train_encoded)
accuracy = lgb_classifier.score(X_test, y_test_encoded)
y_pred_proba = lgb_classifier.predict_proba(X_test)
top_k_accuracy = top_k_accuracy_score(y_test_encoded, y_pred_proba, k=k)
print(f"LightGBM Accuracy: {accuracy:.2f}")
print(f"LightGBM Top-{k} Accuracy: {top_k_accuracy:.2f}")
del lgb_classifier

print('')
print('')

catboost_classifier = CatBoostClassifier(
    iterations=1000, depth=6, learning_rate=0.1, loss_function='MultiClass', custom_metric=['Accuracy'])
catboost_classifier.fit(X_train, y_train_encoded)
accuracy = catboost_classifier.score(X_test, y_test_encoded)
y_pred_proba = catboost_classifier.predict_proba(X_test)
k = 5
top_k_accuracy = top_k_accuracy_score(y_test_encoded, y_pred_proba, k=k)
print(f"CatBoost Accuracy: {accuracy:.2f}")
print(f"CatBoost Top-{k} Accuracy: {top_k_accuracy:.2f}")
del catboost_classifier

print('')
print('')

# tabPFN_classifier = TabPFNClassifier(device='gpu', N_ensemble_configurations=32)]
# tabPFN_classifier.fit(X_train, y_train_encoded)
# accuracy = tabPFN_classifier.score(X_test, y_test_encoded)
# y_pred_proba = tabPFN_classifier.predict_proba(X_test)
# top_k_accuracy = top_k_accuracy_score(y_test_encoded, y_pred_proba, k=k)
# print(f"TabPFN Accuracy: {accuracy:.2f}")
# print(f"TabPFN Top-{k} Accuracy: {top_k_accuracy:.2f}")
# del lgb_classifier
