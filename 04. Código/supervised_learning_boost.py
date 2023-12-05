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

data_arrays = np.load(
    '../data/data_top_causes_all_features.npz', allow_pickle=True)
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
    'Real'], colnames=['Predição'], margins=True).to_excel('../03. Planilhas/matriz confusao xgboost.xlsx')

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
model_filename = '../data/catboost_model.cbm'
catboost_classifier.save_model(model_filename)
print(f"Model saved to {model_filename}")

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

print('')
print('')

#####################################################################################################
##
# GRÁFICO DA EVOLUÇÃO DA PRECISÃO COM K
##
#####################################################################################################

# Initialize lists to store accuracies
xgb_accuracies = []
xgb_top_k_accuracies = []
lgb_accuracies = []
lgb_top_k_accuracies = []
catboost_accuracies = []
catboost_top_k_accuracies = []
ks = range(1, 12)

# Looping over k values
for k in range(1, 12):
    # Get predictions for XGBoost
    xgb_y_pred_proba = xgb_classifier.predict_proba(X_test)
    xgb_top_k_accuracy = top_k_accuracy_score(y_test, xgb_y_pred_proba, k=k)
    xgb_accuracies.append(xgb_classifier.score(X_test, y_test))
    xgb_top_k_accuracies.append(xgb_top_k_accuracy)

    # Get predictions for LightGBM
    lgb_y_pred_proba = lgb_classifier.predict_proba(X_test)
    lgb_top_k_accuracy = top_k_accuracy_score(y_test, lgb_y_pred_proba, k=k)
    lgb_accuracies.append(lgb_classifier.score(X_test, y_test))
    lgb_top_k_accuracies.append(lgb_top_k_accuracy)

    # Get predictions for CatBoost
    catboost_y_pred_proba = catboost_classifier.predict_proba(X_test)
    catboost_top_k_accuracy = top_k_accuracy_score(
        y_test, catboost_y_pred_proba, k=k)
    catboost_accuracies.append(catboost_classifier.score(X_test, y_test))
    catboost_top_k_accuracies.append(catboost_top_k_accuracy)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(range(1, 12), [acc * 100 for acc in xgb_accuracies],
         'b--', label='(Top-1) Accuracy')  # Convertendo para porcentagem
plt.plot(range(1, 12), [top_k_acc * 100 for top_k_acc in xgb_top_k_accuracies],
         'ro--', label='Top-k Accuracy')  # Convertendo para porcentagem
plt.axhline(y=100/11, color='gray', linestyle='--',
            label='Random Guess (1/11)')  # Linha para sorteio aleatório
plt.xticks(range(1, 12))  # Exibindo todos os valores de k no eixo x
plt.yticks(range(0, 101, 10))  # Escala do eixo y em intervalos de 10%
plt.xlabel('k')
plt.ylabel('Accuracy (%)')  # Alterando o rótulo do eixo y para porcentagem
plt.title('XGBoost Accuracy vs Top-k Accuracy')
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.legend()
plt.savefig('../02. Relatorio/top-k-accuracy.png', format='png', dpi=300)
plt.show()

# Assumindo que ks e as listas de top_k_accuracies já foram calculadas
plt.figure(figsize=(10, 6))
plt.plot(ks, [acc * 100 for acc in xgb_top_k_accuracies],
         marker='o', linestyle='--', label='XGBoost')  # Linha tracejada para XGBoost
plt.plot(ks, [acc * 100 for acc in lgb_top_k_accuracies],
         marker='s', linestyle='--', label='LightGBM')  # Linha tracejada para LightGBM
plt.plot(ks, [acc * 100 for acc in catboost_top_k_accuracies],
         marker='^', linestyle='--', label='CatBoost')  # Linha tracejada para CatBoost

plt.xlabel('Top-k')
plt.xticks(range(1, 12))  # Exibindo todos os valores de k no eixo x
# Alterado para incluir o símbolo de porcentagem
plt.ylabel('Top-k Accuracy (%)')
plt.title('Top-k Accuracy vs. k')
# Formata o eixo y para mostrar porcentagens
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.legend()
plt.grid(True)
plt.savefig('../02. Relatorio/top-k-accuracy-boost.png', format='png', dpi=300)
plt.show()
