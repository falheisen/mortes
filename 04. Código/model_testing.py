import xgboost as xgb
import numpy as np
import pandas as pd
from numpy import matlib
from scipy import stats
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.metrics import top_k_accuracy_score
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.preprocessing import StandardScaler
import pickle


def sample_group(group, seed=12227):
    return group.sample(n=1000, replace=False, random_state=seed) if len(group) > 1000 else group


num_classes = 11
np.random.seed(12227)

# load label encoder
with open('../data/label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# load model
model_filename = '../data/xgboost_model_all_states.txt'
xgb_classifier = xgb.XGBClassifier(
    objective='multi:softmax', num_class=num_classes)
xgb_classifier.load_model(model_filename)

with open('../data/standard_scaler.pkl', 'rb') as file:
    sc = pickle.load(file)

estados = [
    "SP",
    "RS",
    'PE',
    'GO',
    'PA'
]

estados_completos_accuracy = []
estados_completos_top_k_accuracy = np.zeros((5, num_classes))
i = 0

for estado in estados:

    # load data
    df = pd.read_pickle(
        f'../data/{estado}_treated_base_top_causes_all_states.pkl')

    # print(df.shape)
    y_top_causes = df.causabas_capitulo
    X_top_causes = df.drop(columns=['causabas_capitulo'])
    X_top_causes.columns = X_top_causes.columns.str.lower()
    X_top_causes = X_top_causes.to_numpy()

    X_top_causes = sc.transform(X_top_causes)

    y_encoded_top_causes = label_encoder.transform(y_top_causes)

    # print(X_top_causes.shape)
    # print(y_encoded_top_causes.shape)

    xgb_accuracy = xgb_classifier.score(X_top_causes, y_encoded_top_causes)
    xgb_classifier.predict(X_top_causes)
    xgb_y_pred_proba = xgb_classifier.predict_proba(X_top_causes)
    for k in range(num_classes+1):
        xgb_top_k_accuracy = top_k_accuracy_score(
            y_encoded_top_causes, xgb_y_pred_proba, k=k)
        if k == 3:
            print(f"XGBoost Top-{k} Accuracy: {xgb_top_k_accuracy*100:.2f}%")
        estados_completos_top_k_accuracy[i][k-1] = xgb_top_k_accuracy
    print(f"XGBoost Accuracy: {xgb_accuracy*100:.2f}%")

    estados_completos_accuracy.append(xgb_accuracy)
    i += 1

    del df, y_top_causes, X_top_causes, y_encoded_top_causes, xgb_y_pred_proba, xgb_accuracy, xgb_top_k_accuracy

# Plotting
plt.figure(figsize=(10, 6))
for i in range(len(estados)):
    plt.plot(range(1, 12), [top_k_acc * 100 for top_k_acc in estados_completos_top_k_accuracy[i]],
             'o--', label=estados[i])  # Convertendo para porcentagem
plt.axhline(y=100/11, color='gray', linestyle='--',
            label='Random Guess (1/11)')  # Linha para sorteio aleatório
plt.xticks(range(1, 12))  # Exibindo todos os valores de k no eixo x
plt.yticks(range(0, 101, 10))  # Escala do eixo y em intervalos de 10%
plt.xlabel('k')
plt.ylabel('Accuracy (%)')  # Alterando o rótulo do eixo y para porcentagem
plt.title('Top-k Accuracy in Different States')
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.legend()
plt.savefig('../02. Relatorio/top-k-accuracy-all-states.png',
            format='png', dpi=300)
plt.show()
