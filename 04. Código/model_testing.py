import numpy as np
import pandas as pd
from numpy import matlib
from scipy import stats
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from catboost import CatBoostClassifier
from sklearn.metrics import top_k_accuracy_score
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pickle


def sample_group(group, seed=12227):
    return group.sample(n=1000, replace=False, random_state=seed) if len(group) > 1000 else group


np.random.seed(12227)
k = 3
model_filename = './data/catboost_model.cbm'

# load prediction model
catboost_classifier = CatBoostClassifier()
catboost_classifier.load_model(model_filename)

estados = [
    "SP",
    "RS",
    'BA',
    'DF',
    'GO',
    'PA'
]

estados_completos_accuracy = []
estados_completos_top_3_accuracy = []

for estado in estados:

    # load data
    df = pd.read_pickle(f'./data/{estado}_treated_base_top_causes.pkl')

    # sampled_df = df.groupby('causabas_capitulo', group_keys=False,
    #                         as_index=False).apply(sample_group)
    # sampled_df.reset_index(drop=True, inplace=True)

    y_top_causes = df.causabas_capitulo
    X_top_causes = df.drop(columns=['causabas_capitulo'])
    X_top_causes.columns = X_top_causes.columns.str.lower()
    X_top_causes = X_top_causes.to_numpy()

    # y_top_causes_sampled_df = sampled_df.causabas_capitulo
    # X_top_causes_sampled_df = sampled_df.drop(columns=['causabas_capitulo'])
    # X_top_causes_sampled_df.columns = X_top_causes_sampled_df.columns.str.lower()
    # X_top_causes_sampled_df = X_top_causes_sampled_df.to_numpy()

    # load label encoder
    with open('./data/label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)

    y_encoded_top_causes = label_encoder.fit_transform(y_top_causes)
    # y_encoded_top_causes_sampled_df = label_encoder.fit_transform(
    #     y_top_causes_sampled_df)

    catboost_accuracy = catboost_classifier.score(
        X_top_causes, y_encoded_top_causes)
    catboost_y_pred_proba = catboost_classifier.predict_proba(X_top_causes)
    catboost_top_k_accuracy = top_k_accuracy_score(
        y_encoded_top_causes, catboost_y_pred_proba, k=k)
    print(f"CatBoost Accuracy: {catboost_accuracy*100:.2f}%")
    print(f"CatBoost Top-{k} Accuracy: {catboost_top_k_accuracy*100:.2f}%")

    estados_completos_accuracy.append(catboost_accuracy)
    estados_completos_top_3_accuracy.append(catboost_top_k_accuracy)

    # catboost_accuracy = catboost_classifier.score(
    #     X_top_causes_sampled_df, y_encoded_top_causes_sampled_df)
    # catboost_y_pred_proba = catboost_classifier.predict_proba(
    #     X_top_causes_sampled_df)
    # catboost_top_k_accuracy = top_k_accuracy_score(
    #     y_encoded_top_causes_sampled_df, catboost_y_pred_proba, k=k)
    # print(f"CatBoost Accuracy: {catboost_accuracy*100:.2f}%")
    # print(f"CatBoost Top-{k} Accuracy: {catboost_top_k_accuracy*100:.2f}%")

print(estados_completos_accuracy)
