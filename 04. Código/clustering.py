### LIBRARIES

import pandas as pd
import numpy as np

df = pd.read_pickle('../data/SP_treated_base.pkl')


## IDENTIFICANDO COLUNAS DE CATEGORIAS

import time
start = time.time()
kmeans = KMeans(n_clusters=4)
kmeans.fit(one_hot_encoded_data)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
print("Cluster Labels:", labels)
print("Cluster Centroids:", centroids)
end = time.time()
print(f'{end-start}')