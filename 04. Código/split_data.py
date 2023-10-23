### LIBRARIES

import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_pickle('../data/SP_treated_base.pkl')
target1 = df.causabas_capitulo
target2 = df.causabas_grupo
df = df.drop(columns=['causabas_capitulo','causabas_grupo'])
X_train1, X_test1, y_train1, y_test1 = \
        train_test_split(
            df, 
            target1, 
            train_size=0.80, 
            test_size=0.2, 
            random_state=42
        )

X_train2, X_test2, y_train2, y_test2 = \
        train_test_split(
            df, 
            target2, 
            train_size=0.80, 
            test_size=0.2, 
            random_state=42
        )

X_train1.columns = X_train1.columns.str.lower()
X_test1.columns = X_test1.columns.str.lower()

# Save the variables to binary files using pickle
with open('../data/X_train1.pkl', 'wb') as f:
    pickle.dump(X_train1, f)

with open('../data/X_test1.pkl', 'wb') as f:
    pickle.dump(X_test1, f)

with open('../data/y_train1.pkl', 'wb') as f:
    pickle.dump(y_train1, f)

with open('../data/y_test1.pkl', 'wb') as f:
    pickle.dump(y_test1, f)

X_train2.columns = X_train2.columns.str.lower()
X_test2.columns = X_test2.columns.str.lower()

# Save the variables to binary files using pickle
with open('../data/X_train2.pkl', 'wb') as f:
    pickle.dump(X_train2, f)

with open('../data/X_test2.pkl', 'wb') as f:
    pickle.dump(X_test2, f)

with open('../data/y_train2.pkl', 'wb') as f:
    pickle.dump(y_train2, f)

with open('../data/y_test2.pkl', 'wb') as f:
    pickle.dump(y_test2, f)

# ## IDENTIFICANDO COLUNAS DE CATEGORIAS

# import time
# start = time.time()
# kmeans = KMeans(n_clusters=4)
# kmeans.fit(one_hot_encoded_data)
# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_
# print("Cluster Labels:", labels)
# print("Cluster Centroids:", centroids)
# end = time.time()
# print(f'{end-start}')