import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import top_k_accuracy_score
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier

from infFS_S import infFS_S
import NeurIPS_2023 as coreset


tmp = np.load('data.npz')
X_train, X_test, y_train, y_test = tmp['X_train'], tmp['X_test'], tmp['y_train'], tmp['y_test']

y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)

y_train[np.where(y_train >= 9)[0]] = 9
y_test[np.where(y_test >= 9)[0]] = 9

n_classes = len(np.unique(y_train))
print('#Classes [{}]'.format(n_classes))

fs = infFS_S()
fs.fit(X_train, y_train)

X_train = fs.transform(X_train, top_n=100)
X_test = fs.transform(X_test, top_n=100)

sub_sampling = coreset.fdm_sum_pairwise(X_train, y_train, [5] * n_classes, eps=1e-5)
#sub_sampling = coreset.dm_sum_pairwise(euclidean_distances(X_train), 5*n_classes)
sub_sampling = [index for index_set in sub_sampling for index in index_set]
X_train = X_train[list(sub_sampling)]
y_train = y_train[list(sub_sampling)]

print('[{}] [{}]'.format(X_train.shape, y_train.shape))

#model = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)

#Required only when using non-TabPFN classifiers
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = XGBClassifier(n_estimators=32, max_depth=10, learning_rate=0.1, objective='binary:logistic')

model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)

#print('Accuracy {}'.format(accuracy_score(y_test, y_pred)))
for i in range(1, 5):
    top_k = top_k_accuracy_score(y_test, y_pred, k=i)
    print('Top {} Accuracy {}'.format(i, top_k))