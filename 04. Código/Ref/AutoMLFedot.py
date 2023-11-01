import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.decomposition import PCA
from pprint import pprint
import autosklearn.classification
from fedot.api.main import Fedot

# if __name__ == '__main__':
random_state = 12227

# tmp = np.load('<>.npz')

# X_train = tmp['X_train']
# y_train = tmp['y_train']
# X_test = tmp['X_test']
# y_test = tmp['y_test']

X_train = pd.read_pickle('../../data/X_train1.pkl')
y_test = pd.read_pickle('../../data/y_train1.pkl')
X_test = pd.read_pickle('../../data/X_test1.pkl')
y_train = pd.read_pickle('../../data/y_test1.pkl') 

n_classes = len(np.unique(y_train))

cls = Fedot(problem='classification', timeout=5, preset='best_quality', n_jobs=-1)
cls.fit(features=X_train, target=y_train)
prediction = cls.predict(features=X_test)

y_pred = cls.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Accuracy score [{}]".format(acc))