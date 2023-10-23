import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

X_train1 = pd.read_pickle('../data/X_train1.pkl')
X_test1 = pd.read_pickle('../data/X_test1.pkl') 
y_train1 = pd.read_pickle('../data/y_train1.pkl')
y_test1 = pd.read_pickle('../data/y_test1.pkl') 

X_train1.shape
X_test1.shape
num_classes = len(y_train1.unique())
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train1)
y_test_encoded = label_encoder.transform(y_test1)

"""
existem 19 valores únicos de resposta, logo aleatório 
1/19 = 5,26%
""" 

xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=num_classes)  # Set num_class to the number of classes in your problem
xgb_classifier.fit(X_train1, y_train_encoded)
accuracy = xgb_classifier.score(X_test1, y_test_encoded)
print("XGBoost Accuracy:", accuracy)
del xgb_classifier

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# X_train1_adj = scaler.fit_transform(X_train1)
# X_test1_adj = scaler.transform(X_test1)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train1, y_train1)
print(dt.score(X_test1, y_test1)) # 0.3027334293948127
del dt

# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=50)
# clf.fit(X_train1, y_train1)
# print(clf.score(X_test1, y_test1)) # 0.06333789625360231
# # ABNORMAL_TERMINATION_IN_LNSRCH. 
# del clf

# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(solver='lbfgs')
# lr.fit(X_train1, y_train1)
# lr.score(X_test1, y_test1)
# del lr

# from sklearn.naive_bayes import MultinomialNB
# mnb = MultinomialNB()
# mnb.fit(X_train1_adj, y_train1)
# mnb.score(X_test1_adj, y_test1)
# # ValueError: Negative values in data passed to MultinomialNB (input X)
# del mnb

# from sklearn.ensemble import GradientBoostingClassifier
# gbc40 = GradientBoostingClassifier(n_estimators=200, max_depth=40)
# gbc40.fit(X_train1, y_train1)
# print(gbc40.score(X_test1, y_test1))
# # MemoryError: Unable to allocate 805. MiB for an array with shape (5552000, 19) and data type float64

# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import LinearSVC
# modele_one_vs_linear_SVC= OneVsRestClassifier(LinearSVC())
# modele_one_vs_linear_SVC.fit(X_train1, y_train1)
# print(modele_one_vs_linear_SVC.score(X_test1, y_test1))

# from sklearn.svm import SVC
# modele_one_vs_SVC = OneVsRestClassifier(SVC())
# modele_one_vs_SVC.fit(X_train1, y_train1)
# print(modele_one_vs_SVC.score(X_test1, y_test1))