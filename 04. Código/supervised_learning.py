import pandas as pd

X_train1 = pd.read_pickle('../data/X_train1.pkl')
X_test1 = pd.read_pickle('../data/X_test1.pkl') 
y_train1 = pd.read_pickle('../data/y_train1.pkl')
y_test1 = pd.read_pickle('../data/y_test1.pkl') 

X_train1.shape
X_test1.shape

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train1, y_train1)
dt.score(X_test1, y_test1)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50)
clf.fit(X_train1, y_train1)
clf.score(X_test1, y_test1)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs')
lr.fit(X_train1, y_train1)
lr.score(X_test1, y_test1)

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train1, y_train1)
mnb.score(X_test1, y_test1)

from sklearn.ensemble import GradientBoostingClassifier

gbc40 = GradientBoostingClassifier(n_estimators=200, max_depth=40)
gbc40.fit(X_train1, y_train1)
gbc40.score(X_test1, y_test1)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
modele_one_vs_linear_SVC= OneVsRestClassifier(LinearSVC())
modele_one_vs_linear_SVC.fit(X_train1, y_train1)
modele_one_vs_linear_SVC.score(X_test1, y_test1)

from sklearn.svm import SVC
modele_one_vs_SVC = OneVsRestClassifier(SVC())
modele_one_vs_SVC.fit(X_train1, y_train1)
modele_one_vs_SVC.score(X_test1, y_test1)