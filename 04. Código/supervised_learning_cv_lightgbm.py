import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from skopt import BayesSearchCV
import joblib
import time
import pickle

# Load data
data_arrays = np.load(
    '../data/data_cv_top_causes_all_features.npz', allow_pickle=True)
X = data_arrays['X']
y = data_arrays['y']

num_classes = len(np.unique(y))
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

###############################################################################################################
##
# LightGBM
##
###############################################################################################################

# LightGBM classifier
lgbm_classifier = lgb.LGBMClassifier()

# BAYES SEARCH CV
param_space_lightgbm = {
    'num_leaves': (5, 50),
    'max_depth': (3, 20),
    'learning_rate': (0.001, 1),
    'n_estimators': (50, 2000),
    'min_child_weight': (0.00001, 10000),
    'reg_alpha': (0, 100),
    'reg_lambda': (0, 100),
    'subsample': (0.2, 0.8)
}

bayes_search_lightgbm = BayesSearchCV(
    lgbm_classifier, param_space_lightgbm, n_iter=200, cv=kf, scoring='accuracy', n_jobs=12
)

# Record the start time
start_time_lightgbm = time.time()

# Fit the model with the data
bayes_search_lightgbm.fit(X, y)

# Record the end time
end_time_lightgbm = time.time()

# Calculate the elapsed time
elapsed_time_lightgbm = end_time_lightgbm - start_time_lightgbm

# Print or log the results
print("Time taken to fit the LightGBM model:",
      elapsed_time_lightgbm, "seconds")

joblib.dump(bayes_search_lightgbm,
            '../data/bayes_search_model_lightgbm_all_features.pkl')

# Print the best parameters found by the grid search
best_params = bayes_search_lightgbm.best_params_
print("Best Parameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")

# Print the best accuracy found by the grid search
best_accuracy = bayes_search_lightgbm.best_score_
print("Best Accuracy:", best_accuracy)
