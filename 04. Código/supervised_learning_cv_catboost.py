import numpy as np
import catboost
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
# CatBoost
##
###############################################################################################################

# CatBoost with GPU support
catboost_classifier = catboost.CatBoostClassifier(task_type="CPU")

# BAYES SEARCH CV
param_space_catboost = {
    'learning_rate': (1e-5, 1),
    'random_strength': (1, 20),
    'l2_leaf_reg': (1, 10),
    'bagging_temperature': (0, 1.0),
    'leaf_estimation_iterations': (1, 20),
    'iterations': (100, 4000),
}

bayes_search_catboost = BayesSearchCV(
    catboost_classifier, param_space_catboost, n_iter=200, cv=kf, scoring='accuracy', n_jobs=12
)

# Record the start time
start_time_catboost = time.time()

# Fit the model with the data
bayes_search_catboost.fit(X, y)

# Record the end time
end_time_catboost = time.time()

# Calculate the elapsed time
elapsed_time_catboost = end_time_catboost - start_time_catboost

# Print or log the results
print("Time taken to fit the CatBoost model:",
      elapsed_time_catboost, "seconds")

joblib.dump(bayes_search_catboost,
            '../data/bayes_search_model_catboost_all_features.pkl')

# Print the best parameters found by the grid search
best_params = bayes_search_catboost.best_params_
print("Best Parameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")

# Print the best accuracy found by the grid search
best_accuracy = bayes_search_catboost.best_score_
print("Best Accuracy:", best_accuracy)
