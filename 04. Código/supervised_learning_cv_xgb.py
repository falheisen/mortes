import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from skopt import BayesSearchCV
import joblib
import time
import pickle
import json

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
# XGBoost
##
###############################################################################################################

# Create an XGBoost classifier
xgb_classifier = xgb.XGBClassifier(
    objective='multi:softmax', num_class=num_classes)

# GRID SEARCH CV

# # Define the parameter grid for XGBoost
# param_grid = {
#     'learning_rate': [1e-7, 0.01, 0.1, 0.2, 0.5, 1],
#     'max_depth': range(1, 11),
#     'subsample': [0.2, 0.4, 0.6, 0.8, 1],
#     'colsample_bytree': [0.2, 0.4, 0.6, 0.8, 1],
#     'colsample_bylevel': [0.2, 0.4, 0.6, 0.8, 1],
#     'min_child_weight': [1e-16, 1e-8, 1e-4, 1e-2, 1e0, 1e2, 1e5],
#     'alpha': [1e-16, 1e-8, 1e-4, 1e-2, 1e0, 1e2],
#     'lambda': [1e-16, 1e-8, 1e-4, 1e-2, 1e0, 1e2],
#     'gamma': [1e-16, 1e-8, 1e-4, 1e-2, 1e0, 1e2],
#     'n_estimators': range(100, 4001, 100)
# }

# # Initialize GridSearchCV
# grid_search = GridSearchCV(estimator=xgb_classifier,
#                            param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=12)

# # Fit the model with the data
# grid_search.fit(X, y)

# BAYES SEARCH CV

# Define the parameter search space
param_space = {
    'learning_rate': (1e-7, 1),
    'max_depth': (1, 10),
    'subsample': (0.2, 1),
    'colsample_bytree': (0.2, 1),
    'min_child_weight': (1e-16, 1e5),
    'alpha': (1e-16, 1e2),
    'lambda': (1e-16, 1e2),
    'gamma': (1e-16, 1e2),
    'n_estimators': (100, 4000)
}

# Create an XGBoost classifier
xgb_classifier = xgb.XGBClassifier(
    objective='multi:softmax', num_class=num_classes)

# Initialize BayesSearchCV
bayes_search = BayesSearchCV(
    xgb_classifier, param_space, n_iter=200, cv=kf, scoring='accuracy', n_jobs=12)

# Record the start time
start_time = time.time()

# Fit the model with the data
bayes_search.fit(X, y)

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print or log the results
print("Time taken to fit the model:", elapsed_time, "seconds")

joblib.dump(
    bayes_search, '../data/bayes_search_xgb_model_top_causes_all_features.pkl')

bayes_search = joblib.load(
    '../data/bayes_search_model_top_causes_all_features.pkl')

# Print the best parameters found by the grid search
best_params = bayes_search.best_params_
print("Best Parameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")

# Print the best accuracy found by the grid search
best_accuracy = bayes_search.best_score_
print("Best Accuracy:", best_accuracy)

# Access the best model
best_model = bayes_search.best_estimator_

# Initialize an empty list to store confusion matrices
conf_matrices = []

# Initialize an empty list to store accuracy scores
cv_accuracies = []

# Perform cross-validation with the best model
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    best_model.fit(X_train, y_train)

    predicted_labels = best_model.predict(X_test)

    conf_matrix = confusion_matrix(y_test, predicted_labels)

    conf_matrices.append(conf_matrix)

    accuracy = accuracy_score(y_test, predicted_labels)
    cv_accuracies.append(accuracy)

# Print cross-validated accuracy
print(
    f"Cross-Validated Accuracy: {np.mean(cv_accuracies):.4f} (±{np.std(cv_accuracies):.4f})")

# Print confusion matrices for each fold
for i, conf_matrix in enumerate(conf_matrices):
    print(
        f"Confusion Matrix - Fold {i + 1}:\nAccuracy: {cv_accuracies[i]}\n", conf_matrix)

with open('../data/label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

label_mapping = dict(
    zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

for key, value in label_mapping.items():
    print(f'{key}: {value}')
