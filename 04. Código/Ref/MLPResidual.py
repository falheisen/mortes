# Author: Artur Jordao <arturjlcorreia@gmail.com>
from tensorflow.keras import *
from keras.layers import *
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score
import numpy as np
from sklearn.preprocessing import StandardScaler


def res_block(inputs, norm_type, activation, dropout, projection_dim):
    """Residual block of MLP."""

    norm = (
        layers.LayerNormalization
        if norm_type == 'L'
        else layers.BatchNormalization
    )

    x = norm()(inputs)
    x = layers.Dense(projection_dim, activation=activation)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1], activation=activation)(x)
    res = x + inputs

    return res


def MLPResidual(
        input_shape,
        n_classes,
        norm_type='L',
        activation='relu',
        n_block=3,
        dropout=0,
        projection_dim=128):
    """Build MLP-Residual model."""

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for _ in range(n_block):
        x = res_block(x, norm_type, activation, dropout, projection_dim)

    x = layers.Flatten()(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)


# if __name__ == '__main__':
np.random.seed(12227)

# n_samples, n_features, n_classes = 1000, 300, 13
# X = np.random.rand(n_samples, n_features)
# X = np.expand_dims(X, axis=1)
# y = np.random.randint(0, n_classes, size=n_samples)
# y = np.eye(n_classes)[y]
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.33, random_state=12227)

k = 3
tmp = np.load('../../data/data_top_causes_all_features.npz')
X_train, X_test, y_train, y_test = tmp['X_train'], tmp['X_test'], tmp['y_train'], tmp['y_test']
n_classes = len(np.unique(y_test))
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

input_shape = X_train.shape[1:]
projection_dim = 64
num_heads = [8]*5
MLPResidual_classifier = MLPResidual(
    input_shape, n_classes, projection_dim=projection_dim)

MLPResidual_classifier.compile(optimizer='Adam',
                               loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy',
#               optimizer='Adam', metrics=['accuracy'])
MLPResidual_classifier.fit(
    X_train, y_train, epochs=100, batch_size=64, verbose=2)
y_pred = MLPResidual_classifier.predict(X_test)
loss, accuracy = MLPResidual_classifier.evaluate(X_test, y_test, verbose=0)
MLPResidual_top_k_accuracy = top_k_accuracy_score(
    y_test, y_pred, k=k)

print(f"MLP Residual Accuracy: {accuracy*100:.2f}%")
print(f"MLP Residual Top-{k} Accuracy: {MLPResidual_top_k_accuracy*100:.2f}%")

MLPResidual_top_k_accuracies = []

for k in range(1, 12):
    MLPResidual_top_k_accuracy = top_k_accuracy_score(
        y_test, y_pred, k=k)
    MLPResidual_top_k_accuracies.append(MLPResidual_top_k_accuracy)
