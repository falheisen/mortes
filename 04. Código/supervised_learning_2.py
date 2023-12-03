# Author: Artur Jordao <arturjlcorreia@gmail.com>
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import *
from tensorflow.keras import *
import keras
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import top_k_accuracy_score

#####################################################################################################
##
# CARREGAR DADOS
##
#####################################################################################################

data_arrays = np.load(
    './data/data_top_causes_all_features.npz', allow_pickle=True)
X_test = data_arrays['X_test']
X_train = data_arrays['X_train']
y_test = data_arrays['y_test']
y_train = data_arrays['y_train']

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train_T = np.expand_dims(X_train, axis=1)
X_test_T = np.expand_dims(X_test, axis=1)

k = 3
num_classes = np.unique(y_train)
n_classes = len(num_classes)
np.random.seed(12227)
input_shape = X_train.shape[1:]
input_shape_T = X_train_T.shape[1:]
projection_dim = 64
num_heads = [8]*5

#####################################################################################################
##
# FUNÇÕES AUXILIARES
##
#####################################################################################################


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


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=self.projection_dim)

        # if weights is not None:
        #     self.projection = layers.Dense(units=projection_dim, weights=weights)

        self.position_embedding = layers.Embedding(
            input_dim=num_patches,
            output_dim=self.projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim
        })
        return config


def Transformer(input_shape, projection_dim, num_heads, n_classes):

    inputs = layers.Input(shape=input_shape)
    encoded_patches = PatchEncoder(input_shape[0], projection_dim)(inputs)

    num_transformer_blocks = len(num_heads)
    for i in range(num_transformer_blocks):

        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads[i], key_dim=projection_dim, dropout=0.0)(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP Size of the transformer layers
        transformer_units = [projection_dim * 2, projection_dim]

        x3 = layers.Dense(projection_dim * 2, activation=tf.nn.gelu)(x3)
        x3 = layers.Dense(projection_dim, activation=tf.nn.gelu)(x3)

        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    encoded_patches = layers.Flatten()(encoded_patches)
    outputs = layers.Dense(n_classes, activation='softmax')(encoded_patches)

    return keras.Model(inputs, outputs)


#####################################################################################################
##
# CRIAR MODELOS
##
#####################################################################################################

# XGBoost

xgb_classifier = xgb.XGBClassifier(
    objective='multi:softmax', num_class=num_classes)
xgb_classifier.fit(X_train, y_train)
xgb_accuracy = xgb_classifier.score(X_test, y_test)
xgb_y_pred = xgb_classifier.predict(X_test)
xgb_y_pred_proba = xgb_classifier.predict_proba(X_test)
xgb_top_k_accuracy = top_k_accuracy_score(y_test, xgb_y_pred_proba, k=k)
print(f"XGBoost Accuracy: {xgb_accuracy*100:.2f}%")
print(f"XGBoost Top-{k} Accuracy: {xgb_top_k_accuracy*100:.2f}%")

# MLPResidual

MLPResidual_classifier = MLPResidual(
    input_shape, n_classes, projection_dim=projection_dim)

MLPResidual_classifier.compile(optimizer='Adam',
                               loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy',
#               optimizer='Adam', metrics=['accuracy'])
MLPResidual_classifier.fit(
    X_train, y_train, epochs=100, batch_size=64, verbose=2)
MLPResidual_y_pred = MLPResidual_classifier.predict(X_test)
MLPResidual_loss, MLPResidual_accuracy = MLPResidual_classifier.evaluate(
    X_test, y_test, verbose=0)
MLPResidual_top_k_accuracy = top_k_accuracy_score(
    y_test, MLPResidual_y_pred, k=k)

print(f"MLP Residual Accuracy: {MLPResidual_accuracy*100:.2f}%")
print(f"MLP Residual Top-{k} Accuracy: {MLPResidual_top_k_accuracy*100:.2f}%")

# Transformer

transformer_classifier = Transformer(input_shape_T, 64, num_heads, n_classes)

transformer_classifier.compile(optimizer='Adam',
                               loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy',
#               optimizer='Adam', metrics=['accuracy'])
transformer_classifier.fit(
    X_train_T, y_train, epochs=100, batch_size=512, verbose=2)
transformer_y_pred = transformer_classifier.predict(X_test_T)
transformer_loss, transformer_accuracy = transformer_classifier.evaluate(
    X_test_T, y_test, verbose=0)
transformer_top_k_accuracy = top_k_accuracy_score(
    y_test, transformer_y_pred, k=k)

print(f"Transformer Accuracy: {transformer_accuracy*100:.2f}%")
print(f"Transformer Top-{k} Accuracy: {transformer_top_k_accuracy*100:.2f}%")

#####################################################################################################
##
# GRAFICOS
##
#####################################################################################################

xgb_accuracies = []
xgb_top_k_accuracies = []
MLPResidual_accuracies = [MLPResidual_accuracy] * 11
MLPResidual_top_k_accuracies = []
transformer_accuracies = [transformer_accuracy] * 11
transformer_top_k_accuracies = []
ks = range(1, 12)

# Looping over k values
for k in range(1, 12):
    # Get predictions for XGBoost
    xgb_y_pred_proba = xgb_classifier.predict_proba(X_test)
    xgb_top_k_accuracy = top_k_accuracy_score(y_test, xgb_y_pred_proba, k=k)
    xgb_accuracies.append(xgb_classifier.score(X_test, y_test))
    xgb_top_k_accuracies.append(xgb_top_k_accuracy)

    # Get predictions for MLPResidual
    MLPResidual_top_k_accuracy = top_k_accuracy_score(
        y_test, MLPResidual_y_pred, k=k)
    MLPResidual_top_k_accuracies.append(MLPResidual_top_k_accuracy)

    # Get predictions for CatBoost
    transformer_top_k_accuracy = top_k_accuracy_score(
        y_test, MLPResidual_y_pred, k=k)
    transformer_top_k_accuracies.append(transformer_top_k_accuracy)


# Assumindo que ks e as listas de top_k_accuracies já foram calculadas
plt.figure(figsize=(10, 6))
plt.plot(ks, [acc * 100 for acc in xgb_top_k_accuracies],
         marker='o', linestyle='--', label='XGBoost')  # Linha tracejada para XGBoost
plt.plot(ks, [acc * 100 for acc in MLPResidual_top_k_accuracies],
         marker='s', linestyle='--', label='Residual MLP')  # Linha tracejada para Residual MLP
plt.plot(ks, [acc * 100 for acc in transformer_top_k_accuracies],
         marker='^', linestyle='--', label='Transformers')  # Linha tracejada para Transformers

plt.xlabel('Top-k')
plt.xticks(range(1, 12))  # Exibindo todos os valores de k no eixo x
# Alterado para incluir o símbolo de porcentagem
plt.ylabel('Top-k Accuracy (%)')
plt.title('Top-k Accuracy vs. k')
# Formata o eixo y para mostrar porcentagens
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.legend()
plt.grid(True)
plt.savefig('./02. Relatorio/top-k-accuracy-model-comparison.png',
            format='png', dpi=300)
plt.show()
