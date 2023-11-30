# Author: Artur Jordao <arturjlcorreia@gmail.com>
from tensorflow.keras import *
from keras.layers import *
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score
import numpy as np
from sklearn.preprocessing import StandardScaler


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


# if __name__ == '__main__':
np.random.seed(12227)

# n_samples, n_features, n_classes = 1000, 300, 13
# X = np.random.rand(n_samples, n_features)
# X = np.expand_dims(X, axis=1)
# y = np.random.randint(0, n_classes, size=n_samples)
# y = np.eye(n_classes)[y]
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.33, random_state=12227)

tmp = np.load('../../data/data_top_causes_all_features.npz')
X_train, X_test, y_train, y_test = tmp['X_train'], tmp['X_test'], tmp['y_train'], tmp['y_test']
n_classes = len(np.unique(y_test))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

input_shape = X_train.shape[1:]
projection_dim = 64
num_heads = [8]*5
model = Transformer(input_shape, 64, num_heads, n_classes)

model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy',
#               optimizer='Adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=512, verbose=2)
y_pred = model.predict(X_test)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"Transformer Accuracy: {accuracy*100:.2f}%")
for i in range(1, 5):
    acc = top_k_accuracy_score(y_test, y_pred, k=i)
    print(f'Transformer Top-{i} Accuracy: {acc*100:.2f}%')
