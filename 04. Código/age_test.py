import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd

data_arrays = np.load('../data/data_top_causes_all_features.npz', allow_pickle=True)
X_test = data_arrays['X_test']
X_train = data_arrays['X_train']
y_test = data_arrays['y_test']
y_train = data_arrays['y_train']

k = 3
num_classes = np.unique(y_train)
np.random.seed(12227)
with open('../data/label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

feature_index = 60
X_combined = np.concatenate((X_train, X_test), axis=0)

df = pd.DataFrame(data=X_combined, columns=[f'Feature_{i}' for i in range(X_combined.shape[1])])
df['Label'] = np.concatenate((y_train, y_test), axis=0)
# df['Label'] = label_encoder.inverse_transform(df['Label'])

# plt.figure(figsize=(12, 8))
# sns.boxplot(x='Label', y=f'Feature_{feature_index}', data=df)
# plt.title(f'Boxplot of Feature {feature_index} by Category')
# plt.xlabel('Category')
# plt.ylabel(f'Feature {feature_index} Value')
# plt.show()

# # df2 = df[df.Feature_1 == 2018]

#Violin Plot
plt.figure(figsize=(12, 8))
sns.violinplot(x='Label', y=f'Feature_{feature_index}', data=df)
plt.title(f'Violin Plot of Feature data_nasc by Category in 2018')
plt.xlabel('Category')
plt.ylabel(f'Feature data_nasc Value')
plt.show()

#Swarm Plot
# plt.figure(figsize=(12, 8))
# sns.swarmplot(x='Label', y=f'Feature_{feature_index}', data=df)
# plt.title(f'Swarm Plot of Feature {feature_index} by Category')
# plt.xlabel('Category')
# plt.ylabel(f'Feature {feature_index} Value')
# plt.show()

#Pair Plot
# sns.pairplot(df, hue='Label', vars=[f'Feature_{i}' for i in range(X_combined.shape[1])])
# plt.suptitle('Pair Plot of Features by Category', y=1.02)
# plt.show()

#Correlation Heatmap


