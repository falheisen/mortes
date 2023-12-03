import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def sample_group(group, seed=12227):
    return group.sample(n=1000, replace=False, random_state=seed) if len(group) > 1000 else group


# ALL CAUSES, ALL FEATURES


# df = pd.read_pickle('./data/SP_treated_base.pkl')
# sampled_df = df.groupby('causabas_capitulo', group_keys=False,
#                         as_index=False).apply(sample_group)
# sampled_df.reset_index(drop=True, inplace=True)
# del df

# sampled_df.causabas_capitulo.value_counts().to_excel('causas.xlsx')
# # Supondo que 'sampled_df' contém seus dados e 'causabas_capitulo' é a variável de destino
# y_all_causes = sampled_df.causabas_capitulo

# # Codificar os rótulos
# label_encoder = LabelEncoder()
# y_encoded_all_causes = label_encoder.fit_transform(y_all_causes)
# X_all_causes = sampled_df.drop(columns=['causabas_capitulo'])

# # Fazer o split estratificado
# X_train_all_causes_all_features, X_test_all_causes_all_features, y_train_all_causes_all_features, y_test_all_causes_all_features = train_test_split(
#     sampled_df,
#     y_encoded_all_causes,
#     train_size=0.7,
#     test_size=0.3,
#     stratify=y_encoded_all_causes,  # Estratificação
#     random_state=42
# )

# X_train_all_causes_all_features.columns = X_train_all_causes_all_features.columns.str.lower()
# X_test_all_causes_all_features.columns = X_test_all_causes_all_features.columns.str.lower()

# # y_encoded = y_encoded.to_numpy()
# X_all_causes = X_all_causes.to_numpy()
# X_train_all_causes_all_features = X_train_all_causes_all_features.to_numpy()
# X_test_all_causes_all_features = X_test_all_causes_all_features.to_numpy()
# # y_train_all_causes_all_features = y_train_all_causes_all_features.to_numpy()
# # y_test_all_causes_all_features = y_test_all_causes_all_features.to_numpy()

# np.savez('./data/data_cv_all_causes_all_features.npz',
#          X=X_all_causes,
#          y=y_encoded_all_causes,
#          allow_pickle=True)

# np.savez('./data/data_all_causes_all_features.npz',
#          X_test=X_test_all_causes_all_features,
#          y_test=y_test_all_causes_all_features,
#          X_train=X_train_all_causes_all_features,
#          y_train=y_train_all_causes_all_features,
#          allow_pickle=True)

# SLICED CAUSES, ALL FEATURES

df = pd.read_pickle('./data/SP_treated_base_top_causes.pkl')
sampled_df = df.groupby('causabas_capitulo', group_keys=False,
                        as_index=False).apply(sample_group)
sampled_df.reset_index(drop=True, inplace=True)
del df

# sampled_df.causabas_capitulo.value_counts().to_excel('causas.xlsx')
# Supondo que 'sampled_df' contém seus dados e 'causabas_capitulo' é a variável de destino
y_top_causes = sampled_df.causabas_capitulo

# Codificar os rótulos
# label_encoder = LabelEncoder()
# with open('label_encoder.pkl', 'wb') as file:
#     pickle.dump(label_encoder, file)
with open('./data/label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)
y_encoded_top_causes = label_encoder.fit_transform(y_top_causes)
X_top_causes = sampled_df.drop(columns=['causabas_capitulo'])

# Fazer o split estratificado
X_train_top_causes_all_features, X_test_top_causes_all_features, y_train_top_causes_all_features, y_test_top_causes_all_features = train_test_split(
    X_top_causes,
    y_encoded_top_causes,
    train_size=0.7,
    test_size=0.3,
    stratify=y_encoded_top_causes,  # Estratificação
    random_state=12227
)

X_train_top_causes_all_features.columns = X_train_top_causes_all_features.columns.str.lower()
X_test_top_causes_all_features.columns = X_test_top_causes_all_features.columns.str.lower()

# y_encoded = y_encoded.to_numpy()
X_top_causes = X_top_causes.to_numpy()
X_train_top_causes_all_features = X_train_top_causes_all_features.to_numpy()
X_test_top_causes_all_features = X_test_top_causes_all_features.to_numpy()
# y_train_top_causes_all_features = y_train_top_causes_all_features.to_numpy()
# y_test_top_causes_all_features = y_test_top_causes_all_features.to_numpy()

np.savez('./data/data_cv_top_causes_all_features.npz',
         X=X_top_causes,
         y=y_encoded_top_causes,
         allow_pickle=True)

np.savez('./data/data_top_causes_all_features.npz',
         X_test=X_test_top_causes_all_features,
         y_test=y_test_top_causes_all_features,
         X_train=X_train_top_causes_all_features,
         y_train=y_train_top_causes_all_features,
         allow_pickle=True)

# # ONLY TOP 10 PARAMS

# df_top_10_params = pd.read_pickle('./data/SP_treated_base.pkl')

# target = df_top_10_params.causabas_capitulo
# target2 = df_top_10_params.causabas_grupo
# df_top_10_params = df_top_10_params.drop(
#     columns=['causabas_capitulo', 'causabas_grupo'])
# X_train2, X_test2, y_train2, y_test2 = \
#     train_test_split(
#         df_top_10_params,
#         target,
#         train_size=0.7,
#         test_size=0.3,
#         random_state=42
#     )

# X_train2.columns = X_train2.columns.str.lower()
# X_test2.columns = X_test2.columns.str.lower()

# X_train2.to_pickle('./data/X_train_top10_params.pkl')
# X_test2.to_pickle('./data/X_test_top10_params.pkl')
# y_train2.to_pickle('./data/y_train_top10_params.pkl')
# y_test2.to_pickle('./data/y_test_top10_params.pkl')

# X_train2, X_test2, y_train2, y_test2 = \
#         train_test_split(
#             df,
#             target2,
#             train_size=0.80,
#             test_size=0.2,
#             random_state=42
#         )

# X_train2.columns = X_train2.columns.str.lower()
# X_test2.columns = X_test2.columns.str.lower()

# # Save the variables to binary files using pickle
# with open('./data/X_train2.pkl', 'wb') as f:
#     pickle.dump(X_train2, f)

# with open('./data/X_test2.pkl', 'wb') as f:
#     pickle.dump(X_test2, f)

# with open('./data/y_train2.pkl', 'wb') as f:
#     pickle.dump(y_train2, f)

# with open('./data/y_test2.pkl', 'wb') as f:
#     pickle.dump(y_test2, f)
