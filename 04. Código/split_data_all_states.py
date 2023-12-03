import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def sample_group(group, seed=12227):
    return group.sample(n=1000, replace=False, random_state=seed) if len(group) > 1000 else group


# SLICED CAUSES, ALL FEATURES

df = pd.read_pickle('../data/SP_treated_base_top_causes_all_states.pkl')
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
with open('../data/label_encoder.pkl', 'rb') as file:
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

np.savez('../data/data_cv_top_causes_all_features_all_states.npz',
         X=X_top_causes,
         y=y_encoded_top_causes,
         allow_pickle=True)

np.savez('../data/data_top_causes_all_features_all_states.npz',
         X_test=X_test_top_causes_all_features,
         y_test=y_test_top_causes_all_features,
         X_train=X_train_top_causes_all_features,
         y_train=y_train_top_causes_all_features,
         allow_pickle=True)
