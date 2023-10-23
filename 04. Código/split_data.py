### LIBRARIES

import pandas as pd
from sklearn.model_selection import train_test_split

def sample_group(group):
    return group.sample(n=1000, replace=False) if len(group) > 1000 else group

df = pd.read_pickle('../data/SP_treated_base.pkl')
sampled_df = df.groupby('causabas_capitulo', group_keys=False, as_index=False).apply(sample_group)
sampled_df.reset_index(drop=True, inplace=True)
del df

target1 = sampled_df.causabas_capitulo
target2 = sampled_df.causabas_grupo
sampled_df = sampled_df.drop(columns=['causabas_capitulo','causabas_grupo'])
X_train1, X_test1, y_train1, y_test1 = \
        train_test_split(
            sampled_df, 
            target1, 
            train_size=0.7, 
            test_size=0.3, 
            random_state=42
        )

X_train1.columns = X_train1.columns.str.lower()
X_test1.columns = X_test1.columns.str.lower()

X_train1.to_pickle('../data/X_train1.pkl')
X_test1.to_pickle('../data/X_test1.pkl')
y_train1.to_pickle('../data/y_train1.pkl')
y_test1.to_pickle('../data/y_test1.pkl')

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
# with open('../data/X_train2.pkl', 'wb') as f:
#     pickle.dump(X_train2, f)

# with open('../data/X_test2.pkl', 'wb') as f:
#     pickle.dump(X_test2, f)

# with open('../data/y_train2.pkl', 'wb') as f:
#     pickle.dump(y_train2, f)

# with open('../data/y_test2.pkl', 'wb') as f:
#     pickle.dump(y_test2, f)