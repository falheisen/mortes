### LIBRARIES

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

### GET DATA
# start_year = 1996
# end_year = 2021
# years = [year for year in range(start_year, end_year + 1)]

# dfs = {}

# for year in years:
#     dfs[year] = pd.read_csv(f'../ETLSIM/ETLSIM.DORES_SP_{year}_t.csv')

# df = pd.concat(dfs.values(), ignore_index=True)
# del dfs
# df.to_pickle('../data/SP_total.pkl')

df = pd.read_pickle('../data/SP_total.pkl')

def percentage_ignorados(series, name):
    ignorados = series.loc[lambda x : x == 'Ignorado'].shape[0]
    pct_ignorados = (ignorados / 6993473) * 100
    print(f'{name}: {pct_ignorados:.0f}% dos valores ignorados')

### VERIFICAR ALGUNS PARAMETROS
(df.idade_obito_calculado - df.idade_obito_anos).value_counts()
df.def_tipo_obito.value_counts()

colunas_ignorado = [
    'def_sexo',
    'def_raca_cor',
    'def_est_civil',
    'def_escol',
    'def_loc_ocor',
    'def_escol_mae',
    'def_gravidez',
    'def_gestacao',
    'def_parto',
    'def_obito_parto',
    'def_obito_grav',
    'def_obito_puerp',
    'def_assist_med',
    'def_exame',
    'def_cirurgia',
    'def_necropsia',
    'def_fonte',
]

for coluna in colunas_ignorado:
    series = df[coluna]
    percentage_ignorados(series, coluna)

# df.def_gravidez.value_counts()
# df.def_gestacao.value_counts()
# df.def_parto.value_counts()
# df.def_obito_parto.value_counts()
# df.def_obito_grav.value_counts()
# df.def_obito_puerp.value_counts()
# df.def_assist_med.value_counts()
# df.def_exame.value_counts()
# df.def_cirurgia.value_counts()
# df.def_necropsia.value_counts()
# df.def_fonte.value_counts()

### FILTER PARAMETERS
df = df[[
    # 'def_tipo_obito',
    'ano_obito',
    'NATURAL',
    'idade_obito_calculado',
    'def_sexo',
    'def_raca_cor',
    'def_est_civil',
    'def_escol',
    'OCUP',
    'CODMUNRES',
    'def_loc_ocor',
    'CODMUNOCOR',
    # 'def_escol_mae',
    # 'def_gravidez',
    # 'def_gestacao',
    # 'def_parto',
    # 'def_obito_parto',
    # 'def_obito_grav',
    # 'def_obito_puerp',
    'def_assist_med',
    'def_exame',
    'def_cirurgia',
    'def_necropsia',
    # 'def_fonte',
    'res_CAPITAL',
    'res_MSAUDCOD',
    'res_RSAUDCOD',
    'res_CSAUDCOD',
    'res_LATITUDE',
    'res_LONGITUDE',
    'res_ALTITUDE',
    'res_AREA',
    'res_codigo_adotado',
    'ocor_CAPITAL',
    'ocor_MSAUDCOD',
    'ocor_RSAUDCOD',
    'ocor_CSAUDCOD',
    'ocor_LATITUDE',
    'ocor_LONGITUDE',
    'ocor_ALTITUDE',
    'ocor_AREA',
    'ocor_codigo_adotado',
    'res_SIGLA_UF',
    'res_CODIGO_UF',
    'res_REGIAO',
    'ocor_REGIAO',
    'res_coordenadas',
    'ocor_coordenadas',
    'CAUSABAS',
    'def_circ_obito'
]]

## IDENTIFICANDO COLUNAS DE CATEGORIAS

cat_col = [
    'def_sexo',
    'def_raca_cor',
    'def_est_civil',
    'def_escol',
    'def_loc_ocor',
    'def_assist_med',
    'def_exame',
    'def_cirurgia',
    'def_necropsia'
]

df2 = df[cat_col].copy()

one_hot_encoded_data = pd.get_dummies(df2, columns = cat_col)
one_hot_encoded_data.columns = [c.replace(' ', '_') for c in one_hot_encoded_data.columns]

## IDENTIFICANDO COLUNAS DE CATEGORIAS

import time
start = time.time()
kmeans = KMeans(n_clusters=4)
kmeans.fit(one_hot_encoded_data)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
print("Cluster Labels:", labels)
print("Cluster Centroids:", centroids)
end = time.time()
print(f'{end-start}')