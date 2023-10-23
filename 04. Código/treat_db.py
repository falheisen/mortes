import pandas as pd
import numpy as np

df = pd.read_pickle('../data/SP_total.pkl')
cbo2002 = pd.read_excel('../03. Planilhas/CBO2002.xlsx')

### FILTER PARAMETERS

df = df[[
    'data_obito',
    'ano_obito',
    'dia_semana_obito',
    'NATURAL',
    'data_nasc',
    'idade_obito_calculado',
    'ano_nasc',
    'dia_semana_nasc',
    'idade_obito_anos',
    'def_sexo',
    'def_raca_cor',
    'def_est_civil',
    'def_escol',
    'OCUP',
    # 'CODMUNRES',
    # 'CODMUNOCOR',
    'def_loc_ocor',
    'def_assist_med',
    'def_exame',
    'def_cirurgia',
    'def_necropsia',
    'res_CAPITAL',
    'res_MSAUDCOD',
    'res_RSAUDCOD',
    'res_CSAUDCOD',
    'res_LATITUDE',
    'res_LONGITUDE',
    'res_ALTITUDE',
    'res_AREA',
    'ocor_CAPITAL',
    'ocor_MSAUDCOD',
    'ocor_RSAUDCOD',
    'ocor_CSAUDCOD',
    'ocor_LATITUDE',
    'ocor_LONGITUDE',
    'ocor_ALTITUDE',
    'ocor_AREA',
    'ocor_SIGLA_UF',
    'ocor_REGIAO',
    'idade_obito',
    'CAUSABAS',
    'def_circ_obito'
]]

df.to_pickle('../data/SP_selected_parametes.pkl')
df = pd.read_pickle('../data/SP_selected_parametes.pkl')

# IDENTIFICANDO COLUNAS DE CATEGORIAS

cat_col = [
    'dia_semana_obito',
    'NATURAL',
    'dia_semana_nasc',
    'def_sexo',
    'def_raca_cor',
    'def_est_civil',
    'def_escol',
    'def_loc_ocor',
    'def_assist_med',
    'def_exame',
    'def_cirurgia',
    'def_necropsia',
    'res_CAPITAL',
    'res_MSAUDCOD',
    'res_RSAUDCOD',
    'res_CSAUDCOD',
    'ocor_CAPITAL',
    'ocor_MSAUDCOD',
    'ocor_RSAUDCOD',
    'ocor_CSAUDCOD',
    'ocor_SIGLA_UF',
    'ocor_REGIAO',
    'grande_grupo_ocup',
    'nivel_comp_ocup'
]

outras_colunas = df.columns
outras_colunas = list(set(outras_colunas).difference(set(cat_col)))

# Datas (UNIX) 
df['data_obito'] = pd.to_datetime(df['data_obito'])
df['data_nasc'] = pd.to_datetime(df['data_nasc'])
df['data_nasc'] = df['data_nasc'].astype('int64') 
df['data_obito'] = df['data_obito'].astype('int64') 
# df['DTATESTADO'] = df['DTATESTADO'].apply(lambda x: str(int(x)).zfill(8) if not pd.isnull(x) else np.nan)
# pd.to_datetime(df['DTATESTADO'], format='%d%m%Y')
# df['data_obito'] = pd.to_datetime(df['data_obito'], unit='ns')

# OCUPAÇÃO
df['OCUP'] = df['OCUP'].apply(lambda x: str(int(x)).zfill(6) if not pd.isnull(x) else np.nan)
df['digito_ocup'] = df.OCUP.apply(lambda x: x[0] if not pd.isnull(x) else np.nan)
cbo2002['digito_ocup'] = cbo2002['digito_ocup'].str.strip('\xa0')
df = pd.merge(df,cbo2002, how='left',on='digito_ocup')
df = df.drop(columns=['OCUP','digito_ocup'])

df2 = df[cat_col].copy()
one_hot_encoded_data = pd.get_dummies(df2, columns = cat_col)
one_hot_encoded_data.columns = [c.replace(' ', '_') for c in one_hot_encoded_data.columns]

df = df.drop(columns=cat_col)
df = pd.merge(df, one_hot_encoded_data, left_index=True, right_index=True)
df = df.dropna(axis=0)

df.to_pickle('../data/SP_treated_base.pkl')