
import pandas as pd

df = pd.read_pickle('../data/SP_total.pkl')

def percentage_ignorados(series, name):
    ignorados = series.loc[lambda x : x == 'Ignorado'].shape[0]
    pct_ignorados = (ignorados / 6993473) * 100
    print(f'{name}: {pct_ignorados:.0f}% dos valores ignorados')

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

df.def_gravidez.value_counts()
df.def_gestacao.value_counts()
df.def_parto.value_counts()
df.def_obito_parto.value_counts()
df.def_obito_grav.value_counts()
df.def_obito_puerp.value_counts()
df.def_assist_med.value_counts()
df.def_exame.value_counts()
df.def_cirurgia.value_counts()
df.def_necropsia.value_counts()
df.def_fonte.value_counts()