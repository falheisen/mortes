import pandas as pd

start_year = 1996
end_year = 2021
years = [year for year in range(start_year, end_year + 1)]

dfs = {}

estado = 'RS'

for year in years:
    dfs[year] = pd.read_csv(f'../ETLSIM/ETLSIM.DORES_{estado}_{year}_t.csv')

df = pd.concat(dfs.values(), ignore_index=True)
del dfs
df.to_pickle(f'../data/{estado}_total.pkl')
