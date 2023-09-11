import pandas as pd
from datetime import datetime, timedelta

def convert_to_valid_date(date_int):
    date_str = str(date_int)
    if len(date_str) == 8:
        return pd.to_datetime(date_str, format='%d%m%Y')
    if len(date_str) == 7:
        return pd.to_datetime('0' + date_str, format='%d%m%Y')    
    return None

df = pd.read_csv('DOSP2020.csv')
df = df.drop('Unnamed: 0', axis=1)

df.ORIGEM.unique()
df.DTOBITO  
df.DTOBITO = df.DTOBITO.apply(convert_to_valid_date)
df.IDADE.sort_values().unique()

