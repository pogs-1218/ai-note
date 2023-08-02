'''
melt
unpivot!
'''

import pandas as pd

df = pd.read_csv('weather_m.csv')
print(df)

df1 = pd.melt(df, id_vars=['day'], var_name='city', value_name='temperature')
print(df1)
print(df1[df1['city']=='chicago'])
