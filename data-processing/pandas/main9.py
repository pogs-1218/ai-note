'''
stack
'''

import pandas as pd

df = pd.read_excel('stocks.xlsx', header=[0,1])
print(df)

df_stacked = df.stack()
print(df_stacked)

df2 = pd.read_excel('stocks_3_levels.xlsx',header=[0,1,2])
print(df2)
df_stacked2 = df2.stack(level=1)
print(df_stacked2)