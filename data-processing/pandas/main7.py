'''
pivot table
summurize and aggregate
'''

import pandas as pd

df = pd.read_csv('weather.csv')
df = df.pivot(index='date', columns='city', values='humidity')
print(df)

df = pd.read_csv('weather2.csv')
print(df)

df = df.pivot_table(index='city', columns='date', aggfunc='sum', margins=True)
print(df)

df = pd.read_csv('weather3.csv')
df['date']=pd.to_datetime(df['date'])
df = df.pivot_table(index=pd.Grouper(freq='M',key='date'), columns='city')
print(df)
