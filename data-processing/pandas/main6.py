'''
concat & merge
'''

import pandas as pd

inde_weather = pd.DataFrame({
    'city':['mumbai', 'delhi', 'banglore'],
    'temperature':[32, 45, 30],
    'humidity':[80, 60, 78]
})
# print(inde_weather)

us_weather = pd.DataFrame({
    'city':['newyork', 'chicago', 'orlando'],
    'temperature':[21, 14, 35],
    'humidity':[68, 65, 75]
})
# print(us_weather)

# df = pd.concat([inde_weather, us_weather],ignore_index=True)
df = pd.concat([inde_weather, us_weather], keys=['india', 'us'])
print(df)

temperatrue_df = pd.DataFrame({
    'city':['newyork', 'chicago', 'orlando'],
    'temperature':[21, 14, 35],
})
humidity_df = pd.DataFrame({
    'city':['newyork', 'orlando'],
    'humidity':[68, 75]
})
df2 = pd.concat([temperatrue_df, humidity_df],axis=1)
# print(df2)

df3 = pd.merge(temperatrue_df, humidity_df, on='city', how='outer', indicator=True)
print(df3)