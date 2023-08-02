import pandas as pd
import numpy as np

df = pd.read_csv('weather_data2.csv')
print(df)
# new_df = df.replace([-99999, 0], np.NaN)
# new_df = df.replace({
#     'temperature': -99999,
#     'windspeed':-99999,
#     'event':'0'
# }, np.NaN)

# new_df = df.replace({
#     -99999: np.NaN,
#     'No Event': 'Sunny'
# })

# Whitespace is not removed.
new_df = df.replace({
    'temperature': '[A-Za-z]',
    'windspeed':'[A-Za-z]'
}, '', regex=True)
print(new_df)

df2 = pd.DataFrame({
    'score':['exceptional', 'average', 'good', 'poor','average', 'exceptional'],
    'student':['rob', 'maya', 'parthiv', 'tom', 'julian', 'erica']
})
print(df2)

df2.replace(['poor', 'average', 'good', 'exceptional'],[1, 2, 3, 4], inplace=True)
print(df2)