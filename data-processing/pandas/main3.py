import pandas as pd

df = pd.read_csv('weather_data.csv', parse_dates=['day'])
print(f'{type(df.day[0])}')
df.set_index('day', inplace=True)
print(df)
new_df = df.fillna({
    'temperature': 0,
    'windspeed':0,
    'event': 'no event'
})
#print(new_df.head(4))

new_df = df.fillna(method='bfill', limit=1)
#print(new_df)

new_df = df.interpolate(method='time')
# print(new_df)

# new_df = df.dropna(how='all')
new_df = df.dropna(thresh=2)
print(new_df)

dt = pd.date_range('2017-01-01', '2017-01-11')
idx = pd.DatetimeIndex(dt)
df = df.reindex(idx)
print(df)