import pandas as pd

df = pd.read_csv('weather_by_cities.csv')
# print(df)

# 
g = df.groupby('city')
# for city, city_df in g:
#     print(city)
#     print(city_df)

[print(city_df) for city, city_df in g]

print('-----------------------------------------------')
print(g.get_group('mumbai'))

# print(g.max())

# print(g.mean())