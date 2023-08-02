import pandas as pd

def show(df: pd.DataFrame):
    print(df)

def show_column(df: pd.DataFrame, name: str):
    col = df[name]
    print(type(col))
    print(col)

def show_column_with_range(df: pd.DataFrame,
                           start: int,
                           end: int):
    print(df[start:end])

def show_multi_column(df: pd.DataFrame,
                      *range):
    pass

# https://pandas.pydata.org/docs/user_guide/io.html
def create_dataframe_from_tuple(data: tuple) -> pd.DataFrame:
    return pd.DataFrame(data, columns=['day', 'temperature', 'windspeed', 'event'])

def create_dataframe_from_dict(data: dict) -> pd.DataFrame:
    return pd.DataFrame(data)

def create_dataframe_from_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def create_dataframe_from_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


df1 = create_dataframe_from_csv('weather_data.csv')
df2 = create_dataframe_from_excel('weather_data.xlsx')
df3 = create_dataframe_from_dict({
        'day':['1/1/2017', '1/2/2017','1/3/2017'],
        'temperature':[32,35,28],
        'windspeed':[6, 7, 2],
        'event':['Rain', 'Sunny', 'Snow']
    })
df4 = create_dataframe_from_tuple([
    ('1/1/2017', 32, 6, 'Rain'),
    ('1/2/2017', 35, 7, 'Sunny'),
    ('1/3/2017', 28, 2, 'Snow'),
])

# dictionary list
df5 = pd.DataFrame([
    {'day':'1/1/2017', 'temperature':32, 'windspeed':6, 'event':'Rain'},
    {'day':'1/2/2017', 'temperature':35, 'windspeed':7, 'event':'Sunny'},
    {'day':'1/3/2017', 'temperature':28, 'windspeed':2, 'event':'Snow'},
])

print(df1.head())
print(df4.head())

# print(df.head(2))
# print(df.tail(1))
# print(df.shape)

# print(df[2:5])

# print(df.columns)
# print(df['day'])
# print(df[['event', 'day']])
# # what is series data type?
# print(type(df['day']))

# print(df['temperature'].max())

# print(df.describe())

# # conditional select
# print(df[['day','temperature']][df['temperature']>=32])
# print(df.index)

# df.set_index('day', inplace=True)
# print(df)
# print(df.loc['1/3/2017'])

# print('----------------------------------')
# df.reset_index(inplace=True)
# print(df)