import pandas as pd
import matplotlib.pyplot as plt

dataset_path = 'Users/ghpi/dataset/'

def test_create():
  # DataFrame
  data = {
      'name':['aa', 'osj', 'ghpi'],
      'score':[5.2, 9.8, 3.3],
      'age':[28, 30, 35],
      'sex':['female', 'female', 'male'],
  }
  df = pd.DataFrame(data)
  print(df.head())
  print('-------------------------------')
  print(df.describe())
  print('-------------------------------')

  # Series, Each column of DataFrame
  age_series: pd.Series = df.age
  print(age_series)
  print(f'max age: {age_series.max()}')
  print(f'min age: {age_series.min()}')
  score_series: pd.Series = df['score']
  print(score_series)

  ages = pd.Series([22, 35, 28], name='Age')
  print(ages)

def test_select():
  df = pd.read_csv('/Users/ghpi/dataset/titanic/train.csv')
  # Select Columns
  # Series' head(), 1-D
  print(df['Age'].head())
  print('------------------------------')
  # DataFrame's head(), 2-dim
  print(df[['Age', 'Sex']].head())

  # Select rows
  print('------------------------------')
  above_35 = df[df['Age'] > 35]
  print(above_35.head())

  class_23 = df[df['Pclass'].isin([2, 3])]
  print(class_23.head())

  age_no_na = df[df['Age'].notna()]
  print(age_no_na.info())
  print(df.info())

  # Select row & col
  # loc: label based
  adult_names = df.loc[df['Age']>35, 'Name']

  # iloc: indexed location!
  print(df.iloc[9:25, 2:5])

def test_io():
  df = pd.read_csv('/Users/ghpi/dataset/titanic/train.csv')
  print(df.info())
  df.to_excel('titanic.xlsx', sheet_name='passengers', index=False)
  # df.to_excel('titanic2.xlsx', sheet_name='passengers', index=True)
  titanic_excel = pd.read_excel('titanic.xlsx', sheet_name='passengers')
  print('---------------------------------------')
  print(titanic_excel.info())

def test_plot():
  df = pd.read_csv('/Users/ghpi/dataset/titanic/train.csv')
  # select columns
  age_col = df['Age']
  age_col.plot(kind='hist', grid=True)
  plt.show()

def test_add_col():
  data = pd.read_csv('/Users/ghpi/dataset/air_quality_no2.csv')
  print(data)
  data['london_mg_per_cubic'] = data['station_london'] * 1.882
  print(data)

def test_rename():
  data = pd.read_csv('/Users/ghpi/dataset/air_quality_no2.csv')
  # copied??
  data_renamed = data.rename(
    columns={
      'station_antwerp':'BETR801'
    }
  )
  print(data_renamed)

def test_stat():
  df = pd.read_csv('/Users/ghpi/dataset/titanic/train.csv')
  print(f'mean of age: {df.Age.mean()}')
  print(f'{df[["Sex", "Age"]].groupby("Sex").mean()}')
  print(df.groupby('Sex')[['Age', 'Fare']].mean(numeric_only=True))
  print(df.groupby(['Survived', 'Sex'])['Fare'].mean())
  print(df['Pclass'].value_counts())

def test_reshape():
  titanic = pd.read_csv('/Users/ghpi/dataset/titanic/train.csv')
  air_quality = pd.read_csv('/Users/ghpi/dataset/air_quality_long.csv')

  # print(titanic.sort_values(by='Age', ascending=False).head())
  print(air_quality.parameter.value_counts())
  no2 = air_quality[air_quality['parameter'] == 'no2']
  print(no2)
  print('sroted -------------------------')
  no2_sorted_idx = no2.sort_index()
  print(no2_sorted_idx)
  print('groupby -------------------------')
  no2_grouped = no2_sorted_idx.groupby(['location'])
  print(no2_grouped.head())
  print('head ----------------------------')
  no2_subset = no2.sort_index().groupby(['location']).head(2)
  print(no2_subset)
  print(no2_subset.pivot(columns='location', values='value'))

  air_pvt = air_quality.pivot_table(values='value', index='location', columns='parameter', aggfunc='mean', margins=True)
  print(air_pvt)

if __name__ == '__main__':
  test_reshape()