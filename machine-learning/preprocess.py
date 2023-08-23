from pathlib import Path
import urllib.request
import tarfile

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

def load_housing_data():
    tarball_path = Path('datasets/housing.tgz')
    if not tarball_path.is_file():
        Path('datasets').mkdir(parents=True, exist_ok=True)
        url = 'https://github.com/ageron/data/raw/main/housing.tgz'
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path='datasets')
    return pd.read_csv(Path('datasets/housing/housing.csv'))

def show_housing_attr(housing: pd.DataFrame):
  print(f'basic info--------------')
  housing.info()
  print(f'describe numeric--------')
  housing.describe()
  housing.hist(bins=50, figsize=(20, 10))
  plt.show()

def split_data(data: pd.DataFrame, 
               test_ratio: float):
   train_set, test_set = train_test_split(data, test_size=test_ratio, random_state=1)
   return train_set, test_set

if __name__ == '__main__':
  housing = load_housing_data()
  # show_housing_attr(housing)
  train_set, test_set = split_data(housing, 0.2)
  print(f'train size: {train_set.shape} / {type(train_set)}')
  print(f'test size: {test_set.shape}')

  # train_set is prepared.
  corr = train_set.corr(numeric_only=True)
  print(corr['median_house_value'].sort_values(ascending=False))