from pathlib import Path
import urllib.request
import tarfile
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

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

def extract_labels(data: pd.DataFrame):
  X = housing.drop('median_house_value', axis=1)
  y = housing.loc[:, 'median_house_value']
  return X, y

def fill_missing_value(X):
  imp = SimpleImputer(strategy='median')
  imp_target: pd.DataFrame = X['total_bedrooms'].to_frame()
  # print(imp_target.info())
  imputed_X: np.ndarray = imp.fit_transform(imp_target)
  # print(pd.DataFrame(imputed_X).info())

def encode_categorical_fetures(housing: pd.DataFrame):
  cat_feature: pd.DataFrame = housing[['ocean_proximity']]
  # method 1. ordinal encoder
  # pros & cons
  enc = preprocessing.OrdinalEncoder()
  enc.fit(cat_feature)
  category_encoded = enc.transform(cat_feature)
  # print(enc.categories_)
  # print(category_encoded)
  # method 2. one-hot encoder
  # pros & cons
  enc = preprocessing.OneHotEncoder()
  category_encoded = enc.fit_transform(cat_feature)
  # print(category_encoded)

def normalization(housing: pd.DataFrame):
  housing_num = housing.select_dtypes(include='number')
  scaler = preprocessing.MinMaxScaler(copy=False)
  scaler.fit_transform(housing_num)
  housing.update(housing_num)
  # print(housing.head())
  return housing


if __name__ == '__main__':
  housing = load_housing_data()
  # show_housing_attr(housing)
  train_set, test_set = split_data(housing, 0.2)
  print(f'train size: {train_set.shape} / {type(train_set)}')
  print(f'test size: {test_set.shape}')

  # train_set is prepared.
  corr = train_set.corr(numeric_only=True)
  print(corr['median_house_value'].sort_values(ascending=False))

  # Get train data with labels
  X_train, y_train = extract_labels(train_set)
  fill_missing_value(X_train)
  encode_categorical_fetures(X_train)
  normalization(X_train)

  X_train['population'] = np.log(X_train['population'])
  # X_train.hist(bins=50, figsize=(30, 10))
  # plt.show()
