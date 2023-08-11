from pathlib import Path
from zlib import crc32
import tarfile
import urllib.request

import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

def load_housing_data():
    tarball_path = Path('datasets/housing.tgz')
    if not tarball_path.is_file():
        Path('datasets').mkdir(parents=True, exist_ok=True)
        url = 'https://github.com/ageron/data/raw/main/housing.tgz'
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path='datasets')
    return pd.read_csv(Path('datasets/housing/housing.csv'))

def shuffle_and_split(data: pd.DataFrame, 
                      test_ratio: float):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indicis = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indicis]

def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32

# ? why ID is needed to split data?
def split_data_with_id_hash(data: pd.DataFrame, 
                            test_ratio: float, 
                            id_column: str):
    ids = data[id_column]
    # ? apply()
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing = load_housing_data()
housing_with_id = housing.reset_index()
print(housing.head())
print(housing_with_id.head())
# print(housing.info())
# print(housing.ocean_proximity.value_counts())
# print(housing.describe())

# ? standard deviation
# ? null value?
# ? DataFrame's loc vs iloc

# ? bins?
# ? draw pyplot histogram from dataframe directly.
# housing.hist(bins=50)
# plt.show()

train_set, test_set = shuffle_and_split(housing, test_ratio=0.2)
print(f'{len(train_set)} / {len(test_set)}')

train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, 'index')

housing['income_cat'] = pd.cut(housing['median_income'],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
# housing['income_cat'].value_counts().sort_index().plot.bar(rot=0, grid=True)
# plt.xlabel('income category')
# plt.ylabel('number of districts')
# plt.show()         

# ? meaning of random_state
splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
start_splits = []
# ? splitter.split()
for train_index, test_index in splitter.split(housing, housing['income_cat']):
    start_train_set_n = housing.iloc[train_index]
    start_test_set_n = housing.iloc[test_index]
    start_splits.append([start_train_set_n, start_test_set_n])
start_train_set, start_test_set = start_splits[0]
# vs train_test_split from sklearn
# print(start_test_set['income_cat'].value_counts() / len(start_test_set))

# housing.plot(kind='scatter', x='longitude', y='latitude', grid=True, alpha=0.2,
#              s=housing['population']/100, label='population',
#              c='median_house_value', cmap='jet', colorbar=True,
#              legend=True, sharex=False, figsize=(10, 7))
# plt.show()

# ? what is corr, standard correlation coefficient
# corr_matrix = housing.corr()
# print(corr_matrix['median_house_value'].sort_values(ascending=False))

# ? plot options
attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_media_age']
housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1, grid=True)
plt.show()