from pathlib import Path
from zlib import crc32
import tarfile
import urllib.request
import joblib

import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.cluster import KMeans

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint

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
# print(housing.head())
# print(housing_with_id.head())
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
# print(f'{len(train_set)} / {len(test_set)}')

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
start_splits: list[pd.DataFrame] = []
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
corr_matrix = housing.corr(numeric_only=True)
# print(corr_matrix['median_house_value'].sort_values(ascending=False))

# ? plot options
# attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_media_age']
# housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1, grid=True)
# plt.show()

# housing['rooms_per_house'] = housing['total_rooms'] / housing['households']
# housing['bedrooms_ratio'] = housing['total_bedrooms'] / housing['households']
# housing['peple_per_house'] = housing['population'] / housing['households']
# corr_matrix = housing.corr(numeric_only=True)
# print(corr_matrix['median_house_value'].sort_values(ascending=False))

# ! axis=1 means?
housing: pd.DataFrame = start_train_set.drop('median_house_value', axis=1)
housing_labels: pd.DataFrame = start_train_set['median_house_value'].copy()
print(housing.info())
# toal_bedrooms have missing values.(16512->16344)
# what should I do?

# housing.dropna(subset=['total_bedrooms'], inplace=True)
# housing.drop('total_bedrooms', axis=1)
# median = housing['total_bedrooms'].median()
# housing['total_bedrooms'].fillna(median, inplace=True)

imputer = SimpleImputer(strategy='median')
housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
# print(housing_num.describe())
# print(imputer.statistics_)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

#! housing[''] vs housing[['']]
housing_cat = housing[['ocean_proximity']]
# print(housing_cat.head(8))

#! why it is not proper? 
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:8])
print(ordinal_encoder.categories_)

#! vs Panda's get_dummies
cat_encoder = OneHotEncoder()
# cat_encoder.handle_unknown = 'ignore'
#! SciPy's sparse matrix
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# print(type(housing_cat_1hot))
# convert numpy dense array
# housing_cat_1hot.toarray()
# print(cat_encoder.categories_)

# df_test = pd.DataFrame({'ocean_proximity':['NEAR OCEAN', 'NEAR BAY']})
# print(pd.get_dummies(df_test))
# print('------------------------------------')
# print(cat_encoder.transform(df_test))
# print(cat_encoder.transform(df_test).toarray())

# # min-max scaling(normalization)
# min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
# housing_num_min_scaled = min_max_scaler.fit_transform(housing_num)
# std_scaler = StandardScaler()
# housing_num_std_scaled = std_scaler.fit_transform(housing_num)

# age_simil_35 = rbf_kernel(housing[['housing_media_age']], [[35]], gamma=0.1)

# log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
# log_pop = log_transformer.transform(housing[['population']])

# custom transformer which can be trained.
class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):
        self.with_mean = with_mean

    def fit(self, X, y=None):
        X = check_array(X)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X / self.mean_

num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('standardize', StandardScaler),
])
# or
num_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
housing_num_prepared = num_pipeline.fit_transform(housing_num)
# print(housing_num_prepared[:2].round(2))
df_housing_num_prepared = pd.DataFrame(housing_num_prepared, 
                                       columns=num_pipeline.get_feature_names_out(),
                                       index=housing_num.index)

num_attribs = ['longitude', 'latitude']
cat_attribs = ['ocean_proximity']

cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'),
                             OneHotEncoder(handle_unknown='ignore'))
# preprocessing = ColumnTransformer(('nums', num_pipeline, num_attribs),
#                                   'cat', cat_pipeline, cat_attribs)

# or

# preprocessing = ColumnTransformer((num_pipeline, make_column_selector(dtype_include=np.number)),
                                #   cat_pipeline, make_column_selector(dtype_include=object))

# housing_prepared = preprocessing.fit(housing)
#########################################################################################

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())
preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)  # one column remaining: housing_median_age

housing_prepared = preprocessing.fit_transform(housing)
print(housing_prepared.shape)
print(preprocessing.get_feature_names_out())

# select a model
lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)
housing_predictions = lin_reg.predict(housing)
print(housing_predictions[:5].round(-2))
print(housing_labels[:5].values)

lin_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
print(f'linear: {lin_rmse}')

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)
housing_predictions = tree_reg.predict(housing)
tree_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
print(f'tree: {tree_rmse}')

tree_rmses = -cross_val_score(tree_reg, housing, housing_labels, scoring='neg_root_mean_squared_error', cv=10)
print(pd.Series(tree_rmses).describe())

forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
forest_rmses = -cross_val_score(forest_reg, housing, housing_labels, scoring='neg_root_mean_squared_error', cv=10)
print(pd.Series(forest_rmses).describe())

# find-tune
full_pipeline = Pipeline([('preprocessing', preprocessing),
                          ('random_forest', RandomForestRegressor(random_state=42))])
param_grid = [{'preprocessing__geo__n_clusters':[5, 8, 10], 'random_forest__max_features':[4, 6, 8]},
              {'preprocessing__geo__n_clusters':[10, 15], 'random_forest__max_features':[6, 8, 10]}]                          
grid_search = GridSearchCV(full_pipeline, param_grid, cv=3, scoring='neg_root_mean_squared_error')
grid_search.fit(housing, housing_labels)
print(grid_search.best_params_)

param_distribs = {'preprocessing__geo__n_clusters': randint(low=3, high=50),
                  'random_forest__max_features':randint(low=2, high=20)}
rnd_search = RandomizedSearchCV(full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
                                scoring='neg_root_mean_squared_error', random_state=42)
rnd_search.fit(housing, housing_labels)
final_model = rnd_search.best_estimator_
feature_improtances = final_model['random_forest'].feature_importances_
print(feature_improtances.round(2))
print(sorted(zip(feature_improtances,
                 final_model['preprocessing'].get_feature_names_out()),
                 reverse=True))

# test
X_test = start_test_set.drop('median_house_value', axis=1)
y_test = start_test_set['median_house_value'].copy()
final_predictions = final_model.predict(X_test)
final_rmse = mean_squared_error(y_test, final_predictions, squared=False)
print(final_rmse)

joblib.dump(final_model, 'my_california_housing_model.pkl')

# final_model_reloaded = joblib.load('my_california_housing_model.pkt')