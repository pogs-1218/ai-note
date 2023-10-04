import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer

## missing values



## categorical features
categorical_features = [['male', 'kor'],['female', 'us'], ['male', 'uk']]
test_features = [['male', 'us']]

def ordinal_encode():
  enc = preprocessing.OrdinalEncoder(dtype=np.int16)
  enc.fit(categorical_features)
  print(enc.categories_)
  output = enc.transform(test_features)
  print(output)

def onehot_encode():
  enc = preprocessing.OneHotEncoder()
  enc.fit(categorical_features)
  print(enc.categories_)
  output = enc.transform(test_features)
  print(output.shape)
  print(enc.get_feature_names_out())
  print(output)
  print(output.toarray())

def min_max_scale():
  data = np.array([[1., -1., 2.],
                   [2., 0., 0.],
                   [0., 1., -1.]])
  min_max_scaler = preprocessing.MinMaxScaler()
  out = min_max_scaler.fit_transform(data)
  # print(min_max_scaler.scale_)
  print(out)
  print(min_max_scaler.transform([[-3., -1., 4.]]))

def standard_scale():
  scaler = preprocessing.StandardScaler()
  data = np.array([[1., -1., 2.],
                   [2., 0., 0.],
                   [0., 1., -1.]])
  out = scaler.fit_transform(data)
  print(out)

## transformation
def test_make_pipeline():
  data = np.arange(1, 10)
  p = Pipeline([('log', FunctionTransformer(np.log, inverse_func=np.exp))])
  out = p.fit_transform(data)
  print(out[7:])
  print(np.log(8), np.log(9))
  print(p.inverse_transform(out))

  p = make_pipeline(FunctionTransformer(np.log, inverse_func=np.exp))

def test_column_trans():
  data = {}
  transformer = ColumnTransformer()
  out = transformer.fit_transform(data)


if __name__ == '__main__':
  # onehot_encode()
  # min_max_scale()
  test_make_pipeline()