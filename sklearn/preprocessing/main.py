import numpy as np
from sklearn import preprocessing

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

if __name__ == '__main__':
  # onehot_encode()
  min_max_scale()