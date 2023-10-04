import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing

def test_mean():
  data = np.array([46, 69, 32, 60, 52, 41])
  print(np.mean(data))
  print(np.median(data))
  print(np.var(data))
  print(np.sqrt(np.var(data)))
  print(np.std(data))

def sklearn_standardization():
  data = np.array([[1., -1., 2.],
                   [2., 0., 0.],
                   [0., 1., -1.]])
  scaler = preprocessing.StandardScaler().fit(data)
  # print(scaler.mean_)
  print('(before)mean: ', data.mean(axis=0))
  print('(before)std :', data.std(axis=0))
  data_scaled = scaler.transform(data)
  # print(data_scaled)
  print('(after)mean: ', data_scaled.mean(axis=0))
  print('(after)std : ', data_scaled.std(axis=0))

def sklearn_scaling():
  data = np.array([[1., -1., 2.],
                  [2., 0., 0.],
                  [0., 1., -1.]])
  min_max_scaler = preprocessing.MinMaxScaler()
  data_minmax = min_max_scaler.fit_transform(data)

def test_exp_shape():
  data = np.arange(100, 1, -1)
  # data = np.arange(100)
  print(data[:10])
  exp_data = np.exp(data)
  log_data = np.log(data)
  # print(exp_data)
  # plt.plot(data, exp_data) 
  plt.plot(data, log_data) 
  plt.show()



# test_mean()
# sklearn_standardization()
test_exp_shape()