import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

def load_data(path, features):
  '''
    Return:
      train_data
      train_label
      test_data
      test_label
  '''
  df = pd.read_csv(path) 
  print(f"length: {len(df)}")

  # Split train data and test data, 8:2
  offset = math.ceil(len(df) * 0.8)
  train_df = df[:offset]
  test_df = df[offset:]
#print(f"{len(train_df)}, {len(test_df)}")

  target = ['price']

  train_data = train_df.loc[:,features]
  train_label = train_df.loc[:,target]
#  print("train ================")
#  print(f"{type(train_data)}\n {train_data[:3]} ")
#  print(f"{train_label[:3]}")

#  print("convert pd.DataFrame ot numpy.ndarray")
  train_data = train_data.to_numpy()
  train_label = train_label.to_numpy()
#  print("train as numpy -----------")
#  print(f"{type(train_data)} with {train_data.shape}\n {train_data[:3]}")
#  print(f"{type(train_label)} with {train_label.shape}\n {train_label[:3]}")

  test_data = test_df.loc[:,features]
  test_label = test_df.loc[:,target]
  test_data = test_data.to_numpy()
  test_label = test_label.to_numpy()

  return (train_data, train_label, test_data, test_label)

def load_data2(path):
  d = np.loadtxt(path, delimiter=",")
  y_train = d[:,4]
  x_train = d[:,:4]
  return (x_train, y_train)

def plot_features(data, label, features):  
#xcount = 2
#ycount = math.floor(len(features) / 2)
#fig, ax = plt.subplots(xcount, ycount, figsize=(15, 5))
  fig, ax = plt.subplots(1, len(features), figsize=(20, 5))
#print(len(ax[0,:]))
  for i in range(len(ax)):
    ax[i].scatter(data[:,i], label)
    ax[i].set_xlabel(features[i])
  plt.show()
 

def compute_cost(w, b, x_train, y_train):
  '''
    compute cost function
    Prameters:
      w
      b
      x_train
      y_train

    Return:
      cost
  '''
  m = x_train.shape[0]  # may be 
  n = x_train.shape[1]
  fsum = 0
  for i in range(m):
    fwb = np.dot(x_train[i,:], w) + b
    fsum += (fwb - y_train[i]) ** 2

  cost = fsum / (2*m)
  return cost


def compute_gradient(w, b, x_train, y_train):
  '''
    Parameters:
      w (n, 1)
      b (m, 1)
      x_train (m, n)
      y_train (m, 1)
    Returns:
      dj_dw (n, 1)
      dj_db scalar
  '''
  dj_dw = np.zeros_like(w)
  dj_db = 0
  m = x_train.shape[0]  # the number of training data
  n = x_train.shape[1]  # the number of features

  for i in range(m):
    p = np.dot(x_train[i,:], w) + b
    err = p - y_train[i]  
    for j in range(n):
      dj_dw[j] += err * x_train[i,j]
    dj_db += err

  dj_dw /= m
  dj_db /= m

  return (dj_dw, dj_db)

def gradient_descent(w_init, b_init, x_train, y_train, alpha, steps):
  '''
    Parameters
      w_init (n, 1)
      b_init (m, 1)
      x_train (m, n)
      y_train (m, 1)
  '''
  w, b = w_init, b_init 
  cost_history = []

  for i in range(steps):
    dj_dw, dj_db = compute_gradient(w, b, x_train, y_train)
    w -= alpha * dj_dw
    b -= alpha * dj_db
    c = compute_cost(w, b, x_train, y_train)
   
    if i % 10 == 0:
      print(f'{i} === {c}')

  return (w, b, cost_history) 
