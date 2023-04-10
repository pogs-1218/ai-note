import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def show_def_sigmoid():
  z_tmp = np.arange(-10, 11)
  y = sigmoid(z_tmp)
  print(np.c_[z_tmp, y])
  fig, ax = plt.subplots(1, 1, figsize=(5, 3))
  ax.plot(z_tmp, y, c='b')
  ax.set_ylabel('sigmoid(z)')
  ax.set_xlabel('z')
  plt.show()

def show_train_data(x_train, y_train):
  m = x_train.shape[0]
  fig, ax = plt.subplots(1, 1, figsize=(5, 3))
  ax.axis([0, 4, 0, 3.5])
  for i in range(m):
    if y_train[i] == 0:
      ax.scatter(x_train[i,0], x_train[i,1], marker='o', c='blue')
    else:
      ax.scatter(x_train[i,0], x_train[i,1], marker='x', c='red')
  plt.show()

def show_decision_boundary(x_train, y_train, w, b):
  '''
   we have two features. x0, x1, for example, [0.5, 1.5]
   z = w0x0 + w1x1 + b
   Find a decision boundary
   if w0 = 1, w1 = 1, b = -3
   x0 + x1 -3 = 0
   x1 = 3-x0

   Generalize,
   x1 = -(w0x0+b) / w1
 '''
  m = x_train.shape[0]
  x0 = np.arange(0, 10)
  x1 = -(w[0] * x0 + b) / w[1]
  fig, ax = plt.subplots(1, 1, figsize=(5, 3))
  ax.axis([0, 4, 0, 3.5])
  for i in range(m):
    if y_train[i] == 0:
      ax.scatter(x_train[i,0], x_train[i,1], marker='o', c='blue')
    else:
      ax.scatter(x_train[i,0], x_train[i,1], marker='x', c='red')
  ax.plot(x0, x1) 
  plt.show()

def compute_cost(x_train, y_train, w, b):
  m = x_train.shape[0]
  cost = 0
  for i in range(m):
    z = np.dot(x_train[i,:], w) + b
    a = sigmoid(z)
    loss = (y_train[i] * np.log(a)) + ((1-y_train[i]) * np.log(1-a))
    cost += loss
  cost = -cost / m
  return cost

x_train = np.array([[0.5, 1.5], 
                    [1, 1], 
                    [1.5, 0.5], 
                    [3, 0.5], 
                    [2, 2], 
                    [1, 2.5]])
# x_train.shape = (6, 2)

y_train = np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1)
# y_train shape is chaned from (6,) to (6,1) through reshape()

w = np.array([1, 1])
b = -4
c = compute_cost(x_train, y_train, w, b)
print(f'{c}')
show_decision_boundary(x_train, y_train, w, b)
