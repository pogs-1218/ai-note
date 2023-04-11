import numpy as np
import matplotlib.pyplot as plt

# WARNING!!
# It's not good at debugging...
#np.set_printoptions(precision=3)

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

def compute_gradient(x_train, y_train, w, b):
  '''
    Parameters
      x_train (m, n)
      y_train (m, 1)
      w (n, 1)
      b (m, 1)
    Returns
      dj_dw (n, 1)
      dj_db (scalar)
  '''
  m, n = x_train.shape
  dj_dw = np.zeros(n)
  dj_db = 0
  for i in range(m):
    z = np.dot(x_train[i], w) + b
    err = sigmoid(z) - y_train[i]
    for j in range(n):
      dj_dw[j] += err * x_train[i, j]
    dj_db += err 
  dj_dw /= m
  dj_db /= m

  return (dj_dw, dj_db)

def gradient_descent(x_train, y_train, w_init, b_init, alpha, steps):
  '''
    Parameters
      x_train (m, n)
      y_train (m, 1)
      w (n, 1)
      b (m, 1)     
      alpha (scalar) : learning rate
      steps (scalar) : epochs

    Retruns:
      w (n, 1)
      b (scalar)
  '''
  w, b = w_init, b_init 
  for i in range(steps):
    dj_dw, dj_db = compute_gradient(x_train, y_train, w, b)
    w -= (alpha * dj_dw)
    b -= (alpha * dj_db)
    c = compute_cost(x_train, y_train, w, b)
    if i % 1000 == 0:
      print(f'{i} === {c}')

  return (w, b)

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
#show_decision_boundary(x_train, y_train, w, b)

w_init = np.array([1., 1.])
b_init = 1.
alpha = 1e-1
steps = 10000
w_final, b_final = gradient_descent(x_train, y_train, w_init, b_init, alpha, steps)
print(f'Final parameters: {w_final}, {b_final}')
