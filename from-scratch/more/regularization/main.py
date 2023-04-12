import numpy as np
import matplotlib.pyplot as plt

def compute_cost_linear_reg(x_train, y_train, w, b, lambda_=1):
  m, n = x_train.shape
  cost = 0
  for i in range(m):
    f = np.dot(x_train[i], w) + b
    cost += (f - y_train[i]) ** 2
  cost /= 2*m

  # add regularized term
  # Including this term encourages gradient descent to minimize the size of the parameters.
  # but... how?? why? -> doesn't understand well..
  r = 0
  for j in range(n):
    r += w[j] ** 2
  r *= (lambda_ / (2 * m))
    
  cost += r
  return cost 

# similar with logistic regression.

def compute_gradient_linear_reg(x_train, y_train, w, b, lambda_=1):
  m, n = x_train.shape
  dj_dw = np.zeros_like(w)
  dj_db = 0.

  for i in range(m):
    err = (np.dot(x_train[i], w) + b) - y_train[i]
    for j in range(n):
      dj_dw[j] += err * x_train[i,j]
    dj_db += err
  dj_dw /= m
  dj_db /= m

  # add regularized term
  for j in range(n):
    dj_dw[j] += (lambda_ / m) * w[j]

  return (dj_dw, dj_db)

def test_cost():
  np.random.seed(1)
  x_tmp = np.random.rand(5, 6)
  y_tmp = np.array([0, 1, 0, 1, 0])
  w_tmp = np.random.rand(x_tmp.shape[1]).reshape(-1,) - 0.5
  b_tmp = 0.5
  lambda_tmp = 0.7
  cost_tmp = compute_cost_linear_reg(x_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
  print(f'{cost_tmp}')

def test_gradient():
  np.random.seed(1)
  x_tmp = np.random.rand(5, 3)
  y_tmp = np.array([0, 1, 0, 1, 0])
  w_tmp = np.random.rand(x_tmp.shape[1])
  b_tmp = 0.5
  lambda_tmp = 0.7
  dj_dw_tmp, dj_db_tmp = compute_gradient_linear_reg(x_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
  print(f'{dj_dw_tmp}')
  print(f'{dj_db_tmp}')


test_cost()
