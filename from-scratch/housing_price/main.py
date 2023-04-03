import numpy as np
import matplotlib.pyplot as plt

def prepare_train():
  return (np.array([1.0, 2.0]), np.array([300.0, 500.0]))

def compute_model_output(x, w, b):
  m = x.shape[0]
  f_wb = np.zeros_like(x)
  for i in range(m):
    f_wb[i] = w * x[i] + b
  return f_wb

def compute_cost(x, t, w, b):
  m = x.shape[0]
  c = 0
  for i in range(m):
    y = w*x[i] + b
    c += (y-t[i])**2
  cost = c / (2*m)
  return cost

def compute_cost_arr(x, t, w, b):
  m = x.shape[0]
  cost = np.zeros_like(w)
  for i in range(w.shape[0]):
    c = 0
    for j in range(m):
      y = w[i]*x[j] + b
      c += (y-t[j])**2
    cost[i] = c / (2*m)
  return cost

def compute_gradient(x, t, w, b):
  '''
    x (array): input features
    t (array): target values
  '''
  m = x.shape[0]
  dj_dw = 0
  dj_db = 0
  for i in range(m):
    f_wb = w * x[i] + b
    dj_dw += (f_wb - t[i]) * x[i]
    dj_db += f_wb - t[i]
  dj_dw /= m
  dj_db /= m
  return dj_dw, dj_db

def gradient_descent(x, t, w_init, b_init, alpha, num_iters, cost_function, grad_function):
  w, b = w_init, b_init
  for i in range(num_iters):
    dj_dw, dj_db = grad_function(x, t, w, b)
    w -= alpha * dj_dw
    b -= alpha * dj_db
    cost = cost_function(x, t, w, b)
    if i % 1000 == 0:
# print(f'params: {w: 0.3e}, {b: 0.3e}')
      print(f'[{i}] cost:   {cost: 0.3e}')
  return w, b

def compute_gradient_vec(x, t, w, b):
  m = x.shape[0]
  f_wb = np.dot(w, x) + b
  dj_dw = (f_wb - t) * x
  dj_db = f_wb - t

def gradient_descent_vec(x, t, w_init, b_init, alpha, num_iters, cost_function, grad_function):
  w, b = w_init, b_init
  m = x.shape[0]
  for i in range(num_iters):
    f_wb = np.dot(x, w) + b 
    assert f_wb == x.shape[0]
    dj_dw = (f_wb - t) * x
    dj_db = f_wb - t
    dj_dw = np.sum(dj_dw) / m
    dj_db = np.sum(dj_db) / m
    w -= alpha * dj_dw
    b -= alpha * dj_db
    
def show(x, t, y):
  plt.title('Housing Prices')
  plt.plot(x, y)
  plt.scatter(x, t, marker='x')
  plt.ylabel('Price')
  plt.xlabel('Size')
  plt.show()

def show_with_cost(x, t, y, w, b, cost):
  fig, ax = plt.subplots(1, 2, figsize=(10, 5)) 
  # display prediction with target
  ax[0].set_title('Prediction, Train')
  ax[0].set_xlabel('x')
  ax[0].set_ylabel('y')
  ax[0].scatter(x, t, marker='x')
  ax[0].plot(x, y)
  
  # Display cost function
  range_w = np.arange(0, 500, 1, dtype=int)
  range_cost = compute_cost_arr(x, t, range_w, b)
  ax[1].set_title('Cost')
  ax[1].set_xlabel('W')
  ax[1].set_ylabel('J')
  ax[1].plot(range_w, range_cost)
  ax[1].scatter(w, cost, marker='x', c='red')

  plt.show()

x, t = prepare_train()
w, b = 200, 100

print(f'train values: {x}, {t}')
print(f'params: {w}, {b}')
y = compute_model_output(x, w, b)
print(f'{y}')

cost = compute_cost(x, t, w, b)
print(f'{cost}')
#show(x, t, y)
#show_with_cost(x, t, y, w, b, cost)

w = 0
b = 0
learning_rate = 1e-2
step = 10000
w_result, b_result = gradient_descent(x, t, w, b, learning_rate, step, compute_cost, compute_gradient)
print(f'final parameters: {w_result}, {b_result}')

x_test = np.array([1.0, 1.2, 2.0])
result = compute_model_output(x_test, w_result, b_result)
print(f'test: {x_test}')
print(f'prediction: {result}')
