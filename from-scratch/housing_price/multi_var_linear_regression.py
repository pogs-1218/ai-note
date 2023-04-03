import numpy as np

def predict_single_loop(x, w, b):
  n = x.shape[0]
  p = 0
  for i in range(n):
    f_wb = x[i] * w[i] # w1x1 -> w2x2
    p += f_wb
  p += b
  return p

def predict(x, w, b):
  return np.dot(x, w) + b

def compute_cost(x, y, w, b):
  m = x.shape[0]
  cost_sum = 0
  for i in range(m):
    cost_sum += (predict(x[i], w, b) - y[i])**2
  cost = cost_sum / (2 * m)
  return cost

def compute_gradient(x, y, w, b):
  m = x.shape[0]
  dj_dw = np.zeros(x.shape[1])
  dj_db = 0
  for i in range(m):
    p = predict(x[i], w, b)
    for j in range(x.shape[1]):
      dj_dw[j] += (p - y[i]) * x[i, j]
    dj_db += p - y[i]
  dj_dw /= m
  dj_db /= m
  return dj_dw, dj_db

def gradient_descent(x, y, w_init, b_init, alpha, num_iters, cost_func, grad_func):
  w, b = w_init, b_init
  for i in range(num_iters):
    dj_dw, dj_db = grad_func(x, y, w_init, b_init)
    w -= alpha * dj_dw
    b -= alpha * dj_db
    cost = cost_func(x, y, w, b)
    if i % 100 == 0:
      print(f'{i} === {cost}')
  return w, b

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])

print(f'{X_train.shape[0]}')
print(f'{X_train[0,:]}')
#p = predict_single_loop(X_train[0,:], w_init, b_init)
p = predict(X_train[0,:], w_init, b_init)
print(f'predict: {p}')

cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'cost: {cost}')

dj_dw, dj_db = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_dw: {dj_dw}, dj_db: {dj_db}')

init_w = np.zeros_like(w_init)
init_b = 0.
iters = 1000
alpha = 5.0e-7
w_final, b_final = gradient_descent(X_train, y_train, init_w, init_b, alpha, iters, compute_cost, compute_gradient)
print(f'final w: {w_final}')
print(f'final b: {b_final}')

m = X_train.shape[0]
for i in range(m):
  p = predict(X_train[i], w_final, b_final)
  print(f'predict: {p}, target: {y_train[i]}')


