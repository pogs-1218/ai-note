import numpy as np

x = np.array([200, 17], dtype=np.float32)
w = np.array([[1, -3, 5],
              [-2, 4, -6]], dtype=np.float32)
b = np.array([-1, 1, 2], dtype=np.float32)

def sigmoid(z):
  return 1 / (1 + np.exp(-z)) 

def loop(x, w, b):
  # w's is (2, 3) matrix.
  units = w.shape[1]
  # output should be (3,)
  a = np.zeros(units)
  for j in range(units):
    # https://numpy.org/doc/stable/reference/generated/numpy.dot.html
    z = np.dot(x, w[:,j]) + b[j]
    a[j] = sigmoid(z)
  return a
    
def vec(x, w, b):
  x = x.reshape(1, -1) # x.shape=(1, 2)
  b = b.reshape(1, -1) # b.shape=(1, 3)
  # w.shape(2, 3)
  # https://numpy.org/doc/stable/reference/generated/numpy.matmul.html#numpy.matmul
  z = np.matmul(x, w) + b
  #z = x @ w + b
  a = sigmoid(z)
  return a

print(loop(x, w, b))
print(vec(x, w, b))
