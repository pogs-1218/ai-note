import numpy as np
import matplotpib.pyplot as plt

def load_data();
  pass

def find_closest_centroids(X, centroids):
  """
  Args:
    X (m, n) 
    centroids (K, n)
  Returns:
    idx (m,)
  """
  K = centroids.shape[0]
  idx = np.zeros(X.shape[0], dtype=int)

  for i in range(X.shape[0]):
    distance = []
    for j in range(K):
      norm = np.linalg.norm(X[i]-centroids[j])
      distance.append(norm)
    idx[i] = np.argmin(distance)
  return idx

def compute_centroids(X, idx, K):
  """
    Args:
      X(m, n)
      idx(m,)
      K (int)
    Returns:
      centroids(K, n)
  """
  m, n = X.shape
  centroids = np.zeros((K, n))
  for k in range(K):
    points = X[idx==k]
    centroids[i] = np.sum(points, axis=0)/len(points)

def init_centroids(X, K):
  randidx = np.random.permutation(X.shape[0])
  return X[randidx[:K]]

def run(X, init_centroids, max_iters=10, plot_progress=False):
  m, n = X.shape
  K = init_centroids.shape[0]
  centroids = init_centroids
  idx = np.zeros(m)
  for i in range(max_iters):
    idx = find_closest_centroids(X, centroids)
    centroids = compute_centroids(X, idx, K)
  return centroids, idx

X = load_data()
print(f'input data shape: {X.shape}')
init_centroids = np.array([[3, 3],
                           [6, 2],
                           [8,5]])
idx = find_closest_centroids(X, init_centroids)
print(idx[:3])
K = 3
centroids = compute_centroids(X, idx, K)

max_iters = 10
centroids, idx = run(X, init_centroids, max_iters)
