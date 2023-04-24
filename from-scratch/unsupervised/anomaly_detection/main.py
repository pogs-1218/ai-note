import numpy as np
import matplotlib.pyplot as plt

def estimate_gaussian(X):
  """
    Args:
      X (m, n)
    Returns:
      mu (n,)
      var (n,)
  """
  m, n = X.shape
  mu = np.sum(X, axis=0) / m
  var = np.sum((X-mu)**2, axis=0) / m
  return mu, var

def select_threshold(y_val, p_val):
  """
  """
  best_epsilon = 0
  best_F1 = 0
  F1 = 0
  step_size = (max(p_val)-min(pval)) / 1000
  for epsilon in np.arange(min(pval), max(pval), step_size):
    # which is anomal
    pred = (p_val < epsilon)
    tp = np.sum((y_val==1) & (pred==1))
    fp = np.sum((y_val==0) & (pred==1))
    fn = np.sum((y_val==1) & (pred==0))
    prec = tp / (tp+fp)
    rec = tp / (tp+fn)
    F1 = (2*prec*rec) / (prec+rec)
    if F1 > best_F1:
      best_F1 = F1
      best_epsilon = epsilon

  return best_epsilon, best_F1
