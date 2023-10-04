import numpy as np

def calculate_return(a, s, g):
  result = [None]*len(a)
  term_idx = [0, 5]
  for i in range(len(result)):
    tmp = 0
    for j in range(0, s):
      g = np.power(g, i)   
      tmp += g*a[i]
    print(f'{i}={tmp}')
    result[i] = tmp

a = np.array([100, 0, 0, 0, 0, 40])
g = 0.5
s = 4
calculate_return(a, s, g)
