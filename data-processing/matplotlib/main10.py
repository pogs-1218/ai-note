import matplotlib.pyplot as plt
import numpy as np

def fun1():
    data = {
        'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)
    }
    data['b'] = data['a'] + 10 * np.random.randn(50)
    data['d'] = np.abs(data['d']) * 100
    plt.scatter('a', 'b', c='c', s='d', data=data)
    plt.show()

def fun2():
  x = [1, 2, 3]
  y = [1, 2, 3]

  plt.plot(np.arange(0, 5, 0.2), 'ro')
  plt.axis([0, 5, 0, 10])
  plt.show()

def fun3():
   '''
     categorical variables
   '''
   names = ['group_a', 'group_b', 'group_c']
   values = [1, 10, 100]
   plt.figure(figsize=(9,3))
   ax = plt.subplot(131)
   plt.bar(names, values)
   plt.subplot(132, sharex=ax)
   plt.scatter(names, values)
   plt.subplot(133, sharex=ax)
   plt.plot(names, values)
   plt.show()

def fun4():
  mu, sigma = 100, 15
  x = mu + sigma * np.random.randn(10000)
  n, bins, patches = plt.hist(x, 50, density=True, facecolor='g', alpha=0.75)

  # https://matplotlib.org/stable/tutorials/introductory/pyplot.html#using-mathematical-expressions-in-text
  plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
  plt.axis([40, 160, 0, 0.03])
  plt.grid()
  plt.show()


def fun5():
  plt.subplot()
  t = np.arange(0.0, 5.0, 0.01)
  s = np.cos(2*np.pi*t)
  line, = plt.plot(t, s, lw=2)
  plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5), arrowprops={'facecolor':'black', 'shrink':0.05})
  plt.ylim(-2, 2)
  plt.show()

fun5()