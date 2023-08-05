import matplotlib.pyplot as plt
import numpy as np

data = {'Barton LLC': 109438.50,
        'Frami, Hills and Schmidt': 103569.59,
        'Fritsch, Russel and Anderson': 112214.71,
        'Jerde-Hilpert': 112591.43,
        'Keeling LLC': 100934.30,
        'Koepp Ltd': 103660.54,
        'Kulas Inc': 137351.96,
        'Trantow-Barrows': 123381.38,
        'White-Trantow': 135841.99,
        'Will LLC': 104437.60}
group_data = list(data.values())
group_names = list(data.keys())
group_mean = np.mean(group_data)

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

def fun6():
  def currency(x, pos) -> str:
    if x >= 1e6:
      s = '${:1.1f}M'.format(x*1e-6)
    else:
      s = '${:1.0f}K'.format(x*1e-3)
    return s
  plt.rcParams.update({'figure.autolayout':True})
  fig, ax = plt.subplots()
  ax.barh(group_names, group_data)
  labels = ax.get_xticklabels()
  plt.setp(labels, rotation=45, horizontalalignment='right')

  ax.axvline(group_mean, ls='--', color='r')
  for group in [3, 5, 8]:
    ax.text(145000, group, 'New Company', fontsize=10, verticalalignment='center')
  ax.set(xlim=[-10000, 140000],
         xlabel='Total Revenue',
         ylabel='Company',
         title='Company Revenue')
  ax.xaxis.set_major_formatter(currency)
  fig.savefig('save.png')
  plt.show()

fun6()