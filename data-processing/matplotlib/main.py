'''
provide implicit API
state-based interface
ltiple graphs
'''
import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [4, 7, 5]

x2 = [1, 2, 3]
y2 = [10, 14, 12]

plt.plot(x, y, label='First Line')
plt.plot(x2, y2, label='Second Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
# plt.legend()
plt.show()