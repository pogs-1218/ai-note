import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_title('Z')
ax.set_xlabel('X')
ax.set_ylabel('Y')

x = np.array([1, 2, 3, 4])
y = x*2
z = x + y

ax.scatter(x, y, z, c=z)

plt.show()

