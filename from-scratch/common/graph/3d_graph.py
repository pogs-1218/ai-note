import matplotlib.pyplot as plt
import numpy as np

def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig, axs = plt.subplots(ncols=1, subplot_kw={'projection':'3d'})
axs.contour3D(X, Y, Z, 50, cmap='binary')
axs.set_title('Z')
axs.set_xlabel('X')
axs.set_ylabel('Y')


plt.show()

