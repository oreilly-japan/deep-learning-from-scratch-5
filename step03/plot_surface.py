import numpy as np
import matplotlib.pyplot as plt


X = np.array([[-2, -1, 0, 1, 2],
              [-2, -1, 0, 1, 2],
              [-2, -1, 0, 1, 2],
              [-2, -1, 0, 1, 2],
              [-2, -1, 0, 1, 2]])
Y = np.array([[-2, -2, -2, -2, -2],
              [-1, -1, -1, -1, -1],
              [0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1],
              [2, 2, 2, 2, 2]])
Z = X ** 2 + Y ** 2

ax = plt.axes(projection='3d')  # projection='3d'により3d用グラフを指定
ax.plot_surface(X, Y, Z, cmap='jet')  # cmap='jet'によりグラフの色を指定
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()