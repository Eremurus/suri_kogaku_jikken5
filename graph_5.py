import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

#範囲の設定
x_range = np.linspace(-10, 10, 1000)
y_range = np.linspace(-10, 10, 1000)
x, y = np.meshgrid(x_range, y_range)

#関数f を記述
z = x**2+np.exp(x)+y**4+y**2-2.0*x*y+3.0

ax.plot_surface(x, y, z, cmap = "summer")
ax.contour(x, y, z, colors = "gray", offset = -1)  #底面に等高線を描画

ax.set_xlabel("x_0")
ax.set_ylabel("x_1")
ax.set_zlabel("f")

plt.show() 