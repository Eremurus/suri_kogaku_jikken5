import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**3 + 2.0*(x**2) - 5.0*x - 6.0 #f(x) を定義

#x の定義域
x = np.arange(-10, 10, 0.1)
y = f(x)
plt.plot(x, y, 'red')
#プロットの範囲
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.axhline(0, c='black')
plt.axvline(0, c='black')
#ラベルに名前をつける
plt.xlabel('x')
plt.ylabel('y')
#グラフの名前
plt.title('Graph of y=x^3+2x^2-5x-6')
plt.show()