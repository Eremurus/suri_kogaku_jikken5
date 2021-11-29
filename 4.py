import numpy as np

def f(x):#関数 f(x)を定義
    x0 = x[0]
    x1 = x[1]
    return x0 ** 2 + np.exp(x0) + x1 ** 4 + x1 ** 2 - 2.0*x0*x1 + 3.0

def grad(x):#f(x) の勾配ベクトルを返す
    x0 = x[0]
    x1 = x[1]
    dfdx0 = 2.0 * x0 + np.exp(x0) - 2.0*x1 #x0 に関して微分
    dfdx1 = 4.0 * (x1 ** 3) + 2.0*x1 - 2.0*x0 #x1 に関して微分
    return np.array([dfdx0, dfdx1])

def hesse(x):#f(x) のhesse 行列を返す
    x0 = x[0]
    x1 = x[1]
    x00 = 2.0 + np.exp(x0) #d^2f/dx0^2
    x01 = -2.0 #d^2f/ dx0dx1
    x10 = -2.0 #d^2f / dx1x0
    x11 = 12.0 * x1 + 2.0 #d^2f / dx1x1
    return np.array([[x00, x01],[x10, x11]])
