import numpy as np
import time

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
    x11 = 12.0 * (x1**2) + 2.0 #d^2f / dx1x1
    return np.array([[x00, x01],[x10, x11]])


def backtrack(x, d):#backtrack 法
    #パラメータ
    ita = 10.0 ** (-4)
    rou = 0.5
    t_kari = 1.0
    #反復
    while f(x + t_kari*d) > f(x) + ita * t_kari * np.sum(d*grad(x)):
        t_kari = rou * t_kari
    return t_kari

def norm(x): #ベクトルのノルムを計算
    return np.sqrt(np.sum(x ** 2))

def inverse(x):#行列x の逆行列を計算
    x00 = x[0][0]
    x01 = x[0][1]
    x10 = x[1][0]
    x11 = x[1][1]
    a00 = x11
    a01 = -x01
    a10 = -x10
    a11 = x00
    return np.array([[a00, a01],[a10, a11]]) / (a00*a11 - a01*a10)

def mat_cross_vec(A, b):#行列A とベクトルb の掛け算
    vec = []
    length = len(b)
    for i in range(length):
        vec.append(np.sum(A[i]*b))
    return np.array(vec)

epsilon = 0.1 ** 6 #イプシロン
x = np.array([1.0, 1.0]) #x の初期値
k = 0 #繰り返し回数

#反復
t1 = time.time()
while(norm(grad(x)) > epsilon):
    d = -mat_cross_vec(inverse(hesse(x)),grad(x))
    t = backtrack(x, d) #ステップサイズ
    x += (t * d) #更新
    k += 1
t2 = time.time()

print(x)
print(f(x))
print(grad(x))
print(k)
print(t2 - t1)