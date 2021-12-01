import numpy as np
import time

y = np.array([1.5, 2.25, 2.625])

def f(x):#関数 f(x)を定義
    f = 0.0
    x0 = x[0]
    x1 = x[1]
    for i in range(3):
        f += (y[i]-x0*(1.0-x1**(i+1)))**2
    return f

def grad(x):#f(x) の勾配ベクトルを返す
    fi = []
    x0 = x[0]
    x1 = x[1]
    sum_0 =  0.0
    sum_1 = 0.0
    for i in range(3):
        fi.append(y[i]-x0*(1.0-x1**(i+1)))
    for i in range(3):
        dfdx0 =  x1**(i+1)-1#x0 に関して微分
        dfdx1 = (i+1)*x0*(x1**i) #x1 に関して微分
        sum_0 += (dfdx0*fi[i])
        sum_1 += (dfdx1*fi[i])

    return np.array([sum_0*2, sum_1*2])

def hesse(x):#f(x) のhesse 行列を返す
    x0 = x[0]
    x1 = x[1]
    ans = np.array([[0.0, 0.0],[0.0, 0.0]])

    for i in range(3):
        fi = y[i] - x0*(1.0 - x1**(i+1))
        dfidx0 =  x1**(i+1)-1#x0 に関して微分
        dfidx1 = (i+1)*x0*(x1**i) #x1 に関して微分
        d2fidx0x0 = 0.0
        d2fidx0x1 = (i+1)*(x1**i)
        d2fidx1dx0 = (i+1)*(x1**i)
        if i == 0:
            d2fidx1dx1 = 0.0
        else:
            d2fidx1dx1 = i * (i+1) * x0 * (x1 ** (i-1))

        A = np.array([[0.0, 0.0],[0.0, 0.0]])
        B = np.array([[0.0, 0.0],[0.0, 0.0]])

        A[0][0] = fi * d2fidx0x0
        A[0][1] = fi * d2fidx0x1
        A[1][0] = fi * d2fidx1dx0
        A[1][1] = fi * d2fidx1dx1
        B[0][0] = dfidx0*dfidx0
        B[0][1] = dfidx0*dfidx1
        B[1][0] = dfidx1*dfidx0
        B[1][1] = dfidx1*dfidx1

        ans += (A + B)*2.0
    return ans

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

epsilon = 0.1 ** 6 #イプシロン

#最急降下法
x = np.array([2.0, 0.0]) #x の初期値
k = 0 #繰り返し回数

t1 = time.time()
while(norm(grad(x)) > epsilon):
    d = -grad(x)
    t = backtrack(x, d) #ステップサイズ
    x += (t * d) #更新
    k += 1
t2 = time.time()

print(x)
print(grad(x))
print(k)
print(t2 - t1)
print(f(x))

#ニュートン法
x = np.array([2.0, 0.0]) #x の初期値
k = 0 #繰り返し回数

t1 = time.time()
while(norm(grad(x)) > epsilon):
    w,v = np.linalg.eig(hesse(x))
    tau = abs(np.min(w)) + 10.0 ** (-2)
    
    d = -np.dot(np.linalg.inv(hesse(x)+tau*np.eye(2)),grad(x))
    t = backtrack(x, d) #ステップサイズ
    x += (t * d) #更新
    k += 1
t2 = time.time()

print(x)
print(grad(x))
print(k)
print(t2 - t1)
print(f(x))