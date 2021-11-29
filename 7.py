import numpy as np
import random

def f(x, A):#関数 f(x)を定義
    return np.dot(np.dot(A, x), x) / 2.0

def grad(x, A):#f(x) の勾配ベクトルを返す
    return np.dot(A, x)

def hesse(x, A):#f(x) のhesse 行列を返す
    return A

def backtrack(x, d, A):#backtrack 法
    #パラメータ
    ita = 10.0 ** (-4)
    rou = 0.5
    t_kari = 1.0
    #反復
    while f(x + t_kari*d, A) > f(x, A) + ita * t_kari * np.sum(d*grad(x, A)):
        t_kari = rou * t_kari
    return t_kari

def norm(x): #ベクトルのノルムを計算
    return np.sqrt(np.sum(x ** 2))

def mat_cross_vec(A, b):#行列A とベクトルb の掛け算
    vec = []
    length = len(b)
    for i in range(length):
        vec.append(np.sum(A[i]*b))
    return np.array(vec)

epsilon = 0.1 ** 6 #イプシロン

for n in ([2,5,10]):
    k_newton = 0
    k_gd = 0
    for t in range(5):
        x = np.random.rand(n)
        Z = np.random.rand(n, n)
        A = np.dot(np.transpose(Z), Z)

        while(norm(grad(x, A)) > epsilon):
            d = -grad(x, A)
            t = backtrack(x, d, A) #ステップサイズ
            x += (t * d) #更新
            k_gd += 1
        print(x)
        print(grad(x, A))
    print(n,"次最急降下法のkの平均は ",k_gd / 5.0,"\n")

    for t in range(5):
        x = np.random.rand(n)
        Z = np.random.rand(n, n)
        A = np.dot(np.transpose(Z), Z)
        
        while(norm(grad(x, A)) > epsilon):
            d = -mat_cross_vec(np.linalg.inv(hesse(x, A)),grad(x, A))
            t = backtrack(x, d, A) #ステップサイズ
            x += (t * d) #更新
            k_newton += 1
        print(x)
        print(grad(x, A))
    print(n, "次のニュートン法のkの平均は",k_newton / 5.0, "\n")
