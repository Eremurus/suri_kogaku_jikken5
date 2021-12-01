import numpy as np
import time

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

epsilon = 0.1 ** 6 #イプシロン

for n in ([2,5,10]):
    k_newton = 0
    k_gd = 0
    t_list = []
    for t_ in range(5):
        x = np.array([1.0 for _ in range(n)])
        Z = np.random.rand(n, n)
        A = np.dot(np.transpose(Z), Z)

        t1 = time.time()
        while(norm(grad(x, A)) > epsilon):
            d = -grad(x, A)
            t = backtrack(x, d, A) #ステップサイズ
            x += (t * d) #更新
            k_gd += 1
        t2 = time.time()
        t_list.append(t2 - t1)
        print("最急降下法",t_+1,"回目の計算","...","n:",n,"停留点:",x,"\n")
        print(grad(x, A))
    print(n,"次の最急降下法のkの平均は ",k_gd / 5.0,"\n")
    print(n,"次の","最急降下法の平均時間は ",np.mean(np.array(t_list)),"\n")

    t_list = []
    for t_ in range(5):
        x = np.array([1.0 for _ in range(n)])
        Z = np.random.rand(n, n)
        A = np.dot(np.transpose(Z), Z)
        
        t1 = time.time()
        while(norm(grad(x, A)) > epsilon):
            w,v = np.linalg.eig(hesse(x, A))
            tau = abs(np.min(w)) + 10.0 ** (-2)
            d = -np.dot(np.linalg.inv(hesse(x, A)+tau*np.eye(n)),grad(x, A))
            t = backtrack(x, d, A) #ステップサイズ
            x += (t * d) #更新
            k_newton += 1
        t2 = time.time()
        t_list.append(t2 - t1)
        print("ニュートン法",t_+1,"回目の計算","...","n:",n,"停留点:",x, "\n")
        print(grad(x, A))
    print(n, "次のニュートン法のkの平均は",k_newton / 5.0, "\n")
    print(n,"次の","ニュートン法の平均時間は ",np.mean(np.array(t_list)), "\n")
