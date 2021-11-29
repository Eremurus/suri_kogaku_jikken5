def f(x):
    return x**3 + 2.0*(x**2) - 5.0*x - 6.0

epsilon = 0.1**6 #終了条件
a_and_b = [[-4.0,-2.5],[ -2.5,0.5],[1.5,3.0]] #初期点

for i in range(3):
    a = a_and_b[i][0] #a の値
    b = a_and_b[i][1] #b の値
    c = 5.0 #f(c) がイプシロンより大きくなる初期値
    k = 0

    #反復
    while abs(f(c)) > epsilon:
        c = (a + b) / 2.0
        if f(c) < 0:
            a = c
        elif f(c) >= 0:
            b = c
        k += 1

    #解の出力
    print(c, k)
