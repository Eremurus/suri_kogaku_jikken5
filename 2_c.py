def f(x):
    return x**3 + 2.0*(x**2) - 5.0*x - 6.0

def dfdx(x): #f の微分
    return 3*(x**2) + 4*x - 5.0

x0_list = [-3.5, -0.5, 1.5] #x の初期値
epsilon = 0.1 ** 6 #イプシロン
for i in range(3):
    x = x0_list[i]
    k = 0
    while abs(f(x)) > epsilon:
        dx = -f(x) / dfdx(x) #ニュートン方程式を解く
        x += dx
        k += 1
    print(x, k)