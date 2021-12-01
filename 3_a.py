def f(x):#関数f
    return (x ** 3)/3.0 - x ** 2 - 3.0*x + 5.0/3.0

def dfdx(x):#f の微分
    return x ** 2 - 2.0*x - 3.0

epsilon = 0.1 ** 6 #イプシロン
x = 1.0 / 2.0 #x の初期値
k = 0 #繰り返し回数

while(abs(dfdx(x)) > epsilon):
    d = -dfdx(x)
    t = 1.0/(k+1) #ステップサイズ
    x += (t * d) #更新
    k += 1

print(x)
print(dfdx(x))
print(k)