l = [1,2,3,4,5]
a = []
for i in range(5):
    a.append(str(i)+"\\\\")
print(a)
for i in range(5):
    print(a[i])

print("(")
for i in range(5):
    if i%2 != 0:
        print(str(l[i]),",","\\\\",end='')
    else:
        print(str(l[i]), ",",end='')
print(")")