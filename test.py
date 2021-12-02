l = [1,2,3,4,5]
dora = ""
for i in range(5):
    dora += str(l[i])
    if i % 2 != 0:
        dora += "\\\\"
print(dora)