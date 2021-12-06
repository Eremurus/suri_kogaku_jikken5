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
file = open('7.txt', 'w')
file_grad = open('7_grad.txt','w')

for n in ([2,5,10]):
    k_newton = 0
    k_gd = 0
    t_list = []
    data_list = ["\\begin{table}[ht] \n \centering \n \caption[課題7、最急降下法の$n=$",\
        str(n),"の表]{","$n=",str(n),"$の時の最急降下法の結果.上から順に1回目から5回目まで\
            の計算で得られた停留点(最適解)と最適値の値を記している.下の二つの項目は収束までに要した\
                反復回数$k$の平均とかかった時間の平均である.}\n \\begin{tabular}[ht]{|c|c|} \n \hline \n"]
    if n == 10:
        data_grad_list = ["\\begin{table}[ht] \n \centering \n \caption[課題7、最急降下法、$n=10$の最適値の表]{","$n=10$の時の最急降下法の結果.上から順に1回目から5回目まで\
                の計算で得られた最適値の値を記している.}\n \\begin{tabular}[ht]{|c|c|} \n \hline \n"]
        file_grad.writelines(data_grad_list)
    file.writelines(data_list)
    #5 回繰り返す
    for t_ in range(5):
        x = np.array([1.0 for _ in range(n)])
        Z = np.random.rand(n, n)# 一様分布から乱数生成
        A = np.dot(np.transpose(Z), Z)# Z^T Z

        t1 = time.time()
        #最急降下法
        while(norm(grad(x, A)) > epsilon):
            d = -grad(x, A)
            t = backtrack(x, d, A) #ステップサイズ
            x += (t * d) #更新
            k_gd += 1
        t2 = time.time()
        t_list.append(t2 - t1)
        
        kari_hairetu = []
        for l in range(n):
            kari_hairetu.append(x[l])
        file.write("\multicolumn{2}{|c|}{最急降下法")
        file.write(str(t_+1))
        file.write("回目の計算} \\\ \n \hline \n")
        file.write("停留点 & \\begin{tabular}{c}")
        file.write("(")
        for i in range(len(kari_hairetu)):
            if i!=len(kari_hairetu)-1:
                if i%2 == 0:
                    file.writelines([str(kari_hairetu[i]),","])
                else:
                    file.writelines([str(kari_hairetu[i]),",","\\\\"])
            else:
                file.write(str(kari_hairetu[i]))
        file.write(") \n")
        file.write("\end{tabular} \\\ \n \hline \n")

        if n != 10:
            file.write("最適値 &\\begin{tabular}{c}")
            file.write(str(f(x, A)))
            file.write("\end{tabular} \\\ \n \hline \n")
        
        if n == 10:
            file_grad.write("最適値 &\\begin{tabular}{c}")
            file_grad.write(str(f(x,A)))
            file_grad.write("\end{tabular} \\\ \n \hline \n")               
    data_list = ["\multicolumn{2}{|c|}{$n=",str(n),"$の時の最急降下法の$k$の平均} \\\ \n \hline \n" ,\
        "\multicolumn{2}{|c|}{",str(k_gd / 5.0),"}\\\ \n \hline \n"]
    file.writelines(data_list)
    data_list = ["\multicolumn{2}{|c|}{$n=",str(n),"$の時の最急降下法の平均時間} \\\ \n \hline \n",\
        "\multicolumn{2}{|c|}{",str(np.mean(np.array(t_list))),"}\\\ \n  \hline \n \\end{tabular}  \n \\end{table} \n"]
    file.writelines(data_list)

    if n==10:
        data_grad_list = ["\\end{tabular}  \n \\end{table} \n"]
        file_grad.writelines(data_grad_list)

    t_list = []
    data_list = ["\\begin{table}[ht] \n \centering \n \caption[課題7、ニュートン法の$n=$",str(n),"の表]{",\
        "$n=",str(n),"$の時のニュートン法の結果.上から順に1回目から5回目までの計算で得られた停留点(最適解)と最適値\
            の値を記している.下の二つの項目は収束までに要した反復回数$k$の平均とかかった時間の平均である.\
                }\n \\begin{tabular}[ht]{|c|c|} \n \hline \n"]
    if n == 10:
        data_grad_list = ["\\begin{table}[ht] \n \centering \n \caption[課題7、ニュートン法、$n=10$の最適値の表]{","$n=10$の時のニュートン法の結果.上から順に1回目から5回目まで\
                の計算で得られた最適値の値を記している.}\n \\begin{tabular}[ht]{|c|c|} \n \hline \n"]
        file_grad.writelines(data_grad_list)
    file.writelines(data_list)
    for t_ in range(5):#繰り返す
        x = np.array([1.0 for _ in range(n)])
        Z = np.random.rand(n, n)
        A = np.dot(np.transpose(Z), Z)
        
        t1 = time.time()
        #修正ニュートン法
        while(norm(grad(x, A)) > epsilon):
            w,v = np.linalg.eig(hesse(x, A))
            tau = abs(np.min(w)) + 10.0 ** (-2)
            d = -np.dot(np.linalg.inv(hesse(x, A)+tau*np.eye(n)),grad(x, A))
            t = backtrack(x, d, A) #ステップサイズ
            x += (t * d) #更新
            k_newton += 1
        t2 = time.time()
        t_list.append(t2 - t1)

        #ファイルに書き出す
        kari_hairetu = []
        for l in range(n):
            kari_hairetu.append(x[l])
        file.write("\multicolumn{2}{|c|}{ニュートン法")
        file.write(str(t_+1))
        file.write("回目の計算} \\\ \n \hline \n")
        file.write("停留点 & \\begin{tabular}{c}")
        file.write("(")
        for i in range(len(kari_hairetu)):
            if i!=len(kari_hairetu)-1:
                if i%2 == 0:
                    file.writelines([str(kari_hairetu[i]),","])
                else:
                    file.writelines([str(kari_hairetu[i]),",","\\\\"])
            else:
                file.write(str(kari_hairetu[i]))
        file.write(") \n")
        file.write("\end{tabular} \\\ \n \hline \n")

        #n=10 の時は最適解は別で出力(見づらいので)
        if n != 10:
            file.write("最適値 &\\begin{tabular}{c}")
            file.write(str(f(x,A)))
            file.write("\end{tabular} \\\ \n \hline \n")

        #n=10 の勾配を出力
        if n == 10:
            file_grad.write("最適値 &\\begin{tabular}{c}")
            file_grad.write(str(f(x, A)))
            file_grad.write("\end{tabular} \\\ \n \hline \n")        

    data_list = ["\multicolumn{2}{|c|}{$n=",str(n),"$の時のニュートン法の$k$の平均}\
        \\\ \n \hline \n" ,"\multicolumn{2}{|c|}{",str(k_newton / 5.0),"}\\\ \n \hline \n"]
    file.writelines(data_list)
    data_list = ["\multicolumn{2}{|c|}{$n=",str(n),"$の時のニュートン法の平均時間} \\\ \n\
        \hline \n","\multicolumn{2}{|c|}{",str(np.mean(np.array(t_list))),"}\\\ \n  \hline\
        \n \\end{tabular}  \n \\end{table} \n"]
    file.writelines(data_list)

    if n==10:
        data_grad_list = ["\\end{tabular}  \n \\end{table} \n"]
        file_grad.writelines(data_grad_list)


file.close()
file_grad.close()