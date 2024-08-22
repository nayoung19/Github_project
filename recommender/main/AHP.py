import numpy as np


class AHP:
    a = np.array([[1, 1/3], [3, 1]])
    # c2 = np.array([[1, 1 / 3, 1 / 2, 3, 5], [3, 1, 1, 2, 2], [2, 1, 1, 2, 2], [1 / 3, 1 / 2, 1 / 2, 1, 1],
    #                [1 / 3, 1 / 2, 1 / 2, 1, 1]])
    c2 = np.array([[1, 1 / 2, 1 / 2, 3, 5], [2, 1, 1, 3, 2], [2, 1, 1, 3, 3], [1 / 3, 1 / 3, 1 / 3, 1, 2],
                   [1 / 5, 1 / 2, 1 / 3, 1 / 2, 1]])
    c3 = np.array([[1, 3, 1 / 2], [1 / 3, 1, 1 / 3], [2, 3, 1]])
    w1 = np.linalg.eig(c3)  # 返回特征值和特征向量
    a1_max = np.max(w1[0])
    t = np.argwhere(w1[0] == a1_max)  # 寻找最大特征值所在的行和列
    RILIST = [0, 0, 0, 0.52, 0.89, 1.12, 1.26]
    n1 = c3.shape[0]
    RI1 = RILIST[n1]
    CI1 = (a1_max - n1) / (n1 - 1)
    CR1 = CI1 / RI1
    print(CR1)
    print("矩阵一致性可接受") if CR1 < 0.1 else print("矩阵一致性不可接受")

    '''
    算数平均值法a
    '''
    b = np.sum(a, axis=1)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a[j][i] /= b[j]
    print(np.average(a, axis=0))

    '''
    算数平均值法c2
    '''
    b = np.sum(c2, axis=1)
    for i in range(c2.shape[0]):
        for j in range(c2.shape[1]):
            c2[j][i] /= b[j]
    '''
    算数平均值法c3
    '''
    b = np.sum(c3, axis=1)
    for i in range(c3.shape[0]):
        for j in range(c3.shape[1]):
            c3[j][i] /= b[j]

    w1 = []
    w2 = []
    w3 = []
    for i in range(len(np.average(c2, axis=0))):
        w2.append(np.average(c2, axis=0)[i] * np.average(a, axis=0)[0])
    for i in range(len(np.average(c3, axis=0))):
        w3.append(np.average(c3, axis=0)[i] * np.average(a, axis=0)[1])

    print(w2)
    print(w3)
