import os
import random

import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt


class Smote(object):
    def __init__(self, N=50, k=5, r=2):
        # 初始化self.N, self.k, self.r, self.newindex
        self.N = N  # MOTE算法合成的样本数量占原少数类样本的百分比N%
        self.k = k  # 最近邻的个数k
        # self.r是距离决定因子
        self.r = r  # 最近邻算法采用欧式距离
        # self.newindex用于记录SMOTE算法已合成的样本个数
        self.newindex = 0  # 记录SMOTE算法已合成的样本个数
        self.numattrs = 5  # 少数类样本的特征个数
        self.T = 0  # 少数类样本个数
        self.samples = None
        self.synthetic = None

    # 构建训练函数
    def fit(self, samples):
        # 初始化self.samples, self.T, self.numattrs
        self.samples = samples
        # self.T是少数类样本个数，self.numattrs是少数类样本的特征个数
        self.T, self.numattrs = self.samples.shape

        # 查看N%是否小于100%
        if self.N < 100:
            # 如果是，随机抽取N*T/100个样本，作为新的少数类样本
            np.random.shuffle(self.samples)
            self.T = int(self.N * self.T / 100)
            self.samples = self.samples[0:self.T, :]
            # N%变成100%
            self.N = 100

        # 查看从T是否不大于近邻数k
        if self.T <= self.k:
            # 若是，k更新为T-1
            self.k = self.T - 1

        # 令N是100的倍数
        N = int(self.N / 100)
        # 创建保存合成样本的数组
        self.synthetic = np.zeros((self.T * N, self.numattrs))

        # 调用并设置k近邻函数
        neighbors = NearestNeighbors(n_neighbors=self.k + 1,
                                     algorithm='ball_tree',
                                     p=self.r).fit(self.samples)

        # 对所有输入样本做循环
        for i in range(len(self.samples)):
            # 调用kneighbors方法搜索k近邻
            nnarray = neighbors.kneighbors(self.samples[i].reshape((1, -1)),
                                           return_distance=False)[0][1:]

            # 把N,i,nnarray输入样本合成函数self.__populate
            self.__populate(N, i, nnarray)

        # 最后返回合成样本self.synthetic
        return self.synthetic

    # 构建合成样本函数
    def __populate(self, N, i, nnarray):
        # 按照倍数N做循环
        for j in range(N):
            # attrs用于保存合成样本的特征
            attrs = []
            # 随机抽取1～k之间的一个整数，即选择k近邻中的一个样本用于合成数据
            nn = random.randint(0, self.k - 1)

            # 计算差值
            diff = self.samples[nnarray[nn]] - self.samples[i]
            # 随机生成一个0～1之间的数
            gap = random.uniform(0, 1)
            # 合成的新样本放入数组self.synthetic
            self.synthetic[self.newindex] = self.samples[i] + gap * diff

            # self.newindex加1， 表示已合成的样本又多了1个
            self.newindex += 1


if __name__ == '__main__':
    s = Smote()
    user = []
    level = []
    score1 = pd.read_csv('../files/user_ability.csv', encoding='utf-8')
    score2 = pd.read_csv('../files/read_ability.csv', encoding='utf-8')
    levels = pd.read_csv('../files/user_score.csv', encoding='utf-8')
    for r1, r2, r3 in zip(score1.iterrows(), score2.iterrows(), levels.iterrows()):
        user.append([r1[1][2], r1[1][3], r1[1][4], r1[1][5], r1[1][6], r2[1][2], r2[1][3], r2[1][4], r3[1][3]])
    for r1, r2, r3 in zip(score1.iterrows(), score2.iterrows(), levels.iterrows()):
        user.append([r1[1][2], r1[1][3], r1[1][4], r1[1][5], r1[1][7], r2[1][2], r2[1][3], r2[1][4], r3[1][5]])
    for r1, r2, r3 in zip(score1.iterrows(), score2.iterrows(), levels.iterrows()):
        user.append([r1[1][2], r1[1][3], r1[1][4], r1[1][5], r1[1][8], r2[1][2], r2[1][3], r2[1][4], r3[1][7]])
    for r1, r2, r3 in zip(score1.iterrows(), score2.iterrows(), levels.iterrows()):
        user.append([r1[1][2], r1[1][3], r1[1][4], r1[1][5], r1[1][9], r2[1][2], r2[1][3], r2[1][4], r3[1][9]])
    for r1, r2, r3 in zip(score1.iterrows(), score2.iterrows(), levels.iterrows()):
        user.append([r1[1][2], r1[1][3], r1[1][4], r1[1][5], r1[1][10], r2[1][2], r2[1][3], r2[1][4], r3[1][11]])
    for r1, r2, r3 in zip(score1.iterrows(), score2.iterrows(), levels.iterrows()):
        user.append([r1[1][2], r1[1][3], r1[1][4], r1[1][5], r1[1][11], r2[1][2], r2[1][3], r2[1][4], r3[1][13]])
    sample = np.array(user)
    smote = Smote(N=325)
    synthetic_points = smote.fit(sample)
    for i in range(synthetic_points.shape[0]):
        for j in range(7):
            synthetic_points[i, j] = round(synthetic_points[i, j], 4)
        synthetic_points[i, 8] = round(synthetic_points[i, 8])
    # df = pd.DataFrame(synthetic_points, index=[0])
    # file_path = '../files/smote.csv'
    # if os.path.exists(file_path):
    #     df.to_csv(file_path, mode="a", header=False)
    # else:
    #     df.to_csv(file_path, mode="w")
    np.savetxt("../files/smote.csv", synthetic_points, delimiter=",")
