# coding=utf-8
import numpy as np
import time
import math

__author__ = 'axuanwu'


# 2015年 9 月 25 日
class XMatrix():
    def __init__(self, m=1000, n=100, step=4):
        self.zero_like = 0.01  # 伪零： 差异小于该值认为无差异
        self.step = int(step)  # 数据块的切分方法
        self.m = m  # 原矩阵行数
        self.n = n  # 原矩阵列数
        self.splits = int(math.ceil(1.0 * self.n / self.step))  # 每行记录的分块数
        self.res = self.n % self.step
        self.Memory = []  # 存储数据块的实体
        self.dict_rowid = {}  # 记录数据块的位置字典
        self.Memory_max_rowid = -1
        self.UUU = np.random.random((self.m, 2))
        self.VVV = np.ones((2, 1))  # 本来用做归一化UUU 和 MMM后的 系数矩阵，目前未处理
        self.MMM = np.random.random((2, self.n))

    def set_data(self, tezhenshu):
        self.tezhenshu = tezhenshu
        self.UUU = np.random.random((self.m, tezhenshu))
        self.VVV = np.zeros((tezhenshu, tezhenshu))
        for i in xrange(0, tezhenshu):
            self.VVV[i, i] = 1
        self.MMM = np.random.random((tezhenshu, self.n))

    def intoMatrix(self, i, j, data):
        # 矩阵赋值
        row_id = (int(j / self.step) + i * self.splits)
        i_temp = self.dict_rowid.get(row_id, -1)
        if i_temp == -1:
            self.Memory_max_rowid += 1
            i_temp = self.Memory_max_rowid
            self.dict_rowid[row_id] = i_temp
            self.Memory.append(np.array([0.0] * self.step))  # 增加一块 数据块
        self.Memory[i_temp][j % self.step] = data

    def getitem(self, i, j=-1):
        # 读取稀疏矩阵
        if j == -1:
            # 返回一数据块
            i_temp = self.dict_rowid.get(i, -1)
            if i_temp == -1:
                temp = np.array([0.0] * self.step)
            else:
                temp = self.Memory[i_temp]
            if self.splits == (i % self.splits) + 1:
                return temp[0:self.res]
            else:
                return temp
        else:
            # 返回元素
            i_temp = self.dict_rowid.get((int(j / self.step) + i * self.splits), -1)
            if i_temp == -1:
                return 0
            return self.Memory[i_temp][j % self.step]

    def error_sum(self, k=0):
        error_sum = 0
        if k != 0:  # 计算合成矩阵 与 与矩阵 的 最大差异
            for i in xrange(0, self.m):
                for j in xrange(0, self.splits):
                    start = j * self.step
                    end = min(start + self.step, self.n)
                    temp = np.dot(self.UUU[i, :], self.MMM[:, start:end])
                    i_temp = self.dict_rowid.get((j + i * self.splits), -1)
                    if i_temp == -1:
                        error_sum = max(error_sum, max(np.abs(temp)))
                    else:
                        error_sum = max(error_sum, max(np.abs(temp - self.Memory[i_temp][0:(end - start)])))
            return error_sum
        else:  # 计算合成矩阵 与 与矩阵 所有元素差异的平方和
            for i in xrange(0, self.m):
                for j in xrange(0, self.splits):
                    start = j * self.step
                    end = min(start + self.step, self.n)
                    temp = np.dot(self.UUU[i, :], self.MMM[:, start:end])
                    i_temp = self.dict_rowid.get((j + i * self.splits), -1)
                    if i_temp == -1:
                        error_sum += sum(np.power(temp, 2))
                    else:
                        error_sum += sum(np.power(temp - self.Memory[i_temp][0:(end - start)], 2))
                        # print (np.power(self.test-np.dot(self.UUU[:, :], self.MMM[:, :]),2)).sum(0).sum(0)-error_sum
            return error_sum

    def SVD(self):
        step = 1
        iter = 0
        t_start = time.time()
        pre_error = self.error_sum()
        print("-- ", iter, " --", time.time() - t_start, pre_error, self.error_sum(1))
        while (1):
            du = np.ones((self.m, self.tezhenshu))
            dm = np.zeros((self.tezhenshu, self.n))
            # UUU 阵
            for i in range(0, self.m):
                t = np.zeros((self.tezhenshu, self.n))
                # t0 = np.zeros((self.tezhenshu, self.n))
                for j in range(0, self.splits):  # 按块遍历
                    js = j * self.step
                    je = min(js + self.step, self.n)
                    t[:, js:je] = (self.getitem(i * self.splits + j) - np.dot(self.UUU[i, :],
                                                                              self.MMM[:, js:je])) * self.MMM[:, js:je]
                du[i, :] = t.sum(1)  # - ku * self.UUU[i,:]
            # MMM 阵
            for j in range(0, self.splits):  # 按块遍历
                js = j * self.step
                je = min(js + self.step, self.n)
                t = np.zeros((self.tezhenshu, je - js))
                for i in range(0, self.n):
                    t[:, 0: je - js] += np.mat(self.UUU[i, :]).T * np.mat(
                        self.getitem(i * self.splits + j) - np.dot(self.UUU[i, :], self.MMM[:, js:je]))
                dm[:, js:je] = t  # - ku * self.MMM[:,js:je]
            u = step / max((np.abs(dm)).max(), (np.abs(du)).max())
            self.MMM += u * dm
            self.UUU += u * du
            iter += 1
            error_sum = self.error_sum()
            if error_sum > pre_error:  # 到达当前极值
                step *= 0.3  # 步长减少
                print(step, u, iter, " --", time.time() - t_start, error_sum, self.error_sum(1))
            pre_error = error_sum
            # print "-- ", iter, " --", time.localtime(), error_sum,self.error_sum(1)
            if (self.error_sum(1) < self.zero_like) | (step < 10 ** (-8)):
                print("-- ", iter, " --", time.time() - t_start, error_sum, self.error_sum(1))
                break


if __name__ == "__main__":
    m = 10  # 需要分解的矩阵行数
    n = 8  # 需要分解的矩阵列数
    a = np.zeros((m, n))
    aaa = XMatrix(m, n, 6)
    aaa.set_data(2)  # 3 个主要成分
    for i in range(0, m):
        for j in range(0, n):
            a[i, j] = i + j % 4 + (i % 5) * (5 - j % 4)
            if a[i, j] == 0:
                continue
            aaa.intoMatrix(i, j, a[i, j])  # 将需要分解的矩阵的元素放入对象 aaa 中
    aaa.SVD()
    print
    aaa.error_sum(1)