# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Name:            UserBasedCF.py
# Purpose:  基于已知的训练集，"测试集"中的user的item进行评分预测.
# Data:        MovieLens
# Language:    Python 2.7
# Author:      Hailin
# E-mail:      hailinfufu@outlook.com
# Created:     1-11-2016
# --------------------------------------------------------
import math
from math import sqrt
from loadMovieLens import loadMovieLensTrain
from loadMovieLens import loadMovieLensTest
import numpy as np

class UserBasedCF:
    def __init__(self, train=None, test=None):
        self.trainfile = train
        self.testfile = test
        self.readData()

    def readData(self, train=None, test=None):
        self.trainfile = train or self.trainfile
        self.testfile = test or self.testfile
        self.traindata = loadMovieLensTrain(train)  # 加载训练集
        self.testdata = loadMovieLensTest(test)  # 加载测试集

### 计算pearson相关度
def sim_pearson(prefer, person1, person2):
    sim = {}
    # 查找双方都评价过的项
    for item in prefer[person1]:
        if item in prefer[person2]:
            sim[item] = 1  # 将相同项添加到字典sim中
    # 元素个数
    n = len(sim)
    if len(sim) == 0:
        return -1

    # 所有偏好之和
    sum1 = sum([prefer[person1][item] for item in sim])
    sum2 = sum([prefer[person2][item] for item in sim])

    # 求平方和
    sum1Sq = sum([pow(prefer[person1][item], 2) for item in sim])
    sum2Sq = sum([pow(prefer[person2][item], 2) for item in sim])

    # 求乘积之和 ∑XiYi
    sumMulti = sum([prefer[person1][item] * prefer[person2][item] for item in sim])

    num1 = sumMulti - (sum1 * sum2 / n)
    num2 = sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2) / n))
    if num2 == 0:  ### 如果分母为0，本处将返回0.
        return 0

    result = num1 / num2
    return result

### 获取对item评分的K个最相似用户（K默认20）
def topKMatches(prefer, person, itemId, k=20, sim=sim_pearson):
    userSet = []
    scores = []
    users = []
    # 找出所有prefer中评价过Item的用户,存入userSet
    for user in prefer:
        if itemId in prefer[user]:
            userSet.append(user)
    # 计算相似性
    scores = [(sim(prefer, person, other), other) for other in userSet if other != person]

    # 按相似度排序
    scores.sort()
    scores.reverse()

    if len(scores) <= k:  # 如果小于k，只选择这些做推荐。
        for item in scores:
            users.append(item[1])  # 提取每项的userId
        return users
    else:  # 如果>k,截取k个用户
        kscore = scores[0:k]
        for item in kscore:
            users.append(item[1])  # 提取每项的userId
        return users  # 返回K个最相似用户的ID

### 计算用户的平均评分
def getAverage(prefer, userId):
    count = 0
    sum = 0
    for item in prefer[userId]:
        sum = sum + prefer[userId][item]
        count = count + 1
    return sum / count

### 平均加权策略，预测userId对itemId的评分
def getRating(prefer1, userId, itemId, knumber=20, similarity=sim_pearson):
    sim = 0.0
    averageOther = 0.0
    jiaquanAverage = 0.0
    simSums = 0.0
    # 获取K近邻用户(评过分的用户集)
    users = topKMatches(prefer1, userId, itemId, k=knumber, sim=sim_pearson)

    # 获取userId 的平均值
    averageOfUser = getAverage(prefer1, userId)

    # 计算每个用户的加权，预测
    for other in users:
        sim = similarity(prefer1, userId, other)  # 计算比较其他用户的相似度
        averageOther = getAverage(prefer1, other)  # 该用户的平均分
        # 累加
        simSums += abs(sim)  # 取绝对值
        jiaquanAverage += (prefer1[other][itemId] - averageOther) * sim  # 累加，一些值为负

    # simSums为0，即该项目尚未被其他用户评分，这里的处理方法：返回用户平均分
    if simSums == 0:
        return averageOfUser
    else:
        return (averageOfUser + jiaquanAverage / simSums)


        ##==================================================================

### 计算RMSE评分预测
def getRMSE(records):
    return math.sqrt(sum([(rui-pui)*(rui-pui) for u,i,rui,pui in records]))/float(len(records))

### 计算MAE评分预测
def getMAE(records):
    return sum([abs(rui-pui) for u,i,rui,pui in records])/float(len(records))

##     getAllUserRating(): 获取所有用户的预测评分，存放到fileResult中
##
## 参数:fileTrain,fileTest 是训练文件和对应的测试文件，fileResult为结果文件
##     similarity是相似度度量方法，默认是皮尔森。
##==================================================================
def getAllUserRating(fileTrain='u1.base', fileTest='u1.test',k=20, similarity=sim_pearson):
    traindata = loadMovieLensTrain(fileTrain)  # 加载训练集
    testdata = loadMovieLensTest(fileTest)  # 加载测试集
    inAllnum = 0
    records=[]
    for userid in testdata:  # test集中每个用户
        for item in testdata[userid]:  # 对于test集合中每一个项目用base数据集,CF预测评分
            rating = getRating(traindata, userid, item, k)  # 基于训练集预测用户评分(用户数目<=K)
            records.append([userid,item,testdata[userid][item],rating])
            inAllnum = inAllnum + 1
    #np.savetxt("records.txt",records,fmt='%1.4e')
    print("-------------Completed!!-----------", inAllnum)
    return records


############    主程序   ##############
if __name__ == "__main__":
    print("\n--------------基于MovieLens的推荐系统 运行中... -----------\n")
    trainfile='u1.base'
    testfile='u1.test'
    print("%3s%20s%20s" % ('K', "RMSE","MAE"))
    for k in [10, 25, 50, 75, 100, 125, 150]:
        r=getAllUserRating(trainfile, testfile, k)
        rmse=getRMSE(r)
        mae=getMAE(r)
        print("%3d%19.3f%%%19.3f" % (k, rmse * 100, mae * 100))
