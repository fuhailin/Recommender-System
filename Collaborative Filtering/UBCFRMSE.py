# -*-coding:utf-8-*-
import math
import random
import time
from threading import Thread
from threading import Lock
#基于用户协同推荐算法
class UserBased:
    def __init__(self, userData, k):
        self.userData = userData
        self.k = k
    def simPearson(self, user1, user2):
        data = self.userData
        sim = {}
        for item in data[user1]:
            if item in data[user2]:
                sim[item] = 1
        n = len(sim)
        if not n:
            return -1
        sum1 = sum([data[user1][item] for item in sim])
        sum2 = sum([data[user2][item] for item in sim])
        sum1Sq = sum([math.pow(data[user1][item], 2) for item in sim])
        sum2Sq = sum([math.pow(data[user2][item], 2) for item in sim])
        sumMulti = sum([data[user1][item] * data[user2][item] for item in sim])
        num1 =  sumMulti - sum1 * sum2/n
        num2 = math.sqrt((sum1Sq - math.pow(sum1, 2)/n) * (sum2Sq - math.pow(sum2, 2)/n))
        if not num2:
            return -1
        return 0.5 + 0.5 * (num1 / num2)  # 将皮尔逊相似度转换至[0, 1]
    def kNeibors(self, theUserID, k):
        data = self.userData
        similarities = [(otherID, self.simPearson(theUserID, otherID)) for otherID in data if otherID != theUserID]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[0 : k]
    def estimatePref(self, theUserID, theItemID, simUsers=None):
        data = self.userData
        try:
            truePref = data[theUserID][theItemID]
        except KeyError:
            truePref = 0
        if truePref:
            return truePref
        total = 0.0
        simSum = 0.0
        simUsers = simUsers or self.kNeibors(theUserID, self.k)
        for otherID, sim in simUsers:
            if sim <= 0: continue
            try:
                otherTruePref = data[otherID][theItemID]
            except KeyError:
                continue
            total += otherTruePref * sim
            simSum += sim
        if not simSum:
            return -1 #标记出错
        return total / simSum
    def recommend(self, theUserID, howMany):
        data = self.userData
        kNeighbors = self.kNeibors(theUserID, self.k)
        ranks = []
        for otherID, in kNeighbors:
            tempRanks = [(itemID, self.estimatePref(theUserID, itemID, kNeighbors)) for itemID in data[otherID] if itemID not in data[theUserID]]
            ranks.extend(tempRanks)
        ranks.sort(key=lambda x: x[1])
        return ranks[: -(howMany+1): -1]
class Evaluator:
    def __init__(self):
        self.diSum = 0.0
        self.count = 0
        self.lock = Lock()
    def evaluate(self, data, testPercentage):
        self.data = data
        self.testPercentage = testPercentage
        startTime = time.clock()
        testPercentage = testPercentage or self.testPercentage
        trainData, testData = self.splitData(self.data, self.testPercentage)
        self.recommender = UserBased(trainData, 10)
        part1Data, part2Data, part3Data = self.splitTestDataTo3Parts(testData)
        #开3个线程计算RMSE值
        t1 = Thread(target=self.doEvaluate, args=(trainData, part1Data))
        t2 = Thread(target=self.doEvaluate, args=(trainData, part2Data))
        t3 = Thread(target=self.doEvaluate, args=(trainData, part3Data))
        t1.start()
        t2.start()
        t3.start()
        t1.join()
        t2.join()
        t3.join()
        result = math.sqrt(self.diSum / self.count)
        print '计算RMSE结束, RMSE值为: %s; 用时: %s 秒' % (result, time.clock() - startTime)
        return result
    def splitData(self, data=None, testPercentage=None):
        data = data or self.data
        testPerc = testPercentage or self.testPercentage
        trainData = {}
        testData = {}
        for user in data:
            for item, score in data[user].items():
                if random.random() < testPerc:
                    testData.setdefault(user, {})
                    testData[user][item] = score
                else:
                    trainData.setdefault(user, {})
                    trainData[user][item] = score
        return trainData, testData
    def splitTestDataTo3Parts(self, testData):
        part1Data = {}
        part2Data = {}
        part3Data = {}
        for user in testData:
            x = random.random()
            if x < 0.3:
                part1Data[user] = testData[user]
            elif x < 0.6:
                part2Data[user] = testData[user]
            else:
                part3Data[user] = testData[user]
        return part1Data, part2Data, part3Data
    def doEvaluate(self, trainData, partTestData):
        partDiSum = 0.0
        partCount = 0
        recommender = self.recommender
        k = recommender.k
        for user in partTestData:
            simUsers = recommender.kNeibors(user, k)
            for item, score in partTestData[user].items():
                predictPref = recommender.estimatePref(user, item, simUsers)
                if predictPref < 0: continue
                partDiSum += math.pow(predictPref - score, 2)
                partCount += 1
        self.lock.acquire()
        self.diSum += partDiSum
        self.count += partCount
        self.lock.release()
def loadData(filename):
    startTime = time.clock()
    totalData = {}
    count = 0
    for line in open(filename):
        userID, itemID, score,_ = line.split(',')
        user, item, score = int(userID), int(itemID), int(score)
        totalData.setdefault(user, {})
        totalData[user][item] = score
        count += 1
    print '数据加载成功! 用时: %s秒 总记录: %s 行,用户数: %s'%(time.clock()-startTime, count, len(totalData))
    return totalData
if __name__ == '__main__':
    data = loadData('u.txt')
    Evaluator().evaluate(data, 0.3)