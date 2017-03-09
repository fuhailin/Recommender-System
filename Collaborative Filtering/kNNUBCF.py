#-*- coding: utf-8 -*-
import random
import math
from math import sqrt

class UserBasedCF:
    def __init__(self, train=None, test=None):
        self.trainfile = train
        self.testfile = test
        self.readData()

    def readData(self, train=None, test=None):
        self.trainfile = train or self.trainfile
        self.testfile = test or self.testfile
        self.traindata = {}
        self.testdata = {}
        for line in open(self.trainfile):
            userid, itemid, record, _ = line.split()
            self.traindata.setdefault(userid, {})
            self.traindata[userid][itemid] = record
        for line in open(self.testfile):
            userid, itemid, record, _ = line.split()
            self.testdata.setdefault(userid, {})
            self.testdata[userid][itemid] = record

    
    def userSimilarityBest(self, train=None):
        train = train or self.traindata
        self.userSimBest = dict()
        item_users = dict()
        for u, item in train.items():
            for i in item.keys():
                item_users.setdefault(i, set())
                item_users[i].add(u)
        user_item_count = dict()
        count = dict()
        for item, users in item_users.items():
            for u in users:
                user_item_count.setdefault(u, 0)
                user_item_count[u] += 1
                for v in users:
                    if u == v: continue
                    count.setdefault(u, {})
                    count[u].setdefault(v, 0)
                    count[u][v] += 1
        for u, related_users in count.items():
            self.userSimBest.setdefault(u, dict())
            for v, cuv in related_users.items():
                self.userSimBest[u][v] = cuv / math.sqrt(user_item_count[u] * user_item_count[v] * 1.0)

    def recommend(self, user, train=None, k=8, nitem=40):
        train = train or self.traindata
        rank = dict()
        interacted_items = train.get(user, {})
        for v, wuv in sorted(self.userSimBest[user].items(), key=lambda x: x[1], reverse=True)[0:k]:  # 获取与user相似度最高的k个用户
            for i, rvi in train[v].items():
                if i in interacted_items:
                    continue  # 只选择user没有评分过的物品进行推荐
                rank.setdefault(i, 0)  # 设置初始值，以便做下面的累加运算
                rank[i] += wuv  # 书中为rank[i] +=rvi*wuv
        myresult=dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:nitem])  # 用sorted方法对推荐的物品进行排序，预计评分高的排在前面，再取其中nitem个，nitem为每个用户推荐的物品数量
        return myresult

    def recallAndPrecision(self, train=None, test=None, k=8, nitem=10):
        train = train or self.traindata
        test = test or self.testdata
        hit = 0
        recall = 0
        precision = 0
        for user in train.keys():
            tu = test.get(user, {})  # 如果测试集中没有这个用户，则将tu初始化为空，避免test[user]报错
            rank = self.recommend(user, train=train, k=k, nitem=nitem)
            for item, _ in rank.items():
                if item in tu:
                    hit += 1
            recall += len(tu)
            precision += nitem

        return (hit / (recall * 1.0), hit / (precision * 1.0))

    def coverage(self, train=None, test=None, k=8, nitem=10):
        train = train or self.traindata
        test = test or self.testdata
        recommend_items = set()
        all_items = set()
        for user in train.keys():
            for item in train[user].keys():
                all_items.add(item)
            rank = self.recommend(user, train, k=k, nitem=nitem)
            for item, _ in rank.items():
                recommend_items.add(item)
        return len(recommend_items) / (len(all_items) * 1.0)

    def popularity(self, train=None, test=None, k=8, nitem=10):
        train = train or self.traindata
        test = test or self.testdata
        item_popularity = dict()
        for user, items in train.items():
            for item in items.keys():
                item_popularity.setdefault(item, 0)
                item_popularity[item] += 1
        ret = 0
        n = 0
        for user in train.keys():
            rank = self.recommend(user, train, k=k, nitem=nitem)
            for item, _ in rank.items():
                ret += math.log(1 + item_popularity[item])
                n += 1
        return ret / (n * 1.0)


def testUserBasedCF():
    train = 'ml-100k/u1.base'
    test = 'ml-100k/u1.test'
    cf = UserBasedCF(train, test)
    cf.userSimilarityBest()
    print("%3s%20s%20s%20s%20s" % ('K', "precision", 'recall', 'coverage', 'popularity'))
    for k in [5, 10, 20, 40, 80, 160]:
    #for k in [5, 10]:
        recall, precision = cf.recallAndPrecision(k=k)
        coverage = cf.coverage(k=k)
        popularity = cf.popularity(k=k)
        print("%3d%19.3f%%%19.3f%%%19.3f%%%20.3f" % (k, precision * 100, recall * 100, coverage * 100, popularity))
        #print("%3d%19.3f%%%19.3f" % (k, precision * 100, recall * 100))


if __name__ == "__main__":
    testUserBasedCF()