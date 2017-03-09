#coding=utf-8
'''
基于项目的推荐系统，IBCF：
Created on 2016-10-20
@author: Administrator
'''
import random
import math

class KNN:
    def __init__(self,train = None,test = None):
        self.trainfile = train
        self.testfile = test
        self.readData()
        
    def readData(self,train = None,test = None):
        self.trainfile = train or self.trainfile
        self.testfile = test or self.testfile
        self.traindata = {}
        self.testdata = {}
        for line in open(self.trainfile):
            userid,itemid,record,_ = line.split()
            self.traindata.setdefault(userid,{})
            self.traindata[userid][itemid]=record
        for line in open(self.testfile):
            userid,itemid,record,_ = line.split()
            self.testdata.setdefault(userid,{})
            self.testdata[userid][itemid]=record
                
                
    def ItemSim(self,train = None):
        train = train or self.traindata
        ItemSimcount = dict()
        Item_count = dict()
        for _,items in train.items():
            for itemidi in items.keys():
                Item_count.setdefault(itemidi,0)
                Item_count[itemidi] += 1
                for itemidj in items.keys():
                    if itemidi == itemidj:
                        continue
                    ItemSimcount.setdefault(itemidi,{})
                    ItemSimcount[itemidi].setdefault(itemidj,0)
                    ItemSimcount[itemidi][itemidj] +=1
        self.ItemSimlist = dict()
        for itemidi, related_item in ItemSimcount.items():
            self.ItemSimlist.setdefault(itemidi,{})
            for itemidj,wij in related_item.items():
                self.ItemSimlist[itemidi].setdefault(itemidj,0)
                self.ItemSimlist[itemidi][itemidj] = wij/math.sqrt(Item_count[itemidi]*Item_count[itemidj]*1.0)

    def recommend(self,user,train = None,k = 5,nitem = 10):
        train = train or self.traindata
        recommendlist = dict()
        User_Itemlist = train.get(user,{})
        for i,ri in User_Itemlist.items():
            for j,wij in sorted(self.ItemSimlist[i].items(),key = lambda x:x[1],reverse = True)[0:k]:
                if j in User_Itemlist:
                    continue
                recommendlist.setdefault(j,0)
                recommendlist[j] += float(ri)*wij
        return dict(sorted(recommendlist.items(),key = lambda x :x[1],reverse = True)[0:nitem])
    
    def recallAndPrecision(self,train = None,test = None,k = 5,nitem = 10):
        train  = train or self.traindata
        test = test or self.testdata
        hit = 0
        recall = 0
        precision = 0
        for user in train.keys():
            tu = test.get(user,{})
            rank = self.recommend(user, train = train,k = k,nitem = nitem) 
            for item,_ in rank.items():
                if item in tu:
                    hit += 1
            recall += len(tu)
            precision += nitem
        return (hit / (recall * 1.0),hit / (precision * 1.0))
    
    def coverage(self,train = None,test = None,k = 5,nitem = 10):
        train = train or self.traindata
        test = test or self.testdata
        recommend_items = set()
        all_items  = set()
        for user in train.keys():
            for item in train[user].keys():
                all_items.add(item)
            rank = self.recommend(user, train, k = k, nitem = nitem)
            for item,_ in rank.items():
                recommend_items.add(item)
        return len(recommend_items) / (len(all_items) * 1.0)
    
    def popularity(self,train = None,test = None,k = 5,nitem = 10):
        train = train or self.traindata
        test = test or self.testdata
        item_popularity = dict()
        for user ,items in train.items():
            for item in items.keys():
                item_popularity.setdefault(item,0)
                item_popularity[item] += 1
        ret = 0
        n = 0
        for user in train.keys():
            rank = self.recommend(user, train, k = k, nitem = nitem)
            for item ,_ in rank.items():
                if item in item_popularity:
                    ret += math.log(1+item_popularity[item])
                    n += 1
        return ret / (n * 1.0)

    def RMSE(self, records):
        return math.sqrt( \
            sum([(rui - pui) * (rui - pui) for u, i, rui, pui in records]) \
            / float(len(records)))

    def MAE(self, records):
        return sum([abs(rui - pui) for u, i, rui, pui in records]) \
               / float(len(records))


def testKNNCF():
    train = 'u1.base'
    test = 'u1.test'
    cf  =  KNN(train,test)
    cf.ItemSim()
    print("%3s%20s%20s%20s%20s" % ('K',"precision",'recall','coverage','popularity'))
    #for k in [5,10,20,40,80,160]:
    for k in [5, 10, 20, 40]:
        recall,precision = cf.recallAndPrecision( k = k)
        coverage = cf.coverage(k = k)
        popularity = cf.popularity(k = k)
        RMSE=cf.RMSE()
        MAE=cf.MAE()
        print("%3d%19.3f%%%19.3f%%%19.3f%%%20.3f" % (k,precision * 100,recall * 100,coverage * 100,popularity))
        
if __name__ == "__main__":
    testKNNCF()