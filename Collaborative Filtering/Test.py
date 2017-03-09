# -*- coding: utf-8 -*-
"""
load data set

"""
import random
from numpy import *


def load_rating_data(file_path='ml-100k/u.data'):
    """
    load movie lens 1M ratings from original rating file.
    need to download and put rating data in /data folder first.
    Source: http://www.grouplens.org/
    """
    prefer = []
    for line in open(file_path, 'r'):  # 打开指定文件
        (userid, movieid, rating, ts) = line.split('\t')  # 数据集中每行有4项
        uid = int(userid)
        mid = int(movieid)
        rat = float(rating)
        prefer.append([uid, mid, rat])
    return array(prefer)


def spilt_rating_dat(data, size=0.2):
    train_data = None
    test_data = None
    for line in data:
        rand = random.random()
        if (rand < size):
            test_data.append(line)
        else:
            train_data.append(line)
    return train_data, test_data


temp = load_rating_data()
train, test = spilt_rating_dat(temp)
x=array(train)
y=array(test)
print(x[:,0])
print(y[:,0])
print(x.shape[0])
print(y.shape[0])
print(mean(x[:,2]))