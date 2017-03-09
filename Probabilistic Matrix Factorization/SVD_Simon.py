# This Python file uses the following encoding: utf-8

#################################################
# SVD_Simon
# Author : Liuqinyuan
# Date   : 2015-07-15
# HomePage :
# Email  :
#################################################
from numpy import *
import random
import time
import math

start_time = time.time()


# Step1:Split Data
# Step2：Load Data
def loadData():
    datafile = 'data/ml-100k/u.data'
    print('Step 1:reading data...')
    data = list()
    for line in open(datafile):
        lineArr = line.strip().split('\t')
        user_id = int(lineArr[0])
        movie_id = int(lineArr[1])
        rating = float(lineArr[2])
        data.append((user_id, movie_id, float(rating)))
    print('Reading data successed')
    train = dict()
    test = dict()
    random.seed(1)
    print('Step 2:splitting data...')
    train_count = 0
    test_count = 0
    for user_id, movie_id, rating in data:
        i = random.randint(0, 99)
        if (i % 8 != 1):
            train_count += 1
            train.setdefault(user_id, dict())
            train[user_id][movie_id] = rating
        else:
            test_count += 1
            test.setdefault(user_id, dict())
            test[user_id][movie_id] = rating

    print("Splitting data successed")
    return train, test, train_count, test_count


train_data, test_data, train_count, test_count = loadData()
# Step 3:Initialization SVD_Model :: 初始化系数矩阵和一些参数统计


movie_features = [[0.1 for x in range(10)] for y in range(129651)]
user_features = [[0.1 for x in range(10)] for y in range(706)]
# movies = [[0 for x in range(4)] for y in range(129651)]     #第一列代表RatingCount 第二列代表RatingSum 第三列代表每个电影的平均数 第四列代表每部电影的加权平均数
# users = [[0 for x in range(3)] for y in range(706)]     #第一列代表RatingCount 第二列代表RatingSum
rating_cache = [[0 for x in range(129651)] for y in range(706)]
print('Step3:training...')
"""
def cal_original_Data(train_data):
    for user_id,items in train_data.items():
        for movie_id,rating in items.items():
            movies[int(movie_id) - 1][0] += 1      #统计电影评分个数
            movies[int(movie_id) - 1][1] += rating #统计评分总和       
            users[int(user_id) - 1][0] += 1        #用户数据
            users[int(user_id) - 1][1] += rating
        
    for i in range(4):
        for j in range(129651):
            if(movies[j][0] != 0):
                movies[j][2] = movies[j][1] / (1.0 * movies[j][0])
                movies[j][3] = (3.23 * 25 + movies[j][1]) / (25.0 + movies[j][0])
    print('cal_original_Data finished')  
    
cal_original_Data(train_data) 
"""


def PredictRating(movieid, userid, feature, rating_cache, bTrailing):
    if (rating_cache > 0):
        p_rating = rating_cache
    else:
        p_rating = 1

    p_rating += movie_features[movieid][feature] * user_features[userid][feature]
    if (p_rating > 5):
        p_rating = 5
    if (p_rating < 1):
        p_rating = 1
    """
    if(bTrailing):
        #p_rating += (10 - feature - 1) * (0.1 * 0.1);
        if (p_rating > 5):
            p_rating = 5
        if(p_rating < 1): 
            p_rating = 1
    """
    return p_rating


def PredictRating1(movieid, userid):
    p_rating = 1
    for i in range(10):
        p_rating += movie_features[movieid][i] * user_features[userid][i]
        if (p_rating > 5):
            p_rating = 5
        if (p_rating < 1):
            p_rating = 1
    return p_rating


def cal_error(test_data, test_count):
    sq = 0
    print('Step 4:Testing...')
    for user_id, items in test_data.items():
        for movie_id, rating in items.items():
            p = PredictRating1(movie_id - 1, user_id - 1)

            err = (1.0 * test_data[user_id][movie_id] - p)
            sq += err * err

    rmse = math.sqrt(sq / test_count)
    return rmse


def cal_matric_Feature(train_data, train_count):  # 用来训练特征矩阵中的每一项参数的值
    print("开始训练参数：...")
    fileIn = open('temp.txt', 'w')
    learn_rate = [0.001, 0.001, 0.01]
    lamda = [0.002, 0.02, 0.2]
    epoch = [100, 150, 200]
    rmse = 0
    for lrate in learn_rate:
        for lam in lamda:
            for epo in epoch:
                for i in range(10):
                    # print'训练第 %d 个参数:'%i
                    rmse = 0
                    for j in range(epo):
                        sq = 0
                        for user_id, items in train_data.items():
                            for movie_id, rating in items.items():
                                cache = rating_cache[user_id - 1][movie_id - 1]
                                p = PredictRating(int(movie_id) - 1, int(user_id) - 1, int(i), float(cache), 1)
                                # print(p)
                                err = (1.0 * train_data[user_id][movie_id] - p)
                                sq += err * err
                                u_f = user_features[user_id - 1][i]
                                m_f = movie_features[movie_id - 1][i]
                                user_features[user_id - 1][i] += float(lrate * (err * m_f - lam * u_f))
                                movie_features[movie_id - 1][i] += float(lrate * (err * u_f - lam * m_f))

                    # print(rmse)
                    for user_id, items in train_data.items():
                        for movie_id, rating in items.items():
                            rating_cache[user_id - 1][movie_id - 1] = float(
                                PredictRating(movie_id - 1, user_id - 1, i, rating_cache[user_id - 1][movie_id - 1], 0))
                rmse = math.sqrt(float(sq) / train_count)
                test_rmse = cal_error(test_data, test_count)
                fileIn.write(str(lrate))
                fileIn.write(',')
                fileIn.write(str(lam))
                fileIn.write(',')
                fileIn.write(str(epo))
                fileIn.write(',')
                fileIn.write(str(rmse))
                fileIn.write(',')
                fileIn.write(str(test_rmse))
                fileIn.write('\n')
    fileIn.close()
    print('训练结束')


cal_matric_Feature(train_data, train_count)

# rmse = cal_error(test_data,test_count)
# print('RMSE in test_data is %f' % rmse)
print('The process is over')
print('总计耗时：%fs' % float(time.time() - start_time))


# Step4:测试数据
