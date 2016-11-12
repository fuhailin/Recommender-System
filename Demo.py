# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def train_test_split(fileName,type=1):
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    if(type==1):
        df = pd.read_csv(fileName, sep='\t', names=header)
    else:
        df = pd.read_csv(fileName, sep='::', names=header,engine = 'python')
    n_users = df.user_id.unique().shape[0]
    users = df.user_id.max()
    n_items = df.item_id.unique().shape[0]
    items = df.item_id.max()

    print 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items)
    print 'The biggest ID of users = ' + str(users) + ' | The biggest ID of movies = ' + str(items)
    #
    train_data, test_data = cv.train_test_split(df, test_size=0.1)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)
    #Create two user-item matrices, one for training and another for testing
    train_data_matrix = np.zeros((users, items))
    for line in train_data.itertuples():
        train_data_matrix[line[1]-1, line[2]-1] = line[3]

    test_data_matrix = np.zeros((users, items))
    for line in test_data.itertuples():
        test_data_matrix[line[1]-1, line[2]-1] = line[3]
    return train_data_matrix,test_data_matrix
def fast_similarity(ratings, kind='user', epsilon=1e-9):
    # epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)
def get_rmse_mae(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, actual)),mean_absolute_error(pred, actual)

def predict_topk_nobias(ratings, similarity, kind='user', k=40):
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        user_bias = ratings.mean(axis=1)
        ratings = (ratings - user_bias[:, np.newaxis]).copy()
        for i in xrange(ratings.shape[0]):
            top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]
            for j in xrange(ratings.shape[1]):
                pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users]) 
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
        pred += user_bias[:, np.newaxis]
    if kind == 'item':
        item_bias = ratings.mean(axis=0)
        ratings = (ratings - item_bias[np.newaxis, :]).copy()
        for j in xrange(ratings.shape[1]):
            top_k_items = [np.argsort(similarity[:,j])[:-k-1:-1]]
            for i in xrange(ratings.shape[0]):
                pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T) 
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items])) 
        pred += item_bias[np.newaxis, :]
        
    return pred


if __name__ == "__main__":
    print ("Please choose which dataset you want to use\n1:ml-100k\n2:ml-1m")
    n=input("Your choose:")
    if(n==1):
        train, test = train_test_split('ml-100k/u.data',1)
    else:
        train, test = train_test_split('ml-1m/ratings.dat',2)
    user_similarity = fast_similarity(train, kind='user')
    item_similarity = fast_similarity(train, kind='item')
    print item_similarity[:4, :4]

    print("%3s%20s%20s" % ('K', "RMSE", 'MAE'))
    for k in [25, 50, 75, 100, 125, 150]:
        user_pred = predict_topk_nobias(train, user_similarity, kind='user', k=k)
        item_pred = predict_topk_nobias(train, item_similarity, kind='item', k=k)
        
        user_test_rmse,user_test_mae = get_rmse_mae(user_pred, test)
        item_test_rmse,item_test_mae = get_rmse_mae(item_pred, test)
        
        print("%3d%19.3f%19.3f%20s" % (k, user_test_rmse, user_test_mae,"user_test"))
        print("%3d%19.3f%19.3f%20s" % (k, item_test_rmse, item_test_mae,"item_test"))
