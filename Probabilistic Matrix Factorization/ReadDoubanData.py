# -*- coding: utf-8 -*-
import numpy as np
import math
import pickle
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity


def get_keys(d, value):
    for k, v in d.items():
        if v == value:
            return k


def TfIdf(Tf, M_Tags, N):
    idf = math.log((M_Tags / (N + 1)), math.e)
    return Tf * idf


def Get_Feature_num(corpus):
    word = {}
    for d in corpus:
        if word.get(d, 0) == 0:
            word[d] = 1
        else:
            word[d] += 1
    return word

def cosin_distance(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return dot_product / ((normA * normB) ** 0.5)


# 给定用户实例编号，和相似度矩阵，得到最相似的K个用户
def get_K_Neighbors(userinstance, SimNArray, k=10):
    rank = dict()
    c, d = np.shape(SimNArray)
    for i in range(0, c):
        if userinstance == i: continue
        rank.setdefault(i, 0)  # 设置初始值，以便做下面的累加运算
        rank[i] += SimNArray[userinstance][i]
    myresult = dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[
                    0:k])  # 用sorted方法对推荐的物品进行排序，预计评分高的排在前面，再取其中nitem个，nitem为每个用户推荐的物品数量
    return myresult


if __name__ == '__main__':
    pkl_file1 = open('data/M1/UserTags.pkl', 'rb')
    usertags = dict(pickle.load(pkl_file1))
    pkl_file1.close()

    pkl_file2 = open('data/M1/ItemTags.pkl', 'rb')
    itemtags = pickle.load(pkl_file2)
    pkl_file2.close()

    pkl_file3 = open('data/M1/UserIndex.pkl', 'rb')
    userIndex = pickle.load(pkl_file3)
    pkl_file3.close()

    pkl_file4 = open('data/M1/ItemIndex.pkl', 'rb')
    itemIndex = pickle.load(pkl_file4)
    pkl_file4.close()
    pkl_file5 = open('data/M1/TagIndex.pkl', 'rb')
    tagIndex = pickle.load(pkl_file5)
    pkl_file5.close()

    pkl_file6 = open('data/M1/TagCountUser.pkl', 'rb')
    TagCountUser = pickle.load(pkl_file6)
    pkl_file6.close()
    pkl_file7 = open('data/M1/TagCountItem.pkl', 'rb')
    TagCountItem = pickle.load(pkl_file7)
    pkl_file7.close()

    n_user = len(userIndex)  # 用户总数
    n_tag = len(tagIndex)  # 标签总数
    n_item = len(itemIndex)  # 电影总数
    user_tag_matric = np.zeros((n_user, n_tag))
    item_tag_matric = np.zeros((n_item, n_tag))
    print(n_user)
    print(n_item)
    print(n_tag)
    print(np.shape(user_tag_matric))
    print(np.shape(item_tag_matric))
    for userid in usertags:
        corpus = usertags.get(userid)
        word = Get_Feature_num(corpus)
        for tag, count in word.items():
            Tf = count / len(corpus)
            tfidf = TfIdf(Tf, len(usertags), TagCountUser[tag])
            a = userIndex[userid]  # get_keys(userIndex, userid)
            b = tagIndex[tag]  # get_keys(tagIndex, tag)
            user_tag_matric[a][b] = tfidf
        print("user_tag_matric:"+str(userid))

    user_similarity = np.zeros((n_user, n_user))
    item_similarity = np.zeros((n_item, n_item))
    distance0 = cosin_distance(user_tag_matric[0], user_tag_matric[1])
    distance = cosin_distance(user_tag_matric[1], user_tag_matric[2])
    user_similarity=cosine_similarity(user_tag_matric)
    # user_similarity = pairwise_distances(user_tag_matric, metric='cosine')
    test2 = get_K_Neighbors(1, user_similarity, 10)
    print(test2)
    for movieid in itemtags:
        corpusm = itemtags[movieid]
        wordm = Get_Feature_num(corpusm)
        for tag, count in wordm.items():
            tf = count / len(corpusm)
            tfidfm = TfIdf(tf, len(itemtags), TagCountItem[tag])
            c = itemIndex[movieid]  # get_keys(itemIndex, movieid)
            d = tagIndex[tag]  # get_keys(tagIndex, tag)
            item_tag_matric[c][d] = tfidfm
        print("item_tag_matric:"+str(movieid))
    item_similarity=cosine_similarity(item_tag_matric)
    # item_similarity = pairwise_distances(item_tag_matric.T, metric='cosine')
    test3 = get_K_Neighbors(1, item_similarity, 10)
    print(test3)
    print(0)
