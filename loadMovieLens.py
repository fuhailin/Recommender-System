#-*- coding: utf-8 -*-
#---------------------------------------------------
# Name:        loadMovielens.py
# Purpose:  读取测试集和训练集，以字典格式存储到内存中
#  
# Author:      Python大菜鸟
# E-mail:      zhenboye@gmail.com  
# Created:     14-06-2014  
#---------------------------------------------------
import sys
import os

##==================================
#         加载指定的训练集文件
#  参数fileName 代表某个训练集文件
##==================================
def loadMovieLensTrain(fileName='u1.base'):
    #str1 = './ml-100k/'                         # 目录的相对地址
    
    prefer = {}
    for line in open(fileName,'r'):       # 打开指定文件
        (userid, movieid, rating,ts) = line.split('\t')     # 数据集中每行有4项
        prefer.setdefault(userid, {})      # 设置字典的默认格式,元素是user:{}字典
        prefer[userid][movieid] = float(rating)    

    return prefer      # 格式如{'user1':{itemid:rating, itemid2:rating, ,,}, {,,,}}


##==================================
#        加载对应的测试集文件
#  参数fileName 代表某个测试集文件,如u1.test
##==================================
def loadMovieLensTest(fileName='u1.test'):
    #str1 = './ml-100k/'
    prefer = {}
    for line in open(fileName,'r'):
        (userid, movieid, rating,ts) = line.split('\t')   #数据集中每行有4项
        prefer.setdefault(userid, {})    
        prefer[userid][movieid] = float(rating)   
    return prefer                   


if __name__ == "__main__":
    print ("""这个部分可以进行上面2个函数测试 """)
    
    trainDict = loadMovieLensTrain()
    testDict = loadMovieLensTest()

    print (len(trainDict))
    print (len(testDict))
    print (""" 测试通过 """)
                        

















