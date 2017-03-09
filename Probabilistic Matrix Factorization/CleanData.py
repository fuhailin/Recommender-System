# -*- coding: utf-8 -*-
# Remove those tags which are annotated by less than five distinct users and five distinct items
# @author: Kris
import numpy as np
import pickle


def Get_tag_num(tagset, usertag):
    tagcount = dict()
    for tag in tagset:
        tagcount.setdefault(tag, 0)
        for item in usertag.values():
            if tag in item:
                tagcount[tag] += 1
    return tagcount


def CleanData(filename='data/M1/M1.txt'):
    UserTags = {}  # 用户标签
    ItemTags = {}  # 电影标签
    TagUsers = {}  # 标签下标索引
    TagItems = {}
    TagSet = set()  # 集合set()不可添加重复数据
    User_Tags = dict()
    Item_Tags = dict()
    TagCountUser = dict()
    TagCountItem = dict()
    z = 0
    i = 0
    j = 0
    k = 0
    ItemIndex = {}  # 电影下标索引
    TagIndex = {}  # 标签下标索引
    UserIndex = {}  # 用户下标索引
    for line in open(filename, 'r', encoding='UTF-8'):
        z += 1
        if len(line.rstrip('\n')) == 0:
            continue
        linelist = line.split(',')
        UserID = int(linelist[0])
        MovieID = int(linelist[1])
        Rating = float(linelist[2])
        tags = str(linelist[4].rstrip('\n')).lower()
        if len(tags) > 0:
            tagList = tags.split(',')
            for i in range(0, len(tagList)):
                tag = tagList[i].replace(' ', '')
                if tag not in TagSet:
                    TagSet.add(tag)
                if tag in TagUsers:
                    TagUsers[tag].add(UserID)
                else:
                    TagUsers.setdefault(tag, set())
                    TagUsers[tag].add(UserID)
                if tag in TagItems:
                    TagItems[tag].add(MovieID)
                else:
                    TagItems.setdefault(tag, set())
                    TagItems[tag].add(MovieID)
                if MovieID in ItemTags:
                    ItemTags[MovieID].append(tag)
                else:
                    ItemTags.setdefault(MovieID, list())
                    ItemTags[MovieID].append(tag)
                if UserID in UserTags:
                    UserTags[UserID].append(tag)
                else:
                    UserTags.setdefault(UserID, list())
                    UserTags[UserID].append(tag)
        print(z)
        # if z>30000: break
    # ****************  Remove those tags which are annotated by less than five distinct users and five distinct items.  ***********#
    for t1 in TagUsers:
        a = TagUsers.get(t1)
        b = TagItems.get(t1)
        if len(a) < 5 and len(b) < 5:
            TagSet.remove(t1)
            for u1 in TagUsers.get(t1):
                while t1 in UserTags[u1]:
                    UserTags.get(u1).remove(t1)
            for i1 in TagItems.get(t1):
                while t1 in ItemTags[i1]:
                    ItemTags.get(i1).remove(t1)

    for user, tagu in UserTags.items():
        if len(tagu) > 4:
            User_Tags[user] = tagu
    for item, tagi in ItemTags.items():
        if len(tagi) > 4:
            Item_Tags[item] = tagi
    print(len(User_Tags))
    print(len(Item_Tags))

    for user1, tagi in User_Tags.items():
        if user1 not in UserIndex.keys():
            UserIndex[user1] = i
            i += 1
    for item1 in Item_Tags:
        if item1 not in ItemIndex.keys():
            ItemIndex[item1] = j
            j += 1
    for each in TagSet:
        TagIndex[each] = k
        k += 1
    TagCountUser = Get_tag_num(TagSet, User_Tags)
    TagCountItem = Get_tag_num(TagSet, Item_Tags)

    output1 = open('data/M1/UserTags.pkl', 'wb')
    pickle.dump(User_Tags, output1, protocol=2)
    output1.close()

    output2 = open('data/M1/ItemTags.pkl', 'wb')
    pickle.dump(Item_Tags, output2, protocol=2)
    output2.close()

    output3 = open('data/M1/UserIndex.pkl', 'wb')
    pickle.dump(UserIndex, output3, protocol=2)
    output3.close()

    output4 = open('data/M1/ItemIndex.pkl', 'wb')
    pickle.dump(ItemIndex, output4, protocol=2)
    output4.close()

    output5 = open('data/M1/TagIndex.pkl', 'wb')
    pickle.dump(TagIndex, output5, protocol=2)
    output5.close()

    output6 = open('data/M1/TagCountUser.pkl', 'wb')
    pickle.dump(TagCountUser, output6, protocol=2)
    output6.close()

    output7 = open('data/M1/TagCountItem.pkl', 'wb')
    pickle.dump(TagCountItem, output7, protocol=2)
    output7.close()

if __name__ == '__main__':
    CleanData()
    print(0)
