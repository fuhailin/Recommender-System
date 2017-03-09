#!/usr/bin/env python
#-*- coding:utf-8 -*-
import pandas
import pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from numpy import *
import numpy
import pickle

class ProbabilisticMatrixFactorization:
    """
    Attributes
    * latent_d
    * learning_rate, regularization_strength
    * ratings, users, items
    * num_users, num_items
    * new_ratings, new_items (probably don't need these)

    Methods
    * likelihood
    * update
    * apply_updates
    * try_updates
    * undo_updates
    * print_latent_vectors
    * save_latent_vectors
    """

    def __init__(self, rating_df, latent_d=1, verbose=True):
        self.latent_d = latent_d
        self.learning_rate = .0001  # alpha
        self.regularization_strength = 0.1  # lambda

        self.ratings = numpy.array(rating_df).astype(float)
        self.converged = False

        self.num_users = int(numpy.max(self.ratings[:, 0]) + 1)
        self.num_items = int(numpy.max(self.ratings[:, 1]) + 1)

        if verbose:
            print(self.num_users, self.num_items, self.latent_d)
            print(self.ratings)

        # 初始化潜在用户特征和商品特征
        self.users = numpy.random.random((self.num_users, self.latent_d))
        self.items = numpy.random.random((self.num_items, self.latent_d))

        self.new_users = numpy.random.random((self.num_users, self.latent_d))
        self.new_items = numpy.random.random((self.num_items, self.latent_d))

    def likelihood(self, users=None, items=None):
        if users is None:
            users = self.users
        if items is None:
            items = self.items

        # empirical risk term
        sq_error = 0
        for rating_tuple in self.ratings:
            if len(rating_tuple) == 3:
                (i, j, rating) = rating_tuple
                weight = 1
            elif len(rating_tuple) == 4:
                (i, j, rating, weight) = rating_tuple
            i = int(i)
            j = int(j)
            r_hat = numpy.sum(users[i] * items[j])
            sq_error += weight * (rating - r_hat) ** 2

        # regularization term
        L2_norm = 0
        for i in range(self.num_users):
            for d in range(self.latent_d):
                L2_norm += users[i, d] ** 2

        for i in range(self.num_items):
            for d in range(self.latent_d):
                L2_norm += items[i, d] ** 2

        return -sq_error - self.regularization_strength * L2_norm

    def update(self):
        # updates_o holds updates to the latent features of users
        # updates_d holds updates to the latent features of items
        updates_o = numpy.zeros((self.num_users, self.latent_d))
        updates_d = numpy.zeros((self.num_items, self.latent_d))

        # batch update: run through all ratings for each iteration
        for rating_tuple in self.ratings:
            if len(rating_tuple) == 3:
                (i, j, rating) = rating_tuple
                weight = 1
            elif len(rating_tuple) == 4:
                (i, j, rating, weight) = rating_tuple
            i = int(i)
            j = int(j)
            # r_hat is the predicted rating for user i on item j
            r_hat = numpy.sum(self.users[i] * self.items[j])

            # update each feature according to weight accurracy
            for d in range(self.latent_d):
                updates_o[i, d] += self.items[j, d] * (rating - r_hat) * weight
                updates_d[j, d] += self.users[i, d] * (rating - r_hat) * weight

        # converge if likelihood changes by less than .1 or if learning rate goes below 1e-10
        # speed up by 1.25x if improving, slow down by 0.5x if not improving
        while (not self.converged):
            initial_lik = self.likelihood()

            print("  likelihood =", self.likelihood())
            print("  setting learning rate =", self.learning_rate)
            # apply updates to self.new_users and self.new_items
            self.try_updates(updates_o, updates_d)

            final_lik = self.likelihood(self.new_users, self.new_items)

            # if the new latent feature vectors are better, keep the updates, and increase the learning rate (i.e. momentum)
            if final_lik > initial_lik:
                self.apply_updates(updates_o, updates_d)
                self.learning_rate *= 1.25

                if final_lik - initial_lik < .1:
                    self.converged = True
            else:
                self.learning_rate *= .5
                self.undo_updates()
            if self.learning_rate < 1e-10:
                self.converged = True

        return not self.converged

    def apply_updates(self, updates_o, updates_d):
        for i in range(self.num_users):
            for d in range(self.latent_d):
                self.users[i, d] = self.new_users[i, d]

        for i in range(self.num_items):
            for d in range(self.latent_d):
                self.items[i, d] = self.new_items[i, d]

    def try_updates(self, updates_o, updates_d):
        """
        Update self.new_users and self.new_items with updates calculated with batch GD
        """
        alpha = self.learning_rate
        beta = -self.regularization_strength

        for i in range(self.num_users):
            for d in range(self.latent_d):
                self.new_users[i, d] = self.users[i, d] + \
                                       alpha * (beta * self.users[i, d] + updates_o[i, d])
        for i in range(self.num_items):
            for d in range(self.latent_d):
                self.new_items[i, d] = self.items[i, d] + \
                                       alpha * (beta * self.items[i, d] + updates_d[i, d])

    def undo_updates(self):
        # Don't need to do anything here
        pass

    def save_latent_vectors(self, prefix):
        self.users.dump(prefix + "%sd_users.pickle" % self.latent_d)
        self.items.dump(prefix + "%sd_items.pickle" % self.latent_d)


def load_rating_data(file_path='data/ml-100k/u.data'):
    prefer = []
    for line in open(file_path, 'r'):  # 打开指定文件
        (userid, movieid, rating, ts) = line.split('\t')  # 数据集中每行有4项
        uid = int(userid)
        mid = int(movieid)
        rat = float(rating)
        prefer.append([uid, mid, rat])
    data = array(prefer)
    ratings_df = pandas.DataFrame(data, columns=["user_id", "item_id", "rating"])
    return ratings_df

if __name__ == "__main__":
    ratings = load_rating_data()
    plot_ratings(ratings)
    pmf = ProbabilisticMatrixFactorization(ratings, latent_d=10)
    liks = []
    while (pmf.update()):
        lik = pmf.likelihood()
        liks.append(lik)
        print("L=", lik)
        pass

    plt.figure()
    plt.plot(liks)
    plt.xlabel("Iteration")
    plt.ylabel("Log Likelihood")

    plot_latent_vectors(pmf.users, pmf.items)
    plot_predicted_ratings(pmf.users, pmf.items)
    plt.show()

    pmf.print_latent_vectors()
    pmf.save_latent_vectors("data/models")

