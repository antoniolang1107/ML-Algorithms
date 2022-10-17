import numpy as np
from scipy import spatial

'''
Author: Antonio Lang
Date: 17 October 2022
'''

def KNN_test(X_train,Y_train,X_test,Y_test,K):
    num_correct = 0
    distances = spatial.distance.cdist(X_test, X_train)
    for i, point_distances in enumerate(distances):
        label_dist = []
        for j, distance in enumerate(point_distances):
            label_dist.append((distance, Y_train[j]))
        sorted_dist = sorted(label_dist, key=lambda l:l[0], reverse=False) # sorts in ascending distance
        if Y_test[i] == get_prediction(sorted_dist, K): num_correct += 1
    return num_correct / len(X_train)

def choose_K(X_train,Y_train,X_val,Y_val):
    best_acc = 0
    best_k = 1
    len_train = len(X_train)
    for i in range(1, len_train): # iterate by 2
        num_correct = 0

        acc = num_correct / len_train
        if acc > best_acc: best_acc = acc
        # iterate over all values of k, get best acc
    # find the best value of K
    return best_k

def get_prediction(sorted_distances, K):
    label_sum = 0
    for i in range(K):
        label_sum += sorted_distances[i][1]
    return label_sum