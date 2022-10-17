from scipy import spatial

'''
Author: Antonio Lang
Date: 15 October 2022
'''

def KNN_test(X_train,Y_train,X_test,Y_test,K):
    accuracy = 0
    for i, test_sample in X_test:
        distances = []
        for j, train_sample in X_train:
            # get the distances between points
            pass
        # sort the tuple list in ascending order
        # get the labels of the first k tuples
        # count num correct against Y_test

    return accuracy

def choose_K(X_train,Y_train,X_val,Y_val):
    best_acc = 0
    len_train = len(X_train)
    for i in range(1, len_train):
        num_correct = 0

        acc = num_correct / len_train
        if acc > best_acc: best_acc = acc
        pass # iterate over all values of k, get best acc
    pass # find the best value of K
