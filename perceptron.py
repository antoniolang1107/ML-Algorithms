import numpy as np

'''
Author: Antonio Lang
Date: 17 October 2022
'''

def perceptron_train(X,Y):
    weights = np.zeros(X.size)
    bias = 0
    iteration_limit = 500
    iteration = 0

    while update is True and iteration <= iteration_limit:
        update = False
        for i, sample in X:
            label = Y[i]
            prediction = activation(weights, bias, sample)

            if label * prediction <= 0:
                weights, bias = update(weights, bias, sample, label)
                update = True
        iteration += 1

    return (weights, bias)

def perceptron_test(X_test,Y_test,w, b):
    num_correct = 0

    for i, sample in X_test:
        if Y_test[i] == activation(w,b,sample): num_correct += 1

    return num_correct / len(X_test)

def activation(weights, bias, sample):
    return np.dot(weights, sample) + bias

def update(weights, bias, sample, label):
    for i in range(len(weights)):
        weights[i] = weights[i] * sample[i] + label
    bias = bias + label

    return weights, bias