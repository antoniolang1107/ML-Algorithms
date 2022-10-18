import numpy as np

'''
Author: Antonio Lang
Date: 18 October 2022
'''

def perceptron_train(X,Y):
    weights = np.zeros(X.shape[1]) # one weight per feature
    bias = 0
    epoch_limit = 100
    epoch = 0
    update = True
    while update is True and epoch <= epoch_limit:
        update = False
        for i, sample in enumerate(X):
            label = Y[i]
            prediction = activation(weights, bias, sample)
            if label * prediction <= 0: # if incorrect prediction
                weights, bias = update_values(weights, bias, sample, label)
                update = True
        epoch += 1
    return (weights, bias)

def perceptron_test(X_test,Y_test,w, b):
    num_correct = 0
    for i, sample in enumerate(X_test):
        if Y_test[i] * activation(w,b,sample) > 0: num_correct += 1
    return num_correct / len(X_test) # perceptron accuracy

def activation(weights, bias, sample):
    return np.dot(weights, sample) + bias # w*x + b

def update_values(weights, bias, sample, label):
    for i in range(len(weights)):
        weights[i] = weights[i] + sample[i] * label
    bias = bias + label
    return weights, bias