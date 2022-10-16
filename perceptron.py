import numpy as np

'''
Author: Antonio Lang
Date: 15 October 2022
'''

def perceptron_train(X,Y):
    weights = np.zeros(X.size)
    bias = 0

    '''
    while not Converged:
        for sample in samples:
            label <- y[index]
            prediction <- x \cdot w + bias

            if label * prediction <= 0:
                weights, b = update(weights, bias, sample, label)
    '''

    return (weights, bias)

def perceptron_test(X,Y,weights, bias):
    pass

def activiation():
    pass

def update(weights, bias, sample, label):
    
    return weights, bias