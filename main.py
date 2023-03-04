import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple


# takes the data , turn into a numpy array transpose
# then return the first row as labels
# and the rest as the data
def data_preprocessing(path):
    data = pd.read_csv(path)
    data = np.array(data)
    data = data.T
    labels = data[0]
    data = data[1:]
    return labels, data


# initializes the weights takes a list of integers 'layers'
# each int represents the number of neurons
# the first and the last integers represent the number of inputs and outputs respectively
# returns a list of numpy arrays
def init_weights(layers):
    weights = []
    for i in range(1, len(layers)):
        weights.append(np.array(np.random.uniform(-0.5, 0.5, (layers[i], layers[i - 1]))))
    return weights


# initializes the biases takes a list of integers 'layers'
# each integer represents the number of neurons of the hidden layers
# returns a list of numpy arrays
def init_biases(layers):
    biases = []
    for i in range(1, len(layers)):
        biases.append(np.array(np.random.uniform(-0.5, 0.5, (layers[i], 1))))
    return biases


def init_params(layers):
    return init_weights(layers), init_biases(layers)


def ReLU(Z):
    return np.maximum(0, Z)


# TODO
def sigmoid(Z):
    pass


# W = weights
# B = biases
# X = inputs
# Z is the new updated list of weights we save these to do back propagation
# A is the values of the neurons
def forward_propagation(W, B, X):
    Z = [W[0].dot(X) + B[0]]
    A = [ReLU(Z[0])]
    for i in range(1, len(W)):
        Z.append(W[i].dot(A[i - 1]) + B[i])
        A.append(ReLU(Z[i]))
    return Z, A


def loss(a, y):
    m = y.shape[1]
    j = -1 / m * np.sum(a, y)


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T


def ReLU_derive(Z):
    return Z > 0


# TODO
def back_propagation(Z, A, W, B, X, Y):
    y = one_hot(Y)
    dZ = []
    dW = []
    dB = []
    dZ.append(A[0] - y)
    dB.append(dZ[0])
    for i in range(len(Z) - 1, 0, -1):
        dZ.append(dZ[len(dZ) - 1].dot(W[len(dZ) - 1] * ReLU_derive(Z[i])))
        dW.append()
    pass


# TODO
def update_params():
    pass
