import numpy as np


def Sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def tanh(x, derivative=False):
    if derivative:
        return 1 - x ** 2
    return np.tanh(x)


def makeMatrix(i, j, fill=0.0):
    mat = []
    for i in range(i):
        mat.append([fill] * j)
    return mat
