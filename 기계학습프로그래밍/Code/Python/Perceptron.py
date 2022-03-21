import numpy as np


def MLP(x, w, b):
    y = np.sum(w * x) + b
    return 0 if y <= 0 else 1


def NAND(x1, x2, w, b):
    return MLP(np.array([x1, x2]), w, b)

def OR(x1, x2, w, b):
    return MLP(np.array([x1, x2]), w, b)

def XOR(x1, x2, w, b):
    return MLP(np.array([x1, x2]), w, b)


def Perceptron():
