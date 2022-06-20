# ML 모델 클래스화
# Linear, Logistic, Softmax Regression
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.optim as opt
import tensorflow as tf

class LogisticRegression(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear = nn.Linear(self.in_dim, self.out_dim)

        self.sigmoid = None
        self.relu = None

    def setActivateSigmoid(self):
        self.sigmoid = nn.Sigmoid()

    def setActivateRelu(self):
        self.sigmoid = nn.ReLU()

    def forwardSigmoid(self, x):
        return self.sigmoid(self.linear(x))

    def forwardRelu(self, x):
        return self.relu(self.linear(x))


def Diabetes():
    data = np.loadtxt("./dataset/data-03-diabetes.csv", delimiter=",", dtype=np.float32)
    # print(data)
    X = data[:, :-1]
    Y = data[:, -1]
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, shuffle=False)
    X_train = torch.from_numpy(X[:-20])
    X_test = torch.from_numpy(X[-20:])
    Y_train = torch.from_numpy(Y[:-20])
    Y_test = torch.from_numpy(Y[-20:])

    Y_train = Y_train.view(-1, 1)
    Y_test = Y_test.view(-1, 1)
    print(Y_test[0])

    LR = LogisticRegression(len(X_train[0]), 1)
    LR.setActivateSigmoid()
    optimizer = opt.SGD(LR.parameters(), lr=0.5)
    nb_epochs = 200
    for e in range(nb_epochs+1):
        hypothesis = LR.forwardSigmoid(X_train)
        cost = F.binary_cross_entropy(hypothesis, Y_train)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        if e % 10 == 0:
            prediction = hypothesis >= torch.FloatTensor([0.5])
            correct_pred = (prediction == Y_train)
            acc = correct_pred.sum().item() / len(correct_pred)
            print(f"EP:{e:4d}/{nb_epochs}, cost:{cost.item():.6f}, acc:{acc * 100:3.2f}%")
    pred = LR.forwardSigmoid(X_test)
    pred = pred >= torch.FloatTensor([0.5])
    print(pred)
    hit = 0
    count = 0

    for i in range(len(pred)):
        count += 1

        if pred[i].float() == Y_test[i].float():
            hit += 1
    print(round(hit/count, 3))


Diabetes()