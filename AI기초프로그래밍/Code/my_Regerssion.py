import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

from my_ML_Regression import LogisticRegression

x_data = [[1, 2], [2, 3], [3, 1],
          [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

nb_epochs = 100

"""
nn.Sequential() 사용
nn.Sequential() : nn.Module 층을 차례로 쌓을 수 있도록 하는 것
층을 쌓는 역할, wx + b ==> Sigmoid 여러 함수들을 연결시켜주는 역할로 이해해도됨.
"""


def nonUse_Class():
    model = nn.Sequential(
        nn.Linear(len(x_data[0]), len(y_data[0])),
        nn.Sigmoid()  # 출력은 시그모이드 함수 결과
    )

    optimizer = opt.SGD(model.parameters(), lr=1)
    for e in range(nb_epochs + 1):
        hypothesis = model(x_train)
        cost = F.binary_cross_entropy(hypothesis, y_train)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if e % 10 == 0:
            prediction = hypothesis >= torch.FloatTensor([0.5])
            correct_pred = (prediction == y_train)
            acc = correct_pred.sum().item() / len(correct_pred)
            print(f"EP:{e:4d}/{nb_epochs}, cost:{cost.item():.6f}, acc:{acc * 100:3.2f}%")


def use_Class():
    LR = LogisticRegression(len(x_train[0]), len(y_data[0]))
    optimizer = opt.SGD(LR.parameters(), lr=1)
    for e in range(nb_epochs + 1):
        hypothesis = LR.forward(x_train)
        cost = F.binary_cross_entropy(hypothesis, y_train)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        if e % 10 == 0:
            prediction = hypothesis >= torch.FloatTensor([0.5])
            correct_pred = (prediction == y_train)
            acc = correct_pred.sum().item() / len(correct_pred)
            print(f"EP:{e:4d}/{nb_epochs}, cost:{cost.item():.6f}, acc:{acc * 100:3.2f}%")

nonUse_Class()
print("=====================")
use_Class()
