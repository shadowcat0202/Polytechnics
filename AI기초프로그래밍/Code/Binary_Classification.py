import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

x_data =[[1,2],[2,3],[3,1],
         [4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

w = torch.zeros((2,1), requires_grad=True)  # sigmoid(0) = 0.5 이기 때문에 w초기값을 0으로 해도 괜찮다
b = torch.zeros(1, requires_grad=True)

nb_epochs = 20
for epochs in range(nb_epochs):
    pass


