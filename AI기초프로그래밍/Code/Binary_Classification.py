import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from Activation_function import sigmoid
seperate_line = "=====================================\n"

x_data = [[1, 2], [2, 3], [3, 1],
          [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

w = torch.zeros((2, 1), requires_grad=True)  # sigmoid(0) = 0.5 이기 때문에 w 초기값을 0으로 해도 괜찮다
# w = torch.randn((2, 1), requires_grad=True)  # randn으로 해도 상관 없다
b = torch.zeros(1, requires_grad=True)

optimizer = opt.SGD([w,b], lr=1)
nb_epochs = 1000
for epochs in range(nb_epochs):
    # 1. Hypothesis 함수 (Forward)
    # hx = sigmoid(x_train.matmul(w) + b)
    hx = 1 / (1 + torch.exp(-(x_train.matmul(w) + b)))
    # 2. Cost Function 사용
    # cost(hx) 1 : -log(h(x))
    # cost(hx) 0 : -log(1-h(x))
    cost = -((y_train * torch.log(hx)) + ((1 - y_train) * torch.log(1 - hx))).mean()
    # cost = F.binary_cross_entropy(hx, y_train)    # 동일
    
    # 3. 최적화 (los, cost, function을 미분하고, 경사타고 내려오면서 w, b 업데이트
    optimizer.zero_grad()   # 미분 누적을 막기 위해서
    cost.backward() # Loss function 미분
    optimizer.step()    # SGD, lr 만큼 내려가면서 w, b 업데이트
    if epochs % 100 == 0:
        print(f"EP:{epochs:4d}/{nb_epochs}, cost:{cost.item():.6f}")
print(w, b)

hypothesis = torch.sigmoid(x_train.matmul(w) + b)
print(hypothesis)

# sigmoid 출력 값이 0.5 이상이면 1 아니면 0
pred = []
for i in list(hypothesis):
    if i >= 0.5:
        pred.append(1)
    else:
        pred.append(0)
print(pred, end=seperate_line)

prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction, end=seperate_line)
