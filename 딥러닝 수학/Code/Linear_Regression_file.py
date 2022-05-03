from file_read import read_value
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

# print(read_value("./score_mlr03.txt"))
# w, b 초기값은 랜덤 값으로 셋팅
# print(model().parameters) : Random값이 들어있음
# Iteration or epoch을 돌면서 model parameter의 값들을 update
# 실행시마다 결과값이 다른것을 방지하기 위해서
torch.manual_seed(1)

# train_data 를 일단 읽어옴
origin = read_value("./score_mlr03.txt")
# ndarray 변경
data = np.array(origin)
# train_data 로부터 x, y를 구분 해야함
# x = [mid, final, report] y = [final]

# print(origin.shape)

x_train = data[:-5, :-1]
y_train = data[:-5, -1]
# print(x_train)
# print(y_train)

# array type의 2차원 배열을 tensor 배열로 바꿔주어야 한다
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
# print(x_train)
# print(y_train)


# 모델 설정 torch.nn.Linear(input_din, output_dim)
model = nn.Linear(3, 1)
# optimize 설정
# optimizer = optim.SGD(model.parameters(), lr=0.00005)    #SGD(Stochastic Gradient Descent) = 경사하강법
optimizer = optim.Adam(model.parameters(), lr=0.0005)

nb_epoc = 100000
for epoc in range(nb_epoc + 1):
    pdic = model(x_train)   # forwarding (H(x) 계산: wx + b)

    cost = F.mse_loss(pdic, y_train)    # pytorch에서 제공하는 MSE 함수
    optimizer.zero_grad()   # 누적되는것을 방지하기 위해 zeor로 만들고 시작해야한다
    cost.backward()  # Backward 연산
    optimizer.step()  # w, b update

    if epoc % 10000 == 0:
        print(f"Epoch {epoc:4d}/{nb_epoc} Cost: {cost.item():.6f}")

# model_parm = list(model.parameters())
# print(model_parm[0], model_parm[1])


# test dataset 가져오기
x_test = torch.FloatTensor(data[-5:, :-1])
y_test = torch.FloatTensor(data[-5:, -1])
print(model(x_test))
print(y_test)

