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
origin = read_value("./sell_house.txt")
cnt = len(origin)

# ndarray 변경
data = np.array(origin)

X_train = data[:-5, 1:-1]
Y_train = data[:-5, -1]
X_train = torch.FloatTensor(X_train)
Y_train = torch.FloatTensor(Y_train)

X_test = data[-5:, 1:-1]
Y_test = data[-5:, -1]
X_test = torch.FloatTensor(X_test)
Y_test = torch.FloatTensor(Y_test)


def Linear_Regression_file():
    # 모델 설정 torch.nn.Linear(input_din, output_dim)
    model = nn.Linear(len(X_train[0]), 1)
    # optimize 설정
    # optimizer = optim.SGD(model.parameters(), lr=0.00005)    #SGD(Stochastic Gradient Descent) = 경사하강법
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    nb_epoc = 80000
    for epoc in range(nb_epoc + 1):
        pdic = model(X_train)  # forwarding (H(x) 계산: wx + b)

        cost = F.mse_loss(pdic, Y_train)  # pytorch에서 제공하는 MSE 함수
        # ======================필수!!!!====================== #
        optimizer.zero_grad()  # 누적되는것을 방지하기 위해 zeor로 만들고 시작해야한다
        cost.backward()  # Backward 연산
        optimizer.step()  # w, b update

        if epoc % 10000 == 0:
            print(f"Epoch {epoc:4d}/{nb_epoc} Cost: {cost.item():.6f}")

    # model_parm = list(model.parameters())
    # print(model_parm[0], model_parm[1])

    pred = model(X_test)
    # print(Y_test)
    # print(pred)
    return pred


def Linear_Regression_Proj_file():
    A = np.array(origin)
    A = A[:-5, 1:-1]
    A = A.reshape(cnt-5, len(A[0]))
    # print(A)

    b = np.array(origin)
    b = b[:-5, -1]
    b = b.reshape(cnt-5, 1)
    # print(b)

    x_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), b)
    # print(x_hat)

    test_A = np.array(origin)
    test_A = test_A[-5:, 1:-1]
    test_Y = np.array(origin)
    test_Y = test_Y[-5:, -1]
    pred = np.matmul(test_A, x_hat)
    # print(test_Y)
    # print(pred)
    return pred


Y_test = Y_test.reshape(1, len(Y_test))
# print(Y_test)
LR = Linear_Regression_file()
LRproj = Linear_Regression_Proj_file()
print(Y_test)
print(LR)
print(LRproj)
# with open("result.txt", "w") as f:
#     f.write("GT\tML\tLinear Algerbra_Proj\n")
#     for i in range(len(Y_test[0])):
#         f.write(f"{round(Y_test[0][i],2)}\t{round(LR[0][i],2)}\t{round(LRproj[0][i],2)}\n")

print("finish")

