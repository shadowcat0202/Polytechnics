import torch
import torch.nn as nn  # 뉴런네트워크
import torch.nn.functional as f
import torch.optim as optim

data = [[1, 2], [2, 4], [3, 6]]
# X_train = torch.FloatTensor([[1], [2], [3]])  # 1.0 2.0 3.0
# Y_train = torch.FloatTensor([[2], [4], [6]])
X_train = torch.FloatTensor([[x[0]] for x in data])
Y_train = torch.FloatTensor([[x[1]] for x in data])
# H(x) = Wx + b (W와 b 값을 알아 내고 싶다)
# model 선언을 하고, 초기화
# 단순 선형회귀, 입력 dimension 1, 출력 dimension 1
# model 생성하게되면 model의 값은? model parameeters : w, b
model = nn.Linear(1, 1)
# print(list(model.parameters()))

# 최적화를 하기 위한 방법 정의
# optimizer로 경사하강법을 사용(SGD)
# learning rate 설정 lr = 0.01
# W := w - (lr * gradient)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 전체 데이터 셋(1,2,3)에 대해서 경사 하강법을 얼마나 할래?
nb_epochs = 2000
for epoch in range(nb_epochs + 1):
    pred = model(X_train)  # 예측값과 실제값의 차이를 계산

    # cost 계산 Loss function, Error 값을 계산
    cost = f.mse_loss(pred, Y_train)

    # cost를 기준으로 weight, bias를 업데이트
    # pytorch에서 이거를 사용해야 계산된 값이 누적되지 않는다
    optimizer.zero_grad()

    # cost를 기준으로 weight, bias를 업데이트
    # loss값을 줄이기 위해서
    cost.backward()
    optimizer.step()  # w, b 값이 업데이트

    if epoch % 100 == 0:
        weightBias = list(model.parameters())
        print(f"Cost:{cost.item()}")

print(f"finish:{weightBias[0]}\n{weightBias[1]}")

import numpy as np

A = np.array([[1, 1], [2, 1], [3, 1]])
b = np.array([2, 4, 6])
# np.matmul : 행렬곱
c = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), b)  # hat(X) = inv(A.T A) * A.T * b

print(c)