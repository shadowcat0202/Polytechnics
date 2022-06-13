import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# pytoch에 Dataset, DataLoader 라이브러리 이용,
# 데이터를 쉽게 load가능
# Data shuffle 가능, Batch 등과 같은 옵션 지정 가능
# augmentation: Transform 별도의 라이브러리 존재
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

x_train = torch.FloatTensor([
    [73, 80, 75],
    [98, 60, 71],
    [78, 21, 100],
    [64, 86, 70],
    [23, 56, 33],
    [43, 62, 86],
    [77, 81, 88],
    [22, 56, 74]
])

y_train = torch.FloatTensor([
    [124],
    [235],
    [663],
    [324],
    [234],
    [754],
    [346],
    [964]
])

dataset = TensorDataset(x_train, y_train)
'''
batch_size 는 통상적으로 2의 배수 자주 사용
shuffle=True: (권장)Epoch마다 데이터셋을 섞어서 데이터가 학습되는 순서를 변경 모델이 데이터 셋의 순서에 익숙해 지는 것을 방지.
'''
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
# nn.Linear(input_dim, output_dim)
print(x_train.shape)
print(y_train.shape)

model = nn.Linear(x_train.shape[1], y_train.shape[1])
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs):
    print("===========================================")
    for batch_idx, sampels in enumerate(dataloader):
        x_train, y_train = sampels
        # H(x) 계산 (prediction)
        pred = model(x_train)

        # cost 계산
        cost = F.mse_loss(pred, y_train)

        # optimization
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()  # w, b update

        print(f"epoch:{epoch:4d}/{nb_epochs}, "
              f"Batch:{batch_idx + 1}/{len(dataloader)}, "
              f"Cost:{cost.item():6f}")

        
        # cost: Tensor, Tensor에서 가져오기 위해서 .item()사용
