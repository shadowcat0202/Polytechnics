import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd  # pandas ---> get_dummies() ---> Dataframe ---> array_ type변경
import numpy as np
# from tensorflow.python.data.experimental.ops.optimization import model

# one-hot encoding y_train 변경
# 1) pandas의 get_dummies() 사용



def useGetDummies(x, y):
    df = pd.DataFrame({
        "y_train": y
    })
    # print(data)
    df = pd.get_dummies(df["y_train"])
    # print(data_copy)
    new_array = np.array(df)
    # print(new_array)
    y_train_new = torch.FloatTensor(new_array)
    print(y_train_new)
    return torch.FloatTensor(x), y_train_new


# 2) torch: one_hot() 메서드는 존재하지 않는다
# 파이토치 시작할때 사용한 scatter unsqueeze 메서드 등을 이용해서 구현
def useScatterUnsqeeze(x, y):
    x_tF = torch.LongTensor(x)
    y_tF = torch.LongTensor(y)
    y_one_hot = torch.zeros(8, 3)
    # unsqueeze(0) ==> 1x8, unsqueeze(1) ==> 8x1
    # scatter: 흩어지다 (열 또는 행 방향으로 흩어짐)
    # scatter: parameter 첫번째: 어느 dim으로 확장할건지 결정
    # 3번째 Param: y_train 값에 해당하는 인덱스에 1을 삽입
    # _: inplace=True와 같은 개념
    y_one_hot.scatter_(1, y_tF.unsqueeze(1), 1)
    # y_one_hot.scatter_(1, y_tF.unsqueeze(1), 2)
    print(y_one_hot.shape)
    print(y_one_hot)
    return x_tF, y_one_hot

x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]

y_train = [2, 2, 2, 1, 1, 1, 0, 0]



x_train, y_one_hot = useGetDummies(x_train, y_train)
# x_train, y_one_hot = useScatterUnsqeeze(x_train, y_train)

# 3) check: nn.Linear(), F.CrossEntropy 사용시 (y_train 자체를 one-hot encoding으로 하지 않아도 자동으로 변환)

## 학습을 위한 준비
# 1) weight, base shape 정의
# 2) optimizer를 뭘 사용할지 등을 정의
# 3) for문에서 정의된 epoch수만큼 돌면서 w, b, 학습
# 3-1) 가설함수(hypothesis: softmax)
# 3-2) cost function: cross-entropy

"""
8x4 matrix multiplication 4x3 ===> 8x3
"""
w = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(3, requires_grad=True)
optimizer = optim.SGD([w, b], lr=0.1)

# model = nn.Linear(4, 3)
# optimizer = optim.SGD(model.parameters(), lr=0.1)


z = torch.FloatTensor([1, 2, 3])
# print(torch.exp(z[0]/(torch.exp(z[0]) + torch.exp(z[1]) + torch.exp(z[2]))))
# print(F.softmax(z))
# scatter, unsqueeze를 이용한 one-hot encoding 구현
# y_one_hot 사이즈 정의
# row 8, classes 3, 있으므로 8x3 one-hot encoding된 데이터 필요
correct_pred = 0
nb_epochs = 10
for epoch in range(nb_epochs + 1):
    # softmax = exp(wx + b) / sum(exp(wx+b))
    hx = F.softmax(x_train.matmul(w) + b, dim=1)

    # Cost Function
    # cross-entropy: 1/n(y*-log(h(x)) # *: elements-wise
    cost = (y_one_hot * -torch.log(hx)).sum(dim=1).mean()

    # pred = model(x_train)
    # cost = F.cross_entropy(pred, y_one_hot) # 위에서 구현한 cross_entropy에 log를 먹인 상태

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 1 == 0:
        # print(f"EP:{epoch:4d}/{nb_epochs}, cost:{cost.item():.6f}, acc:{acc * 100:3.2f}%")
        print(f"EP:{epoch:4d}/{nb_epochs}, cost:{cost.item():.6f}")




