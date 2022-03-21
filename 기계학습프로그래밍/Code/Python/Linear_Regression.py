import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d  # 3D 그래프를 그리는 lib
import pandas as pd


def Mse(y, y_hat):
    # 평균 제곱 오차
    return ((y - y_hat) ** 2).mean()


def Mes_value(y, predict_value):
    return Mse(np.array(y), np.array(predict_value))


def predict(X, W):
    print(X)
    predict_res = []
    if type(X[0]) == "list":  # feature가 많을경우
        for row in len(X):
            predict_val = 0.1
            for idx in len(row):
                predict_val += W[idx] * row[idx]
            predict_val += W[-1]
            predict_res.append(predict_val)
    else:
        for x in X:
            predict_res.append(W[0] * x + W[-1])

    return predict_res


def Mult_linear_regression():
    train = pd.read_csv("./data.csv")
    train_list = train.values.tolist()

    print(train_list)
    x1, x2, y = [], [], []
    for row in train_list:
        x1.append(row[0])
        x2.append(row[1])
        y.append(row[-1])

    x1_data = np.array(x1)
    x2_data = np.array(x2)
    y_data = np.array(y)

    w1, w2, b = 0, 0, 0
    epochs = 2001
    lr = 0.02

    for i in range(epochs):
        y_pred = w1 * x1_data + w2 * x2_data + b  # 예측값 구하기
        err = y_data - y_pred

        w1_diff = -(2 / len(x1_data)) * sum(x1_data * err)
        w2_diff = -(2 / len(x2_data)) * sum(x2_data * err)
        b_diff = -(2 / len(x1_data)) * sum(err)

        w1 -= lr * w1_diff
        w2 -= lr * w2_diff
        b -= lr * b_diff

        if i % 100 == 0:
            print("%.04f, %.04f, %.04f" % (w1, w2, b))

    ax = plt.axes(projection="3d")  # 그래프 종류 = 3D
    ax.set_xlabel("time")  # 폰트 없는 상태에서 영어를 제외한 다른 언어로 적으면 폰트 없다고 에러 뜸
    ax.set_ylabel("count")
    ax.set_zlabel("score")
    ax.scatter(x1, x2, y)
    plt.show()


def Liner_regression():
    w = 0
    b = 0

    data = [[2, 81], [4, 93], [6, 91], [8, 97]]

    x = [i[0] for i in data]
    y = [i[1] for i in data]

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y)
    plt.show()

    # list로 되어있는것을 numpy 배열로 바꾸기(인덱스를 주어 하나씩 불러와 계산 가능)
    x_data = np.array(x)
    y_data = np.array(y)

    lr = 0.03

    epochs = 2001

    for i in range(1000):
        y_pred = w * x_data + b
        err = y_data - y_pred

        w_diff = -(2 / len(x_data)) * sum(err * x_data)
        b_diff = -(2 / len(x_data)) * sum(err)
        if i % 100 == 0:
            print("w_diff=%.04f, b_diff=%.04f" % (w_diff, b_diff))
        w -= lr * w_diff
        b -= lr * b_diff

        # print("epoch=%.f, w=%.04f, b=%.04f" % (i, w, b))

    y_pred = w * x_data + b
    plt.scatter(x, y)
    plt.plot([min(x_data), max(x_data)], [min(y_pred), max(y_pred)])
    plt.show()
