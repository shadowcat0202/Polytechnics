import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 활성화 함수 시그모이드 형식으로 하겠다는 의미
def sigmoid(x):
    return 1 / (1 + np.e ** (-x))


def Logistic_regression():
    train_data = pd.read_csv("./data.csv")
    train_data = train_data.values.tolist()
    print(train_data)

    x_data = [i[0] for i in train_data]
    y_data = [i[1] for i in train_data]

    plt.scatter(x_data, y_data)
    plt.xlim(0, 15)
    plt.ylim(-.1, 1.1)

    w, b = 0, 0
    lr = 0.05
    epochs = 2001

    for i in range(epochs):
        for x, y in train_data:
            w_diff = x * (sigmoid(w * x + b) - y)
            b_diff = sigmoid(w * x + b) - y

            w -= lr * w_diff
            b -= lr * b_diff

        if i % 400 == 0:
            print("%4d, %.04f, %.04f" % (i, w, b))

    plt.scatter(x_data, y_data)
    plt.xlim(0, 15)
    plt.ylim(-0.1, 1.1)
    x_range = (np.arange(0, 15, 0.1))
    plt.plot(np.arange(0, 15, 0.1), np.array([sigmoid(w * x + b) for x in x_range]))
    plt.show()
