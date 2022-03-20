import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
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
            print("w_diff=%.04f, b_diff=%.04f" %(w_diff, b_diff))
        w -= lr * w_diff
        b -= lr * b_diff

        #print("epoch=%.f, w=%.04f, b=%.04f" % (i, w, b))

    y_pred = w * x_data + b
    plt.scatter(x, y)
    plt.plot([min(x_data), max(x_data)], [min(y_pred), max(y_pred)])
    plt.show()
