import numpy as np


def Mse(y, y_hat):
    # 평균 제곱 오차
    return ((y - y_hat) ** 2).mean()


def Mes_value(y, predict_value):
    return Mse(np.array(y), np.array(predict_value))


def predict(X, W):
    print(X)
    predict_res = []
    if type(X[0]) == "list":    #feature가 많을경우
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


