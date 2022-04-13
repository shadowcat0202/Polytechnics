import numpy as np
from matplotlib import pyplot as plt
import math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def lifeGame():
    start = int(input("초기 패턴 입력:"))
    currentList = list(start)
    while True:
        print("현재 상태")


def showGraph(function, arg):
    x = arg
    y = function(x)
    plt.plot(x, y)
    plt.show()


def LG():
    X = [0.0, 0.52, 1.05, 1.57, 2.09, 2.62, 3.14,3.67, 4.19, 4.71, 5.24, 5.76]
    Y = [0.0, 0.5, 0.87, 1.0, 0.87, 0.5, 0.0, -0.5, -0.87, -1.0, -0.87, -0.5]

    plt_row = 2
    plt_col = 2
    plt.subplot(plt_row, plt_col, 1)
    X = []
    Y = []
    for i in range(0, 361, 30):
        x = i * math.pi / 180
        X.append(x)
        Y.append(math.sin(x))
    plt.scatter(X, Y)

    lr = LinearRegression()
    X_2d = []
    for x in X:
        X_2d.append([x])
    lr.fit(X_2d, Y)
    plt.plot(X, lr.predict(X_2d))

    X = []
    Y = []
    plt.subplot(plt_row, plt_col, 2)
    X = np.linspace(0.2*math.pi, 12)
    Y = np.sin(X)
    plt.scatter(X, Y)
    lr = LinearRegression()
    X = X.reshape(-1,1)
    lr.fit(X, Y)
    plt.plot(X, lr.predict(X))

    plt.subplot(plt_row, plt_col, 3)
    poly_feat = PolynomialFeatures(degree=2)
    X_poly = poly_feat.fit_transform(X)
    # 변화된 x에 대하여 선형 회귀 한다.
    lr.fit(X_poly, Y)
    plt.scatter(X, lr.predict(X_poly))

    plt.subplot(plt_row, plt_col, 4)
    n = 100
    x = 6 * np.random.rand(n, 1) - 3
    y = 0.5 * x ** 2 + x + 2 + np.random.rand(n, 1)
    plt.scatter(x, y, s=2)

    poly_feat = PolynomialFeatures(degree=2)
    x_poly = poly_feat.fit_transform(x)
    lr = LinearRegression()
    lr.fit(x_poly, y)
    plt.scatter(x, lr.predict(x_poly), s=2)
    plt.show()

if __name__ == '__main__':
    LG()
