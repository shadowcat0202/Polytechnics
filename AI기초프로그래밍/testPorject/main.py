import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from Perceptron import Perceptron


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # 마커와 컬러맵을 설정합니다
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 결정 경계를 그립니다
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 샘플의 산점도를 그립니다
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')


if __name__ == '__main__':
    # 불꽃 데이터 셋
    df = pd.read_csv('https://archive.ics.uci.edu/ml/' 'machine-learning-databases/iris/iris.data', header=None)

    df.tail()

    # df = pd.read.cvs('your/local/path/to/iris.data', header=None) #로컬 디렉토리에 있다면 이 코드 사용

    # setosa와 versicolor를 선택합니다
    y = df.iloc[0:100, 4].values
    # setosa일 경우 -1, 아니면(versicolor) 1
    y = np.where(y == 'Iris-setosa', -1, 1)  # y = (y == 'Iris-setosa') ? -1 : 1

    # 꽃받침 길이와 꽃잎 길이를 추출합니다
    X = df.iloc[0:100, [0, 2]].values

    # 산점도를 그립니다
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')  # 데이터의 종류 이름 설명을 어느 위치에 그려 넣을 것인가

    plt.show()  # 화면을 먼저 그려 넣는다

    # 선형회귀 객체 생성
    ppn = Perceptron(eta=0.1, n_iter=10)
    # 학습시작
    ppn.fit(X, y)

    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of errors')
    plt.show()

    plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')

    plt.show()
