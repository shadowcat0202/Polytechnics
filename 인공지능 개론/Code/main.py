import numpy as np
from matplotlib import pyplot as plt


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



if __name__ == '__main__':
    print(np.array([1, 2, 3, 4]))
    print(np.arange(10))
    c = np.linspace(-3.14, 3.14, 1000)
    # print(c)
    showGraph(np.cos, c)
