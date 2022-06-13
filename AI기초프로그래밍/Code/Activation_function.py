import numpy as np
import matplotlib.pyplot as plt
from Linear_Regression import plot_variable


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    x = np.arange(-5.0, 5.0, 0.1)
    y1 = sigmoid(x)
    y2 = sigmoid(3 * x)
    y3 = sigmoid(x + 2)

    plot_variable(x, y1, z="r", label="Fitten Line")
    plot_variable(x, y2, z="g", label="Fitten Line")
    plot_variable(x, y3, z="b", label="Fitten Line")
    plt.show()
