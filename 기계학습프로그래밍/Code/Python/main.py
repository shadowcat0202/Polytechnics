import Linear_Regression as LineR
import Logistic_Regression as logR
import pandas as pd
import numpy as np
import tensorflow as tf

def stub():
    np.random.seed(3)
    tf.random.set_seed(3)

if __name__ == '__main__':
    LineR.Mult_linear_regression()
    stub()
