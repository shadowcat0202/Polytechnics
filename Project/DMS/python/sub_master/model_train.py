import os

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import tensorflow.keras.optimizers
import numpy as np
from Look_CNN import look_cnn


def train():
    lc = look_cnn()
    X = lc.load_npy("D:/JEON/dataset/look_direction/X_classification_leftmidright.npy")
    Y = lc.load_npy("D:/JEON/dataset/look_direction/Y_classification_leftmidright.npy")

    X_train, X_test, Y_train, Y_test = lc.train_test_split(X, Y, test_size=0.1, shuffle=True)
    lc.model_fit(EPOCH=10, BATCH_SIZE=30,
                 X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, lr=0.005,
                 _save_path="D:/JEON/dataset/look_direction/")


# file_name_list = os.listdir("D:/JEON/dataset/look_direction/cnn_eyes_number_sort/")
# X = []
# for i, name in enumerate(file_name_list):
#     if i % 2000 == 0:
#         print(i)
#     file_path = "D:/JEON/dataset/look_direction/cnn_eyes_number_sort/" + name
#     img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
#     X.append(img.reshape(36, 86, 1)/255)
# X = np.array(X)
# np.save("D:/JEON/dataset/look_direction/X_classification_leftmidright", X)
# print(X.shape)
# print(X[0])



# Y = []
# file_list = os.listdir("D:/JEON/dataset/look_direction/cnn_eyes_number_sort/")
# for i, name in enumerate(file_list):
#     if name[0] in ['1','4']:    # right
#         Y.append([0])
#     elif name[0] in ['2','5']:  # center
#         Y.append([1])
#     else:   # left
#         Y.append([2])
#
# Y = tf.keras.utils.to_categorical(Y)
# Y = np.array(Y)
# print(Y[0])
# print(Y[-1])
# np.save("D:/JEON/dataset/look_direction/Y_classification_leftmidright", Y)

train()
