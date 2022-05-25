from tensorflow.keras.datasets import mnist
from keras.utils import np_utils

import numpy
import os

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten


def load_dataset(online=False):
    if online:
        (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
    else:
        path = "D:/JEON/Polytechnics/기계학습프로그래밍/Code/Python/dataset/mnist.npz"
        (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data(path)
    return (X_train, Y_train), (X_test, Y_test)


def make_model():
    model = Sequential()
    model.add(Dense(512, input_dim=784, activation="relu"))
    model.add(Dense(10, activation="softmax"))  # sorftmax는 모든 출력의 합을 1로 나오게 한다 = 출력을 정규화한다 = 확률로 뽑아낸다
    model.summary()
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model


def make_model_CNN():
    model = Sequential()
    model.add(Conv2D(filters=40, kernel_size=(5, 5),
                     strides=(1, 1),  # 이동하는 범위(건너뛰는 범위)
                     activation="relu",
                     input_shape=(28, 28, 1),
                     padding="same"))  # 28 x 28
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # 단순 사이즈를 줄이는 것이기 때문에 W가 늘어나지는 않는다 14 x 14
    model.add(Conv2D(64, (5, 5), activation="relu", padding="same"))  # Conv2d(filter, kernel_size 부터 시작한다)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # 단순 사이즈를 줄이는 것이기 때문에 W가 늘어나지는 않는다

    model.add(Flatten())  # 7 x 7 로 만들어진 이미지를 1 차원으로 핀다
    model.add(Dense(1000, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    model.summary()

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model


def train(model, X, Y):
    MY_EPOCH = 10
    MY_BATCHSIZE = 200
    Y = tf.keras.utils.to_categorical(Y, 10)
    history = model.fit(X, Y, epochs=MY_EPOCH, batch_size=MY_BATCHSIZE)
    filename = "./model/cnn_e({0}).h5".format(MY_EPOCH)
    model.save(filename)
    return history


device = tf.device('cuda')

numpy.random.seed(0)

(train_set, train_label), (test_set, test_label) = load_dataset()

# NN으로 할때 1차원으로 만들어주어야 하기 때문에
# train_set_1d = train_set.reshape(len(train_set), train_set.shape[1] * train_set.shape[2])
# test_set_1d = test_set.reshape(len(test_set), test_set.shape[1] * test_set.shape[2])
# model = make_model_CNN()
# his = train(model, train_set_1d, train_label)

model = make_model_CNN()
# CNN은 2차원 배열(이미지)를 그대로 해주어야 한다
train_data_2d = train_set.reshape(train_set.shape[0], train_set.shape[1], train_set.shape[2], 1)
train(model, train_data_2d, train_label)

from tensorflow.keras.models import load_model

filename = "model/cnn_e(20).h5"
cnn = load_model(filename)
test_data_2d = test_set.reshape(test_set.shape[0], test_set.shape[1], test_set.shape[2], 1)
test_label = tf.keras.utils.to_categorical(test_label, 10)
cnn.evaluate(test_data_2d, test_label)
