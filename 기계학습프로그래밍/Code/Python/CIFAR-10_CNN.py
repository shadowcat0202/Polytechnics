import os

import numpy
import tensorflow as tf

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten

from tensorflow.keras.models import load_model


MY_EPOCH = 10
MY_BATCHSIZE = 200
filename = f"./model/cnn_e({MY_EPOCH}).h5"

def make_model_CNN():
    model = Sequential()
    # Layer 1
    model.add(Conv2D(512, (5, 5), activation='relu',
                     input_shape=(32, 32, 3),
                     padding='same'))  # 32 x 32 x 3
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # 단순 사이즈를 줄이는 것이기 때문에 W가 늘어나지는 않는다 16 x 16

    # Layer 2
    model.add(Conv2D(256, (5, 5), activation='relu', padding='same'))  # Conv2d(filter, kernel_size 부터 시작한다)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # 단순 사이즈를 줄이는 것이기 때문에 W가 늘어나지는 않는다 8 x 8

    # Layer 3
    model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))  # Conv2d(filter, kernel_size 부터 시작한다)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # 단순 사이즈를 줄이는 것이기 때문에 W가 늘어나지는 않는다 4 x 4

    model.add(Flatten())  # 4 x 4 로 만들어진 이미지를 1 차원으로 핀다

    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation="softmax"))
    model.summary()

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model


def train(model, X, Y):
    X = X / 255.
    Y = tf.keras.utils.to_categorical(Y, 10)
    history = model.fit(X, Y, epochs=MY_EPOCH, batch_size=MY_BATCHSIZE)
    filename = "./model/cnn_e({0}).h5".format(MY_EPOCH)
    model.save(filename)
    return history


def test_all(x, y):
    model = load_model(filename)
    x = x / 255.
    y = tf.keras.utils.to_categorical(y, 10)
    test_loss, test_acc = model.evaluate(x, y)
    return test_loss, test_acc


device = tf.device('cuda')

numpy.random.seed(0)

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

cnn = make_model_CNN()
train(cnn, train_images, train_labels)
print(test_all(test_images, test_labels))
