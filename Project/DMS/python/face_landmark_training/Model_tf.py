import numpy as np
import tensorflow.keras.optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import tensorflow as tf
import cv2
from glob import glob
import matplotlib.pyplot as plt

hidden_node = [32, 8, 4, 128]


def my_test_cnn_model_Sequential(input_shape=None, output_shape=None, lr=0.001):
    try:
        if input_shape is None:
            raise Exception("input_shape is None")
        if output_shape is None:
            raise Exception("output_shape is None")
    except Exception as e:
        print(e)

    model = Sequential()
    model.add(Conv2D(hidden_node[0], (3, 3), activation='tanh',
                     input_shape=(input_shape[1], input_shape[2], input_shape[3]),
                     padding='same'))  # 32 x 32 x 3
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # model.add(Conv2D(hidden_node[1], (3, 3), activation='tanh', padding='same'))
    # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # model.add(Conv2D(hidden_node[2], (3, 3), activation='tanh', padding='same'))
    # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(hidden_node[3], activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape[1], activation="relu"))
    model.summary()

    opt = tensorflow.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss="mean_squared_error",
                  optimizer=opt,
                  metrics=["accuracy"])
    return model


def my_test_cnn_model_functional(input_shape, output_shape, lr=0.001):
    try:
        if input_shape is None:
            raise Exception("input_shape is None")
        if output_shape is None:
            raise Exception("output_shape is None")
    except Exception as e:
        print(e)

    # if len(input_shape) == 3:
    X = tf.keras.layers.Input(shape=(input_shape[1], input_shape[2], input_shape[3]))
    # tanh
    H = tf.keras.layers.Conv2D(hidden_node[0], kernel_size=(5, 5), activation="tanh", padding='same')(X)
    H = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(H)

    H = tf.keras.layers.Conv2D(hidden_node[1], kernel_size=(5, 5), activation="tanh", padding='same')(H)
    H = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(H)
    H = tf.keras.layers.Flatten()(H)

    H = tf.keras.layers.Dense(hidden_node[2], activation="tanh")(H)

    Y = tf.keras.layers.Dense(output_shape[1], activation="relu")(H)  # ??? sigmaid??? ??????????
    model = tf.keras.models.Model(X, Y)

    model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss="mean_squared_error",
                  optimizer=opt,
                  metrics=["accuracy"])

    return model


