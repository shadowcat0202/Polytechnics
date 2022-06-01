import random

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential
import tensorflow as tf
import cv2
from glob import glob
import matplotlib.pyplot as plt
import tensorflow.keras.optimizers
import numpy as np


class look_cnn:
    def __init__(self):
        self.input_data_shape = (36, 86, 1)
        self.output_data_shape = 3

    def make_model(self, lr=0.001):
        X = Sequential()
        X.add(Conv2D(256, (5, 5), activation='relu',  # Conv2D 필터 개수에 따른 차이는 미확인 상태
                     input_shape=self.input_data_shape,
                     padding='same'))
        X.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        X.add(Conv2D(128, (2, 2), activation='relu',  # Conv2D 필터 개수에 따른 차이는 미확인 상태
                     input_shape=self.input_data_shape,
                     padding='same'))
        X.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        # X.add(Conv2D(8, (3, 3), activation='tanh', padding='same'))
        # X.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))
        X.add(Flatten())

        X.add(Dense(255, activation='relu'))
        X.add(Dense(16, activation='relu'))
        X.add(Dense(self.output_data_shape, activation='softmax'))
        # X.add(Dense(self.output_data_shape))
        X.summary()

        opt = tensorflow.keras.optimizers.Adam(learning_rate=lr)
        X.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])
        # https://cheris8.github.io/artificial%20intelligence/DL-Keras-Loss-Function/
        # Binary classification
        # sigmoid
        # binary_crossentropy
        # Dog vs cat, Sentiemnt analysis(pos / neg)
        #
        # Multi-class, single-label classification
        # softmax
        # categorical_crossentropy
        # MNIST has 10 classes single label (one prediction is one digit)
        #
        # Multi-class, multi-label classification
        # sigmoid
        # binary_crossentropy
        # News tags classification, one blog can have multiple tags
        #
        # Regression to arbitrary values
        # None
        # mse
        # Predict house price(an integer / float point)
        #
        # Regression to values between 0 and 1
        # sigmoid
        # mse or binary_crossentropy
        # Engine health assessment where 0 is broken, 1 is new
        return X


look = look_cnn()
print("X, Y nparray loading...")
X = np.load("D:/JEON/dataset/look_direction/X_classification_leftmidright.npy")
X = X / 255
Y = np.load("D:/JEON/dataset/look_direction/Y_classification_leftmidright.npy")
print(f"X{X.shape}, Y{Y.shape} nparray load completion")

# test_list = [random.randint(0, 48036) for _ in range(20)]
# for test in test_list:
#     print(Y[test])
#     cv2.imshow(f"{test}", X[test])
#     cv2.waitKey(delay=5000)
#
# exit()

# cv2.imshow("sample", X[5000])
# cv2.waitKey(delay=3000)
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1, shuffle=True)
print(f"train number:{len(train_X)}, test number:{len(test_X)}")

tf.random.set_seed(0)
MY_EPOCH = 10
MY_BATCHSIZE = 20
model = look.make_model(lr=0.001)
with tf.device("/device:GPU:0"):
    hist = model.fit(train_X, train_Y, epochs=MY_EPOCH, batch_size=MY_BATCHSIZE, verbose=1,
                     validation_data=(test_X, test_Y))
model.save(f"D:/JEON/dataset/look_direction/eye_leftmidright_classification_model.h5")

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()