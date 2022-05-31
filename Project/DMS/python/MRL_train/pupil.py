import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential
import tensorflow.keras.optimizers
class Pupil:
    def __init__(self):
        self.input_img_size = (92, 92)
        self.output_size = 2

    def make_model(self, lr=0.001):
        X = Sequential()
        X.add(Conv2D(64, (4, 4), activation='tanh',
                     input_shape=(self.input_img_size[0], self.input_img_size[1], 1),
                     padding='same'))
        X.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        X.add(Conv2D(32, (4, 4), activation='tanh',
                     input_shape=(self.input_img_size[0], self.input_img_size[1], 1),
                     padding='same'))
        X.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        X.add(Flatten())

        X.add(Dense(64, activation='relu'))  # 8일때 acc=0.5 언저리 이부분 노드 개수에 따라 학습율에 변동이 크다
        # X.add(Dropout(0.25))   # 드랍 하지 말자 acc 0.5 수준밖에 안나옴
        X.add(Dense(32, activation='relu'))
        X.add(Dense(self.output_size))
        X.summary()

        opt = tensorflow.keras.optimizers.Adam(learning_rate=lr)
        X.compile(loss="MAE",  # class가 이진 분류 문제의 손실함수는 binary_crossentropy
                  optimizer=opt,
                  metrics=["accuracy"])
        return X

    def load_model(self, _path):
        m = tf.keras.models.load_model(_path)
        print("load model")
        return m
