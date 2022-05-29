from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import tensorflow as tf
import cv2
from glob import glob
import matplotlib.pyplot as plt
import tensorflow.keras.optimizers
from tensorflow.keras.models import Sequential
import numpy as np


class eye_Net:
    def __init__(self):
        self.img_size = (90, 90, 1)

    # https://www.kaggle.com/datasets/tauilabdelilah/mrl-eye-dataset?resource=download
    def model(self, lr=0.001):
        X = Sequential()
        X.add(Conv2D(8, (3, 3), activation='relu',  # Conv2D 필터 개수에 따른 차이는 미확인 상태
                     input_shape=self.img_size,
                     padding='same'))  # input_shape = (None, 90, 90, 1)
        X.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))

        # X.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        # X.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))
        X.add(Flatten())

        X.add(Dense(13, activation='relu'))  # 8일때 acc=0.5 언저리
        X.add(Dense(1, activation='sigmoid'))  # output = 1
        X.summary()

        opt = tensorflow.keras.optimizers.Adam(learning_rate=lr)
        X.compile(loss="binary_crossentropy",  # class가 이진 분류 문제의 손실함수는 binary_crossentropy
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

    def make_dataset(self, _path, _file_name):
        print("train image reading...")
        find_img_path = glob(_path + "open eyes/*.png")
        Y = np.full((len(find_img_path), 1), 1)
        X = np.array([np.expand_dims(cv2.resize(plt.imread(path), (90, 90)), axis=-1) for path in find_img_path])
        print("train label reading...")
        find_img_path = glob(_path + "close eyes/*.png")
        # np.r_[a, b] --> 수평으로 이어 붙이기 or np.r_[[a],[b]] --> 수직으로 이어 붙이기 와 같은 형식으로 사용 가능
        Y = np.vstack([Y, np.full((len(find_img_path), 1), 0)])
        X = np.vstack(
            [X, np.array([np.expand_dims(cv2.resize(plt.imread(path), (90, 90)), axis=-1) for path in find_img_path])])

        np.save(_path + f"X_{_file_name}", X)  # .npy
        np.save(_path + f"Y_{_file_name}", Y)  # .npy
        print("nparray train data save completion")
        return X, Y

    def load_dataset(self, _path, _file_name):
        print("X, Y nparray loading...")
        X = np.load(_path + f"X_{_file_name}.npy")
        Y = np.load(_path + f"Y_{_file_name}.npy")
        print(f"X{X.shape}, Y{Y.shape} nparray load completion")
        return X, Y

    def eye_predictor(self, _path):
        model = tf.keras.models.load_model(_path)
        print("load model")
        return model

    def make_input_shape(self, img):
        # print(f"make_input_shape(img){type(img)}, img.shape={img.shape}, len(img.shape)={len(img.shape)}")
        result = None
        if len(img.shape) == 2:
            print("흑백임")
            # 차원 증가 (a, b) --> (a, b, 1)
            img = cv2.resize(img, (self.img_size[0], self.img_size[1]))
            img = np.expand_dims(img, axis=-1)
            img = np.expand_dims(img, axis=0)
        elif len(img.shape) == 3:
            # print("다른 차원이 존재하는데",end=" ")
            if img.shape[2] == 3:
                # print("RGB 다 있음", end=" ")
                img, _, _ = cv2.split(img)  # 흑백만 남긴다
                img = cv2.resize(img, (self.img_size[0], self.img_size[1]))
                result = img.reshape(1, self.img_size[0], self.img_size[1], 1)
                # img = np.expand_dims(img, axis=-1)
                # result = np.expand_dims(img, axis=0)
        # (1, 90, 90, 1)로 만들어주기 위해서 reisze(90,90)
        # -> 차원 추가 뒤(axis=-1)(90, 90, 1)
        # -> 차원 추가 앞(axis= 0)(1, 90, 90, 1)
        cv2.imshow("wow", result[0])
        return result / 255

    def eye_img_split(self, img, eye_lm):
        row = [i[0] for i in eye_lm]
        col = [i[1] for i in eye_lm]

        row_min, row_max = min(row), max(row)
        col_min, col_max = min(col), max(col)

        row_mid = row_max - row_min
        col_mid = col_max - col_min

        row_border1 = int(row_min - row_mid * 0.5)
        col_border1 = int(col_min - col_mid * 1.5)
        row_border2 = int(row_max + row_mid * 0.5)
        col_border2 = int(col_max + col_mid * 1.5)

        result = np.array(img[col_border1:col_border2, row_border1:row_border2])
        return result, [row_border1, col_border1, row_border2, col_border2]
