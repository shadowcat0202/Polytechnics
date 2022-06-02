import random

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import tensorflow.keras.optimizers
import numpy as np


class look_cnn:
    def __init__(self):
        # (H, W)
        self.model = None
        self.input_data_shape = (36, 86, 1)
        self.output_data_shape = 3

    def make_model(self, lr=0.001):
        # acc=0.999 loss=0.00014?
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

    # 테스트 용
    # def make_model(self, lr=0.001):
    #     # 수치 변경 테스트
    #     X = Sequential()
    #     X.add(Conv2D(256, (5, 5), activation='relu',  # Conv2D 필터 개수에 따른 차이는 미확인 상태
    #                  input_shape=self.input_data_shape,
    #                  padding='same'))
    #     X.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    #     X.add(Conv2D(128, (2, 2), activation='relu',  # Conv2D 필터 개수에 따른 차이는 미확인 상태
    #                  input_shape=self.input_data_shape,
    #                  padding='same'))
    #     X.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    #
    #     # X.add(Conv2D(8, (3, 3), activation='tanh', padding='same'))
    #     # X.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))
    #     X.add(Flatten())
    #
    #     X.add(Dense(255, activation='relu'))
    #     X.add(Dense(16, activation='relu'))
    #     X.add(Dense(self.output_data_shape, activation='softmax'))
    #     # X.add(Dense(self.output_data_shape))
    #     X.summary()
    #
    #     opt = tensorflow.keras.optimizers.Adam(learning_rate=lr)
    #     X.compile(loss="categorical_crossentropy",
    #               optimizer=opt,
    #               metrics=["accuracy"])
    #     # https://cheris8.github.io/artificial%20intelligence/DL-Keras-Loss-Function/
    #     # Binary classification
    #     # sigmoid
    #     # binary_crossentropy
    #     # Dog vs cat, Sentiemnt analysis(pos / neg)
    #     #
    #     # Multi-class, single-label classification
    #     # softmax
    #     # categorical_crossentropy
    #     # MNIST has 10 classes single label (one prediction is one digit)
    #     #
    #     # Multi-class, multi-label classification
    #     # sigmoid
    #     # binary_crossentropy
    #     # News tags classification, one blog can have multiple tags
    #     #
    #     # Regression to arbitrary values
    #     # None
    #     # mse
    #     # Predict house price(an integer / float point)
    #     #
    #     # Regression to values between 0 and 1
    #     # sigmoid
    #     # mse or binary_crossentropy
    #     # Engine health assessment where 0 is broken, 1 is new
    #     return X

    def model_fit(self, EPOCH=10, BATCH_SIZE=10, X_train=None, Y_train=None, X_test=None, Y_test=None, _save_path="", file_name="default", lr=0.001):
        try:
            if X_train is None or Y_train is None:
                raise Exception("X_train or Y_train is None")
        except Exception as e:
            print(e)

        self.model = self.make_model(lr=lr)
        with tf.device("/device:GPU:0"):
            if X_test is None or Y_test is None:
                hist = self.model.fit(X_train, Y_train,
                                      epochs=EPOCH, batch_size=BATCH_SIZE,
                                      validation_data=(X_test, Y_test),
                                      verbose=1)
                self.draw_model_fit_graph(hist, tset=True)
            else:
                hist = self.model.fit(X_train, X_train,
                                      epochs=EPOCH, batch_size=BATCH_SIZE,
                                      verbose=1)
                self.draw_model_fit_graph(hist)
        self.model.save(_save_path + file_name + ".h5")
        print(f"{_save_path+file_name}.h5 save completion")
        return hist

    def img_split_eye(self, img, eye_landmark):
        """
        :argument
            img: 한 프레임(이미지)
            eye_landmark: 눈의 랜드마크 좌표
        :return
            result: img에서 눈 영역의 이미지를 잘라서 반환
            border: 잘라낸 눈의 영역의 [(p1.x, p1.y), (p2.x, p2.y)]
        """
        W = [i[0] for i in eye_landmark]
        H = [i[1] for i in eye_landmark]

        W_min, W_max = min(W), max(W)
        H_min, H_max = min(H), max(H)

        W_btw = W_max - W_min
        H_btw = H_max - H_min

        W_border_size = W_btw * 0.2
        H_border_size = H_btw * 0.5

        p1_W_border = int(W_min - W_border_size)
        p1_H_border = int(H_min - H_border_size)
        p2_W_border = int(W_max + W_border_size)
        p2_H_border = int(H_max + H_border_size)

        border = [
            (p1_W_border, p1_H_border),
            (p2_W_border, p2_H_border)
        ]
        result = np.array(img[p1_H_border:p2_H_border, p1_W_border:p2_W_border])
        return result, border

    def load_npy(self, _path):
        """
        :param _path: .npy file full path
        :return:
            NPY: ndarray type data
        """
        print(f"{_path[_path.rfind('/') + 1:]} loading...")
        result = np.load(_path)
        print(f"{result.shape} nparray load completion")
        return result

    def eye_crop(self, img, d=1):
        """
        cnn 모델에 입력으로 들어갈 shape를 맞춰주기 위한 함수
        :param
            img: 이미지
            d: 차원
        :return
            result: ([1 | N], self.input_data_shape[0], self.input_data_shape[1], self.input_data_shape[2]) 형식의 이미지
        """
        # np.expand_dims(img, axis=0)   # 앞에 차원 추가
        # np.expand_dims(img, axis=-1)  # 뒤에 차원 추가
        result = None
        if len(img.shape) == 2:  # 들어오는 이미지의 차원이 명시 되어 않는 경우
            img = cv2.resize(img, (self.input_data_shape[0], self.input_data_shape[1]))
            result = img.reshape(1, self.input_data_shape[0], self.input_data_shape[1], d)
            # result = np.expand_dims(img, axis=-1)
        elif len(img.shape) == 3:  # 이미지의 차원이 명시 되어 있는 경우
            if img.shape[2] == 3:  # 3 차원 이라면
                img, _, _ = cv2.split(img)  # 한개의 차원만 사용한다
                img = cv2.resize(img, (self.input_data_shape[0], self.input_data_shape[1]))
                result = img.reshape(self.input_data_shape[0], self.input_data_shape[1], d)
                # result = np.expand_dims(img, axis=-1)
        return result


    def draw_model_fit_graph(self, hist, tset=False):
        """        
        acc loss 그래프 그려주는 함수
        :param hist: model fit 한 결과값
        :param tset: test 데이터 셋 존재 여부
        :return: None
        """
        fig, loss_ax = plt.subplots()
        acc_ax = loss_ax.twinx()

        loss_ax.plot(hist.history['loss'], 'y', label='train loss')
        acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
        if tset:
            loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
            acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuray')
        loss_ax.legend(loc='upper left')
        acc_ax.legend(loc='lower left')
        plt.show()


lc = look_cnn()

lc.make_model(lr=0.001)