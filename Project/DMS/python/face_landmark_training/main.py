# https://www.kaggle.com/code/richardarendsen/face-landmarks-with-cnn/notebook
import pprint
from glob import glob

import tensorflow.keras.optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
# from tensorflow.python.keras.optimizer_v2.adam import Adam
import pandas as pd
import cv2
from tensorflow.python.keras.layers import AvgPool2D
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD


def make_model(x, y):
    demention = 1
    if len(x.shape) == 4:
        demention = 3
    outShape = len(y[0])
    print(x.shape, outShape, demention)
    # model = Sequential()
    # model.add(Conv2D(32, (3, 3), padding='same', activation='tanh', input_shape=(inSahpe[1], inSahpe[2], demention)))
    # model.add(MaxPool2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(32, activation='tanh'))
    # model.add(Dropout(0.5))
    # model.add(Dense(outShape, activation='sigmoid'))
    # model.summary()

    model = Sequential()
    # Layer 1
    model.add(Conv2D(64, (5, 5), activation='tanh',
                     input_shape=(x.shape[1], x.shape[2], demention),
                     padding='same'))  # 32 x 32 x 3
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # 단순 사이즈를 줄이는 것이기 때문에 W가 늘어나지는 않는다 16 x 16
    # model.add(AvgPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Layer 2
    # model.add(Conv2D(32, (5,5), activation='tanh', padding='same'))  # Conv2d(filter, kernel_size 부터 시작한다)
    # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # 단순 사이즈를 줄이는 것이기 때문에 W가 늘어나지는 않는다 8 x 8
    # model.add(AvgPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Layer 3
    model.add(Conv2D(32, (5, 5), activation='tanh', padding='same'))  # Conv2d(filter, kernel_size 부터 시작한다)
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # 단순 사이즈를 줄이는 것이기 때문에 W가 늘어나지는 않는다 4 x 4
    # model.add(AvgPool2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Flatten())  # 4 x 4 로 만들어진 이미지를 1 차원으로 핀다

    model.add(Dense(16, activation='tanh'))
    model.add(Dense(outShape, activation="softmax"))
    model.summary()

    opt = tensorflow.keras.optimizers.Adam(learning_rate=0.5)
    # opt = SGD(lr=1.5, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="mean_squared_error",
                  optimizer=opt,
                  metrics=["accuracy"])
    return model




def read_text_to_np():
    file = open("../../dataset/face_landmark_train_set/Train_daytime/", "r")

    full_data = {}
    while True:
        line = file.readline()
        if not line:
            break
        sp = line.split("\t")
        full_data[sp[0]] = [(int(sp[i]), int(sp[i + 1])) for i in range(1, len(sp) - 1, 2)]
    file.close()
    return full_data


def make_X_Y(path=None):
    if path is None:
        print("경로 설정 해주세요")
        return None
    find_img_name = glob(path + "*.bmp")

    # img_name_list = [name[-9:] for name in glob(path + "*.bmp")]  # 이미지 파일 이름 전부 가져오기
    img_name_list = [name[name.rfind("\\"):] for name in glob(path + "*.bmp")]  # 이미지 파일 이름 전부 가져오기
    # X = np.array([plt.imread(path + name)[:, :, 0] for name in img_name_list])  # 이미지를 읽어와서 1차원으로 저장
    X = np.array([plt.imread(path + name) for name in img_name_list])  # 이미지를 읽어와서 3차원으로 저장
    print("img done")

    pos_array = []
    find_txt = glob(path + "*.txt")  # GT_Point.txt파일 이름 찾기
    print(find_txt)
    file = open(path + find_txt[0][find_txt[0].rfind("\\"):], "r")  # 절대 경로에서 txt파일 이름만 분리해서 해당 파일 열기
    # file = open(path + "GT_Points.txt", "r")
    cnt = 0
    while True:
        line = file.readline()
        cnt += 1
        if not line:
            break
        # '이미지번호.bmp' 123\t124\t234... 방식으로 저장 되어있기 때문에 숫자 부분만 분리
        pos_array.append(np.array(list(map(int, line.strip().split("\t")[5:]))))
        # print(cnt)
    Y = np.array(pos_array)
    print("pos done")
    return X, Y


# print(img_name_list)

# print(img_name_list)

# print(img.shape)    #(480, 720, 3)



# Xtrain, Ytrain = make_X_Y("../../dataset/face_landmark_train_set/Train_daytime/")
Xtrain, Ytrain = make_X_Y("../../dataset/face_landmark_train_set/train/5_chs/")
print(len(Ytrain[0]) / 2)

print(f"Xtype{type(Xtrain)}, Ytype{type(Ytrain)}, shape: {Xtrain.shape}")
Xtest, Ytest = make_X_Y("../../dataset/face_landmark_train_set/test/8_test_sj/")
print(type(Xtest))
print(type(Ytest))

MY_EPOCH = 10
MY_BATCHSIZE = 20

print(f"input_shape={Xtrain.shape}, output_size={len(Ytrain[0])}")
model = make_model(Xtrain, Ytrain)
with tf.device("/gpu:0"):
    # model.fit(Xtrain, Ytrain, batch_size=MY_BATCHSIZE, epochs=MY_EPOCH, validation_data=(Xtest, Ytest), verbose=1)
    model.fit(Xtrain, Ytrain, batch_size=MY_BATCHSIZE, epochs=MY_EPOCH, verbose=1)
filename = "./model/cnn_test({0}).h5".format(MY_EPOCH)
model.save(filename)

Ytrain_pred = model.predict(Xtest[0])
# for i in range(2, 21):
#     plt.imshow(img)
#     x, y = zip(df[img_name_list[0]][i])
#     plt.scatter(x, y)
#     plt.show()
