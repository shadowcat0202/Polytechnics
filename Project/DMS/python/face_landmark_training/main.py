# https://www.kaggle.com/code/richardarendsen/face-landmarks-with-cnn/notebook
from glob import glob
import Model_tf

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
# from tensorflow.python.keras.optimizer_v2.adam import Adam
import pandas as pd
import cv2
from EYE import eye_Net

MY_EPOCH = 10
MY_BATCHSIZE = 5


def draw_plot(img, gt):
    gt = np.array([[gt[i], gt[i + 1]] for i in range(0, len(gt), 2)])
    plt.imshow(img)
    x, y = zip(*gt)
    plt.scatter(x, y, 2, color="r", marker="x")
    plt.show()


def draw_subplots(img, gt, row, col):
    print("draw_subplots")
    print(img.shape, row, col)
    if 3 > row * col:
        print("가로 세로 크기보다 이미지 크기가 더 많음")
        return
    fig = plt.figure()
    for i in range(3):
        fig.add_subplot(row, col, i + 1)
        # x, y = zip(*gt)
        # plt.scatter(x, y)
        plt.imshow(img[:, :, i])
    plt.show()


def train_fit_save(X, Y, seq=True):
    lr = 0.8
    if seq:
        model = Model_tf.my_test_cnn_model_Sequential(X.shape, Y.shape, lr=lr)
    else:
        model = Model_tf.my_test_cnn_model_functional(X.shape, Y.shape, lr=lr)

    with tf.device("/device:GPU:0"):
        history = model.fit(X, Y, epochs=MY_EPOCH, batch_size=MY_BATCHSIZE, verbose=1)

    # loss, acc = model.evaluate(X, Y)
    # print(f"loss={loss}, acc={acc}")

    if seq:
        filename = "./weight/cnn_seq({0}).h5".format(MY_EPOCH)
        model.save(filename)
    else:
        filename = "./weight/cnn_weight({0})".format(MY_EPOCH)
        model.save_weights(filename)
    print(f"{filename} save complite")


def test_predict(X, Y, h5=True):
    print(f"X.shape:{X.shape}")
    if h5:
        model = tf.keras.models.load_model("./weight/cnn_seq(10).h5")
    else:
        model = Model_tf.my_test_cnn_model_functional(X.shape, Y.shape)
        model.load_weights("./weight/cnn_weight(10)")

    pred = model.predict(X)
    for img, p in zip(X[::len(X) // 10], pred[::len(pred) // 10]):
        draw_plot(img, p)


def make_X_Y(path=None):
    if path is None:
        print("경로 설정 해주세요")
        return None

    print("img reading")
    find_img_name = glob(path + "*.bmp")
    img_name_list = [name[name.rfind("\\"):] for name in glob(path + "*.bmp")]  # 이미지 파일 이름 전부 가져오기
    X = np.array([plt.imread(path + name)[:, :, 0] / 255 for name in img_name_list])  # 이미지를 읽어와서 3차원으로 저장
    print(X.shape)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    print("img done")

    print("GT redding...")
    Y = []
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
        Y.append(np.array(list(map(int, line.strip().split("\t")[5:]))))  # 앞 2개 좌표 무시
        # print(cnt)
    Y = np.array(Y)
    file.close()
    print("GT done")

    return X, Y


# _X, _Y = make_X_Y("../../dataset/Train_daytime/")
# train_fit_save(_X, _Y, seq=True)
# train_fit_save(_X, _Y, seq=False)


# _X, _Y = make_X_Y("../../dataset/Test_daytime/1_test_ej/")
# test_predict(_X, _Y, h5=True)
test_predict(_X, _Y, h5=False)


# Xtext, Ytest = make_X_Y("../../dataset/Test_daytime/1_test_ej/")
# for x, y in zip(_X[:100:5], _Y[:100:5]):
#     draw_plot(x, y)


# img = np.expand_dims(img, axis=-1)

# print("reading...")
# find_img_path = glob(PATH + class_path + detail + "open eyes/*.png")
# Y = np.full((len(find_img_path), 1), 1)
# X = np.array([np.expand_dims(cv2.resize(plt.imread(path), (90, 90)), axis=-1) for path in find_img_path])
#
# find_img_path = glob(PATH + class_path + detail + "close eyes/*.png")
# # np.r_[a, b] --> 수평으로 이어 붙이기 or np.r_[[a],[b]] --> 수직으로 이어 붙이기 와 같은 형식으로 사용 가능
# Y = np.vstack([Y, np.full((len(find_img_path), 1), 0)])
# X = np.vstack([X, np.array([np.expand_dims(cv2.resize(plt.imread(path), (90, 90)), axis=-1) for path in find_img_path])])
#
# np.save("D:/Dataset/eye/X_save", X)
# np.save("D:/Dataset/eye/Y_save", Y)
# print("nparray 저장 완료")

eye = eye_Net()
X, Y = eye.load_dataset("D:/Dataset/eye/train/", "train_save")
model = eye.model()
with tf.device("/device:GPU:0"):
    model.fit(X,Y,epochs=MY_EPOCH, batch_size=MY_BATCHSIZE,verbose=1)
model.save("D:/Dataset/eye/model/keras_eye_trained_model.h5")

# X, Y = eye.load_dataset("D:/Dataset/eye/test/", "test_save")
# model = eye.eye_predictor("D:/Dataset/eye/model/cnn_eye_open_close(0.0918 0.9684).h5")
# test_loss, test_acc = model.evaluate(X, Y, verbose=0)
# print(f"loss:{test_loss}, acc:{test_acc}")
