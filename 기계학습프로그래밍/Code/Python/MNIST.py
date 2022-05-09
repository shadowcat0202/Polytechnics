import os

import tensorflow as tf
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import numpy
import seaborn as sns

MY_EPOCH = 30
MY_BATCHSIZE = 400

def load_dataset(online=False):
    if online:
        (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
    else:
        path = "D:/JEON/Polytechnics/기계학습프로그래밍/Code/Python/dataset/mnist.npz"
        (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data(path)
    return (X_train, Y_train), (X_test, Y_test)


(train_set, train_label), (test_set, test_label) = load_dataset()
# 2차원 배열을 1차원 배열로 내린다
# train_set_1d = train_set.reshape(len(train_set), train_set.shape[1] * train_set.shape[2]).astype('float32') / 255  # (60000, 784)
# test_set_1d = test_set.reshape(len(test_set), test_set.shape[1] * test_set.shape[2]).astype('float32') / 255  # (100000, 784)
train_set_1d = train_set.reshape(len(train_set), train_set.shape[1] * train_set.shape[2])
test_set_1d = test_set.reshape(len(test_set), test_set.shape[1] * test_set.shape[2])
# train_set_1d = train_set.reshape(len(train_set), train_set.shape[1] * train_set.shape[2])  # (60000, 784)
# test_set_1d = test_set.reshape(len(test_set), test_set.shape[1] * test_set.shape[2])  # (100000, 784)

def show_data_values(*args):
    for i in range(len(args)):
        plt.subplot(1, 2, i + 1)
        count_value = np.bincount(args[i])
        print(count_value)
        plt.bar(np.arange(0, 10), count_value)
        plt.xticks(np.arange(0, 10))
        plt.grid()
    plt.show()


def show_image(img):
    plt.imshow(255 - img, cmap="gray")
    plt.show()


def RF_train(x, y):
    clf = RandomForestClassifier()
    clf.fit(x, y)
    joblib.dump(clf, "./model/rf-mnist.plk")  # pickle 파일로 저장 -> 파이썬 객체를 바이너리로 저장


# def make_model():
#     model = Sequential()
#     model.add(Dense(500, input_dim=784, activation="relu"))
#     model.add(Dense(300, input_dim=500, activation="relu"))
#     model.add(Dense(10, activation="softmax"))
#     model.summary()
#     model.compile(loss="categorical_crossentropy",
#                   optimizer="adam",
#                   metrics=["accuracy"])
#     return model

def make_model():
    model = Sequential()
    model.add(Dense(200, input_dim=784, activation="relu"))
    model.add(Dense(200, input_dim=200, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    model.summary()
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model

def train(m, x, y, i):
    history = m.fit(x, y, epochs=i, batch_size=MY_BATCHSIZE)
    m.save("./model/mlp_hd512_e{0}.h5".format(i))
    return history


# 숙제 score할때 내부적으로 왜 틀렸는가 ex) 내 예측은 a인데 정답은 b이다를 보고 싶다
def wrong_predict_img():
    model = joblib.load("./model/rf-mnist.plk")
    pred = model.predict(test_set_1d)
    cnt = 0
    print("예축값 : 결과값")
    for i in range(len(pred)):
        if pred[i] != test_label[i]:
            print(f"{pred[i]}, {test_label[i]}")
            show_image(test_set[i])


def confusion_Matrix():
    from sklearn.metrics import confusion_matrix, plot_confusion_matrix
    label = [i for i in range(10)]
    plot = plot_confusion_matrix(joblib.load("./model/rf-mnist.plk"),
                                 test_set.reshape(len(test_set), test_set.shape[1] * test_set.shape[2]),
                                 test_label,
                                 display_labels=label,
                                 cmap=plt.cm.Blues,
                                 normalize="true")
    plot.ax_.set_title("Confusion Matrix")
    plt.show()


def heat_map_2d(modl):
    pred = modl.predict(test_set_1d)

    arr2d = [[0 for _ in range(10)] for _ in range(10)]

    for (i, j) in zip(pred, test_label):
        arr2d[i][j] += 1

    for i, l in enumerate(arr2d):
        total = sum(l)
        for j, value in enumerate(l):
            arr2d[i][j] = round((arr2d[i][j] / total * 100), 2)

    sns.heatmap(arr2d, annot=True, cmap="Blues")
    plt.show()


# show_data_values(train_label, test_label)

# 학습이나 시키자
# train_set.reshape(len(train_set), 784)와 같은 역할
# 2차원 -> 1차원 리스트로 내려서 학습데이터로 대입
# RF_train(train_set.reshape(len(train_set), train_set.shape[1] * train_set.shape[2]), train_label)


# clf = joblib.load("./model/rf-mnist.plk")
# train_set = train_set.reshape(len(train_set), train_set.shape[1] * train_set.shape[2])
# result = clf.predict(train_set[:20])
# print(result)
# print(train_label[:20])
#
# print(test_set.shape)


# print("%.8f" % clf.score(test_set, test_label))

# ======================================================================================================
clf = joblib.load("./model/rf-mnist.plk")
# print(clf.score(train_set_1d, train_label))
# print(clf.score(test_set_1d, test_label))

# 틀린거 보여주기 + 이미지==============================================================================
# wrong_predict_img()

# 히트맵 ========================================================================
# confusion_Matrix()
heat_map_2d(clf)


# MLP


# train_label_one_hot = tf.keras.utils.to_categorical(train_label, 10)
# test_label_one_hot = tf.keras.utils.to_categorical(test_label, 10)
# r = range(10, 11)
# mlp = make_model()
# for i in r:
#     train(mlp, train_set_1d, train_label_one_hot, i)
#

# train_his = []
# test_his = []
# good = [0, 0]
# for i in r:
#     mlp = load_model("./model/mlp_hd512_e{0}.h5".format(i))
#     # 정확도만 저장
#     # train_his.append(mlp.evaluate(train_set_1d, train_label_one_hot)[1])
#     test_his.append(mlp.evaluate(test_set_1d, test_label_one_hot)[1])
#     if test_his[-1] > good[1]:
#         good[0] = i
#         good[1] = test_his[-1]
#     del mlp

# mlp = load_model("./model/mlp_hd512_e{best}.h5")
# 정확도만 저장
# train_his.append(mlp.evaluate(train_set_1d, train_label_one_hot)[1])
# test_his.append(mlp.evaluate(test_set_1d, test_label_one_hot)[1])


# x_len = numpy.arange(1, 41)
# plt.xlabel("EPOCH")
# plt.ylabel("ACC")
# plt.plot(x_len, train_his, color="red")
# plt.plot(x_len, test_his, color="blue")
# plt.show()
# print(f"가장 좋은 Epoch:{good[0]}, 정확도:{good[1]*100}")
