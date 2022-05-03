import tensorflow as tf
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt


def load_dataset(online=False):
    if online:
        (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
    else:
        path = "D:/JEON/Polytechnics/기계학습프로그래밍/Code/Python/dataset/mnist.npz"
        (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data(path)
    return (X_train, Y_train), (X_test, Y_test)


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


def make_model(x, y):
    model = Sequential()
    model.add(Dense(700, input_dim=len(x), activation="relu"))
    # model.add(Dense(60, input_dim=700, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    model.summary()
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model


# 숙제 score할때 내부적으로 왜 틀렸는가 ex) 내 예측은 a인데 정답은 b이다를 보고 싶다
def my_score(x, y):
    res = []
    model = joblib.load("./model/rf-mnist.plk")
    pred = model.predict(x)
    cnt = 0
    print("예축값 : 결과값")
    for i in range(len(pred)):
        if pred[i] != y[i]:
            res.append([i, pred[i], y[i]])
            cnt += 1
    print(f"오답 {cnt}개")
    return res, pred


(train_set, train_label), (test_set, test_label) = load_dataset()

# show_data_values(train_label, test_label)

# 학습이나 시키자
# train_set.reshape(len(train_set), 784)와 같은 역할
# 2차원 -> 1차원 리스트로 내려서 학습데이터로 대입
RF_train(train_set.reshape(len(train_set), train_set.shape[1] * train_set.shape[2]), train_label)

# clf = joblib.load("./model/rf-mnist.plk")
# train_set = train_set.reshape(len(train_set), train_set.shape[1] * train_set.shape[2])
# result = clf.predict(train_set[:20])
# print(result)
# print(train_label[:20])
#
# print(test_set.shape)


# print("%.8f" % clf.score(test_set, test_label))

# 틀린거 보여주기==========================================================================
# output_res, pred = my_score(test_set.reshape(len(test_set), test_set.shape[1] * test_set.shape[2]), test_label)
# userinput = ""
# print("next=Enter, exit=q:")
# for res in output_res:
#     print(f"예측:{res[1]} 결과:{res[2]} ", end="")
#     show_image(test_set[res[0]])
#     userinput = input("next?:")
#     if userinput == "q":
#         break


# confusion_matrix 그래프 출력==========================================================================
# from sklearn.metrics import confusion_matrix, plot_confusion_matrix
# 
# label = [i for i in range(10)]
# plot = plot_confusion_matrix(joblib.load("./model/rf-mnist.plk"),
#                       test_set.reshape(len(test_set), test_set.shape[1] * test_set.shape[2]),
#                       test_label,
#                       display_labels=label, normalize="true")
# plot.ax_.set_title("Confusion Matrix")
# plt.show()


