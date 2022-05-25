from glob import glob

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# def make_model():
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), padding='same', activation='tanh', input_shape=(Spic, Spic, 1)))
#     model.add(MaxPool2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Flatten())
#     model.add(Dense(256, activation='tanh'))
#     model.add(Dropout(0.5))
#     model.add(Dense(8, activation='sigmoid'))
#
#     sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#     model.compile(loss='mean_squared_error', optimizer=sgd)
from matplotlib.patches import Rectangle


def read_text_to_np(sp=None):
    file = open("../../dataset/face_landmark_train_set/Train_daytime/GT_Points_ForTraining_daytime.txt", "r")

    full_data = {}
    while True:
        line = file.readline()
        if not line:
            break
        sp = line.split("\t")
        pos = []
        for i in range(1, len(sp)-1, 2):
            pos.append([int(sp[i]), int(sp[i+1])])
        full_data[sp[0]] = pos
    file.close()
    return full_data




df = read_text_to_np()
img_name_list = glob("../../dataset/face_landmark_train_set/Train_daytime/*.bmp")
img = plt.imshow(plt.imread(img_name_list[0]))
print(img_name_list)
x, y = zip(*df[img_name_list[0][-9:]])
plt.scatter(x, y)
plt.show()


