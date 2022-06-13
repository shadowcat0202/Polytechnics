import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import load_model


BLUE = [255, 0, 0]
GREEN = [0, 255, 0]
RED = [0, 0, 255]
def d20220613():
    src = cv2.imread("Lenna.png")
    face = src[240:380, 217:350].copy()

    b = src[:, :, 0]
    g = src[:, :, 1]
    r = src[:, :, 2]
    face_rgb = np.hstack([b, g, r])

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # 0 dim rgb 평균
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)  # 3 dim
    cv2.imshow("gray", gray)
    cv2.imshow("hsv", hsv)
    v = hsv[:, :, 2]
    cv2.imshow("hsv-v", v)

    # face_rgb = np.hstack([gray, hsv])
    # cv2.imshow("[r,g,b]", face_rgb)

    cv2.waitKey(0)
    cv2.destroyWindow("rgb")

def practice2():
    filename = "./Python/model/cnn_e(10).h5"
    model = load_model(filename)

    src = cv2.imread("hand_write_number.png")
    row_size = 100
    col_size = 100

    img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # _, img = cv2.threshold(img, 200, 255, cv2.THRESH_TRUNC)
    _, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.erode(img, None, iterations=2)

    cv2.imshow("img", img)
    # plt.imshow(img)
    # plt.show()
    # zero = img[59:139, 52:142]
    # zero = cv2.resize(zero, (28, 28))  # (28, 28)
    # zero = np.expand_dims(zero, axis=-1)  # (28, 28, 1)
    # zero = np.expand_dims(zero, axis=0)  # (1, 28, 28, 1)
    #
    # six = img[21:125, 729:800]
    # six = cv2.resize(six, (28, 28))  # (28, 28)
    # six = np.expand_dims(six, axis=-1)  # (28, 28, 1)
    # six = np.expand_dims(six, axis=0)  # (1, 28, 28, 1)
    #
    # one = img[47:134, 168:227]
    # one = cv2.resize(one, (28, 28))  # (28, 28)
    # one = np.expand_dims(one, axis=-1)  # (28, 28, 1)
    # one = np.expand_dims(one, axis=0)  # (1, 28, 28, 1)
    #
    # pal = img[25:118, 994:1029]
    # pal = cv2.resize(pal, (28, 28))  # (28, 28)
    # pal = np.expand_dims(pal, axis=-1)  # (28, 28, 1)
    # pal = np.expand_dims(pal, axis=0)  # (1, 28, 28, 1)
    #
    # zero = model.predict(zero)
    # six = model.predict(six)
    # one = model.predict(one)
    # pal = model.predict(pal)
    #
    #
    # print(zero[0].argmax())
    # print(six[0].argmax())
    # print(one[0].argmax())
    # print(pal[0].argmax())

    for row in range(30, img.shape[0], 150):
        for col in range(0,img.shape[1], 5):
            roi = img[row:row+row_size, col:col+col_size]
            # cv2.imshow("crop", roi)
            view = roi.copy()

            roi = cv2.resize(roi, (28,28))  # (28, 28)
            cv2.imshow("resize", roi)
            roi = np.expand_dims(roi, axis=-1)  # (28, 28, 1)
            roi = np.expand_dims(roi, axis= 0)  # (1, 28, 28, 1)

            pred = model.predict(roi)
            idx = None
            # idx = pred[0].argmax()


            # idx = pred.index(max(pred))
            # print(pred)

            for i, p in enumerate(pred[0]):
                print(round(p*100, 3), end=", ")
                if p > 0.95:
                    idx = i
            print()
            cv2.putText(view, f"{idx}", (10, 20), 1, 1, (0, 0, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)
            cv2.imshow("view", view)

            key = cv2.waitKey(1)
            if key == ord("q"):
                cv2.destroyAllWindows()
                exit(0)
    cv2.destroyAllWindows()
    exit(0)

if __name__ == '__main__':
    practice2()


