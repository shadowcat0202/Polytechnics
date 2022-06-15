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
    def crop_img(img, pos):
        crop = img[pos[0][0]:pos[1][0], pos[0][1]:pos[1][1]]
        crop = cv2.resize(crop, (28, 28))  # (28, 28)
        crop = np.expand_dims(crop, axis=-1)  # (28, 28, 1)
        show_crop = crop.copy()
        crop = np.expand_dims(crop, axis=0)  # (1, 28, 28, 1)
        return crop, show_crop

    filename = "./Python/model/cnn_e(5).h5"
    model = load_model(filename)
    # (row, col)
    number_pos = [[(54, 47), (139, 139)], [(37, 156), (134, 222)],
                  [(52, 269), (134, 350)], [(33, 381), (140, 437)],
                  [(35, 478), (137, 558)], [(37, 610), (134, 676)],
                  [(25, 729), (127, 785)], [(28, 840), (158, 908)],
                  [(28, 949), (120, 1024)], [(28, 1067), (134, 1126)]]

    src = cv2.imread("hand_write_number.png")
    img = src.copy()
    row_size = 100
    col_size = 100

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 200, 255, cv2.THRESH_TRUNC)
    # cv2.imshow("트렁크", img)
    _, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow("BINARY + OTUS", img)
    img = cv2.erode(img, None, iterations=2)
    # cv2.imshow("erode", img)
    # hst = []
    # pred_y = []
    # for p in number_pos:
    #     x, i = crop_img(img, p)
    #     hst.append(i)
    #     pred_y.append(model.predict(x)[0].argmax())
    # hst = np.hstack(hst)
    #
    # # cv2.imshow("img", img)
    # print(pred_y)
    # plt.imshow(hst)
    # plt.show()
    # =========================================================================
    for row in range(30, img.shape[0], 150):
        for col in range(0, img.shape[1], 5):
            roi = img[row:row + row_size, col:col + col_size]
            # cv2.imshow("crop", roi)
            view = roi.copy()

            roi = cv2.resize(roi, (28, 28))  # (28, 28)
            cv2.imshow("resize", roi)
            roi = np.expand_dims(roi, axis=-1)  # (28, 28, 1)
            roi = np.expand_dims(roi, axis=0)  # (1, 28, 28, 1)

            pred = model.predict(roi)
            idx = None
            # idx = pred[0].argmax()

            # idx = pred.index(max(pred))
            # print(pred)

            for i, p in enumerate(pred[0]):
                print(round(p * 100, 3), end=", ")
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


def practice3():
    def crop_border(_img, _x, _y, _w, _h):
        center = (_x + _w // 2, _y + _h // 2)
        if _w < 28 and _h < 28:
            _x = center[0] - (14 + 6)
            _y = center[1] - (14 + 6)
            _w = _h = 40
        elif _w < 28 and not _h < 28:
            _x = center[0] - (_h // 2 + 6)
            _y = center[1] - (_h // 2 + 6)
            _h += 12
            _w = _h
        elif not _w < 28 and _h < 28:
            _x = center[0] - (_h // 2 + 6)
            _y = center[1] - (_w // 2 + 6)
            _w += 12
            _h = _w
        else:
            bigger = max(_w, _h)
            _x = center[0] - (bigger // 2 + 12)
            _y = center[1] - (bigger // 2 + 12)
            _w = bigger + 24
            _h = bigger + 24

        return _img[_y:_y + _h, _x:_x + _w], (_x, _y, _w, _h)

    filename = "./Python/model/cnn_e(5).h5"
    model = load_model(filename)

    src = cv2.imread("hand_write_number.png")
    view = src.copy()
    img = src.copy()

    img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # (a,b)
    # cv2.imshow("gray", img)

    # _, img = cv2.threshold(img, 50, 255, cv2.THRESH_TOZERO)  # (a,b)
    # _, img = cv2.threshold(img, 230, 255, cv2.THRESH_TRUNC)  # (a,b)
    # cv2.imshow("trunc", img)

    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # (a,b)
    # _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)  # (a,b)
    # cv2.imshow("binary+otsu", img)

    img = cv2.dilate(img, None, iterations=3)
    # plt.imshow(img)
    # plt.show()

    n_blod, label_img, stats, centroids = cv2.connectedComponentsWithStats(img)

    hst = []
    pred_y = []
    for i in range(1, n_blod):
        x, y, w, h, area = stats[i]

        roi, pos = crop_border(img, x, y, w, h)
        roi = 255 - roi
        roi = cv2.resize(roi, (28, 28))  # (28, 28)

        hst.append(roi)
        roi = np.expand_dims(roi, axis=-1)  # (28, 28, 1)
        roi = np.expand_dims(roi, axis=0)  # (1, 28, 28, 1)
        pred_y.append(model.predict(roi)[0].argmax())

        # cv2.putText(view, f"{pred_y[-1]}", (pos[0], pos[1]), 1, 2, (255, 0, 255), cv2.LINE_4)  # border pos
        # cv2.rectangle(view, pos, (255, 0, 255), thickness=2)
        cv2.putText(view, f"{pred_y[-1]}", (x, y), 1, 2, (255, 0, 255), cv2.LINE_4) # original pos
        cv2.rectangle(view, (x,y,w,h), (255, 0, 255), thickness=2)

    # print(pred_y)
    hst = np.hstack(hst)
    cv2.imshow("hst", hst)
    cv2.imshow("view", view)

    while True:
        key = cv2.waitKey(0)
        if key == ord("q"):
            cv2.destroyAllWindows()
            exit(0)


if __name__ == '__main__':
    practice3()
