import cv2
import numpy as np
from branch_test1.img_draw import *


def up(img):
    re_size = cv2.pyrUp(img)

    filter_size = (200, 200)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, filter_size)
    topHat = cv2.morphologyEx(re_size, cv2.MORPH_TOPHAT, k)

    # HE: It is a method that improves the contrast in an image, in order to stretch out the intensity range (see also the corresponding Wikipedia entry).
    gray = cv2.cvtColor(topHat, cv2.COLOR_BGR2GRAY)

    eqhist = cv2.equalizeHist(gray)

    return re_size, topHat, gray, eqhist


def down(img):
    re_size = cv2.pyrDown(img)
    cv2.imshow("down", re_size)
    filter_size = (30, 30)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
    topHat = cv2.morphologyEx(re_size, cv2.MORPH_TOPHAT, k)

    # HE: It is a method that improves the contrast in an image, in order to stretch out the intensity range (see also the corresponding Wikipedia entry).
    gray = cv2.cvtColor(topHat, cv2.COLOR_BGR2GRAY)
    result = cv2.equalizeHist(gray)
    return re_size, result


# video_capture = cv2.VideoCapture("../test_face.png")  # 카메라
# src = cv2.imread("D:/mystudy/pholythec/Project/DMS/face_sample/face_sample3.png")
src = cv2.imread("../dark_face.png")
src = cv2.resize(src, (250, 250))
filterSize = (150, 150)

rows, cols, _channels = map(int, src.shape)
# print(f"{rows}, {cols}, {_channels}")
while True:
    m1, m2, m3, m4 = up(src)

    rows = max(m1.shape[0], m2.shape[0], m3.shape[0], m4.shape[0])
    cols = max(m1.shape[1], m2.shape[1], m3.shape[1], m4.shape[0])
    _channels = max(m1.shape[2], m2.shape[2])
    IM = imgMake()

    dstimage = IM.create_image_multiple(rows, cols, _channels, 2, 2)
    IM.showMultiImage(dstimage, m1, rows, cols, m1.shape[2], 0, 0)
    IM.showMultiImage(dstimage, m2, rows, cols, m2.shape[2], 0, 1)
    IM.showMultiImage(dstimage, m3, rows, cols, 1 if len(m3[0])*len(m3[1]) == rows * cols else m3.shape[2], 1, 0)
    IM.showMultiImage(dstimage, m4, rows, cols, 1 if len(m4[0])*len(m4[1]) == rows * cols else m3.shape[2], 1, 1)

    cv2.imshow("original, topHat, gray, eqhist", dstimage)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
