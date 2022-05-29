import cv2
from functools import wraps
import time

import numpy as np

lastsave = 0


# (두 점 사이의 유클리드 거리 계산)
def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1 / 2)


def rotate(brx, bry, midx, midy, angle):
    crx = brx - midx
    cry = bry - midy
    arx = np.cos(-angle) * crx - np.sin(-angle) * cry
    ary = np.sin(-angle) * crx + np.cos(-angle) * cry
    rx = int(arx + midx)
    ry = int(ary + midy)

    return [rx, ry]


def get_rotate_border_box(img, box, mark):
    x = box.left()
    y = box.top()
    x1 = box.right()
    y1 = box.bottom()
    bdx = x - (x1 - x) / 2
    bdy = y - (y1 - y) / 2
    bdx1 = x1 + (x1 - x) / 2
    bdy1 = y1 + (y1 - y) / 2
    midx = (x + x1) / 2
    midy = (y + y1) / 2


    rex = mark[45][0]
    rey = mark[45][1]
    lex = mark[36][0]
    ley = mark[36][1]

    mex = int(lex + (rex - lex) / 2)
    mey = int(ley + (rey - ley) / 2)

    tanx = mex - lex
    tany = ley - mey
    tan = tany / tanx
    angle = np.arctan(tan)

    rsd_1 = rotate(x, y,midx, midy,angle)
    rsd_2 = rotate(x1, y,midx, midy,angle)
    rsd_3 = rotate(x, y1,midx, midy,angle)
    rsd_4 = rotate(x1, y1,midx, midy,angle)
    d2_1 = rotate(bdx, bdy,midx, midy,angle)
    d2_2 = rotate(bdx1, bdy,midx, midy,angle)
    d2_3 = rotate(bdx, bdy1,midx, midy,angle)
    d2_4 = rotate(bdx1, bdy1,midx, midy,angle)

    pts1 = np.float32([[d2_1[0], d2_1[1]], [d2_2[0], d2_2[1]], [d2_3[0], d2_3[1]], [d2_4[0], d2_4[1]]])
    pts2 = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    return cv2.warpPerspective(img, M, (400, 400))


def prespective_Transform(rotate_border_box, img, mulrate=1):
    width = int(mulrate * distance([rotate_border_box[0][0], rotate_border_box[0][1]], [rotate_border_box[1][0],rotate_border_box[0][1]]))
    hight = int(mulrate * distance([rotate_border_box[0][0], rotate_border_box[0][1]], [rotate_border_box[0][0],rotate_border_box[1][1]]))
    # print(width, hight)

    pts1 = np.float32([[rotate_border_box[0][0], rotate_border_box[0][1]],
                       [rotate_border_box[1][0], rotate_border_box[0][1]],
                       [rotate_border_box[0][0], rotate_border_box[1][1]],
                       [rotate_border_box[1][0], rotate_border_box[1][1]]])

    pts2 = np.float32([[0, 0], [width, 0], [0, hight], [width, hight]])
    # pts2 = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    return cv2.warpPerspective(img, M, (width, hight))
    # return cv2.warpPerspective(img, M, (400, 400))


# https://wjh2307.tistory.com/21
# 눈 감기(함수) + 자는거 판단(부정확)
def close_counter(func):
    @wraps(func)
    def tmp(*args, **kwargs):
        tmp.count += 1
        global lastsave
        # print(f"during close: {time.time() - lastsave}")
        if time.time() - lastsave > 5:
            lastsave = time.time()
            tmp.count = 0
        return func(*args, **kwargs)

    tmp.count = 0
    return tmp


class eye_calculation:
    def __init__(self):
        """Initialization"""
        self.both_ER_ratio_avg = 0

    # 눈 비율 값 계산
    def ER_ratio(self, eye_point):
        # 얼굴 특징점 번호 사진 참조
        A = distance(eye_point[1], eye_point[5])
        B = distance(eye_point[2], eye_point[4])
        C = distance(eye_point[0], eye_point[3])
        return (A + B) / (2.0 * C)

    def eye_close(self, left_eye, right_eye):
        left_ER = self.ER_ratio(left_eye)
        right_ER = self.ER_ratio(right_eye)

        avg = round((left_ER + right_ER) / 2, 2)
        self.both_ER_ratio_avg = avg
        return avg < 0.2

    def draw_eye_close_ratio(self, img, color=(0, 255, 0)):
        cv2.putText(img, f"{self.both_ER_ratio_avg}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    @close_counter
    def close(self, img, color=(0, 0, 255)):
        cv2.putText(img, "close", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
