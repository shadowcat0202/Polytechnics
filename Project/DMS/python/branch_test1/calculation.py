import cv2
from functools import wraps
import time

import numpy as np

lastsave = 0


# (두 점 사이의 유클리드 거리 계산)
def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1 / 2)


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
        self.clahe = cv2.createCLAHE(clipLimit=2.0,
                                     tileGridSize=(8, 8))

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
        return avg < 0.3

    def draw_eye_close_ratio(self, img, color=(0, 255, 0)):
        cv2.putText(img, f"{self.both_ER_ratio_avg}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def img_Preprocessing(self, img_frame):
        """
        cv2.resize((fx,fy),interpilation )
        1. cv2.INTER_NEAREST - 최근방 이웃 보간법
         가장 빠르지만 퀄리티가 많이 떨어집니다. 따라서 잘 쓰이지 않습니다.
        2. cv2.INTER_LINEAR - 양선형 보간법(2x2 이웃 픽셀 참조)
         4개의 픽셀을 이용합니다.
         효율성이 가장 좋습니다. 속도도 빠르고 퀄리티도 적당합니다.

        3. cv2.INTER_CUBIC - 3차회선 보간법(4x4 이웃 픽셀 참조)
         16개의 픽셀을 이용합니다.
         cv2.INTER_LINEAR 보다 느리지만 퀄리티는 더 좋습니다.
        4. cv2.INTER_LANCZOS4 - Lanczos 보간법 (8x8 이웃 픽셀 참조)
         64개의 픽셀을 이용합니다.
         좀더 복잡해서 오래 걸리지만 퀄리티는 좋습니다.
        5. cv2.INTER_AREA - 영상 축소시 효과적
         영역적인 정보를 추출해서 결과 영상을 셋팅합니다.
         영상을 축소할 때 이용합니다."""
        re_size = cv2.resize(img_frame, (400, 400), cv2.INTER_AREA)

        gray = cv2.cvtColor(re_size, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(re_size, cv2.COLOR_BGR2LAB)

        # # CLAHE
        gray = self.clahe.apply(gray)
        # lab = self.clahe.apply(lab)

        # # 빛 제거? 무슨 원리인지는 모르겠다
        L = lab[:, :, 0]
        med_L = cv2.medianBlur(L, 99)  # median filter  # 뭔지 모르겠음
        invert_L = cv2.bitwise_not(med_L)  # invert lightness   # 빛 제거?
        composed = cv2.addWeighted(gray, 0.75, invert_L, 0.25, 0)

        # # Histograms Equalization
        # equ = cv2.equalizeHist(gray)  # 무조건 색상이 1차원이여야 한다 = gray
        # result = np.hstack((gray, equ))

        cv2.imshow("Histograms Equalization", composed)
        return re_size, composed

    @close_counter
    def close(self, img, color=(0, 0, 255)):
        cv2.putText(img, "close", (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
