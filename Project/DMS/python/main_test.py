import pprint
import timeit

import cv2
import dlib
import numpy as np
import time
from functools import wraps
import math

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
SKYBLUE = (255, 255, 0)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INLINE = list(range(51, 68))
FACE_OUTLINE = list(range(0, 17))
NOTHING = list(range(0, 0))
INDEX = RIGHT_EYE + LEFT_EYE
print(INDEX)

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("./asset/shape_predictor_68_face_landmarks.dat")
print("stub loading facial landmark predictor...")
# video_capture = cv2.VideoCapture("./test1.mp4")  # 사진
video_capture = cv2.VideoCapture(0)  # 카메라

lastsave = 0


# (두 점 사이의 유클리드 거리 계산)
def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1 / 2)


# 눈 비율 값 계산
def ER_ratio(eye_point):
    # 얼굴 특징점 번호 사진 참조
    A = distance(eye_point[1], eye_point[5])
    B = distance(eye_point[2], eye_point[4])
    C = distance(eye_point[0], eye_point[3])
    return (A + B) / (2.0 * C)


def eye_close(left_eye, right_eye):
    left_ER = ER_ratio(left_eye)
    right_ER = ER_ratio(right_eye)

    avg = round((left_ER + right_ER) / 2, 2)
    cv2.putText(img, f"{avg}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
    return avg < 0.3


# https://wjh2307.tistory.com/21
def counter(func):
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


@counter
def close():
    cv2.putText(img, "close", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)



def yawn():


if video_capture.isOpened():
    print("camera is ready")

    while True:
        start_t = time.time()
        key = cv2.waitKey(1)
        if key == 27:
            break
        ret, img = video_capture.read()
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

        img = cv2.resize(img, (400, 400), cv2.INTER_AREA)
        img = cv2.flip(img, 1)  # cv2.flip(frame, [0 | 1]) 0 상하, 1 좌우 반전
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0,
                                tileGridSize=(8, 8))
        clahe_image = clahe.apply(gray)
        detection = face_detector(clahe_image, 0)

        if detection:
            for d in detection:  # 얼굴을 감지한것이 여러명일 경우도 있기때문에 for문으로 작성
                # print(f"{type(d)},")

                shape = shape_predictor(clahe_image, d)

                landmarks = list([p.x, p.y] for p in shape.parts())

                check = eye_close(landmarks[42:48], landmarks[36:42])
                if check:
                    close()
                    # print(f'close count : {close.count}')
                    if close.count >= 20:
                        # waring_flag += 1
                        cv2.putText(img, "SLEEPING!!!", (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

                for i in INDEX:
                    cv2.circle(img, (landmarks[i][0], landmarks[i][1]), 1, GREEN, -1)

        cv2.putText(img, f"FPS:{int(1. / (time.time() - start_t))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

        cv2.imshow("test", img)
        # cv2.imshow("gray", gray)

cv2.destroyAllWindows()
video_capture.release()
