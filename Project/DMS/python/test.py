# http://www.ntrexgo.com/archives/36038
import random
import timeit

import cv2, dlib
import numpy as np
import time

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

# 100프레임을 미리 저장해서 눈의 비율의 최대치등을 계산
ER_array_ready = False
ER_array_ready_size = 100
left_eye_ER_array = []
right_eye_ER_array = []
MAX_ER_left = 0
MAX_ER_right = 0
eye_ratio_limit = 0.00

count_time = [0, 0]
program_switch = False

open_eye = False
eye_close_count = 0
driving_state_step = [15, 35]


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


# 얼굴이 기울었을때 인식률을 높이기 위한 detection_img의 재 정렬을 위한 좌표 계산 함수
# def rotate(brx, bry):
#     crx = brx - midx
#     cry = bry - midy
#     arx = np.cos(-angle) * crx - np.sin(-angle) * cry
#     ary = np.sin(-angle) * crx + np.cos(-angle) * cry
#     rx = int(arx + midx)
#     ry = int(ary + midy)
#     return rx, ry


# 눈 비율 계산
def calculate_EAR(eye):
    # 얼굴 특징점 번호 사진 참조
    A = distance(eye[1], eye[5])
    B = distance(eye[2], eye[4])
    C = distance(eye[0], eye[3])
    return (A + B) / (2.0 * C)


face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
print("stub loading facial landmark predictor...")
video_capture = cv2.VideoCapture(0)  # 카메라
index = NOTHING


if video_capture.isOpened():
    print("camera is ready")
    while True:
        ret, frame = video_capture.read()
        if not ret: break
        start_t = 0
        if ret is True:
            start_t = timeit.default_timer()
        # 얼굴 랜드마크 종류별로 변경
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0,
                                    tileGridSize=(8, 8))
            clahe_image = clahe.apply(gray)
            detection = face_detector(clahe_image, 0)

            # 1-1 dlib를 통해 검출된 Face landmarks좌표를 기반으로 3D좌표를 이용한 Head pose estimation을 수행
            # 1-2 향상된 방법으로는 CNN 기반의 Facial Landmkark Detection알고리즘을 사용해서 landmark까지 추출한다



            FPS = int(1./(timeit.default_timer() - start_t))
            cv2.putText(frame, f"FPS:{FPS}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, SKYBLUE, 2)
            cv2.imshow("test", frame)

cv2.destroyAllWindows()
video_capture.release()
