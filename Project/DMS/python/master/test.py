import pprint

import cv2
import dlib
import numpy as np
import imutils
import time

from my_face_detector import *
from my_mark_detector import *
from pose_estimator import PoseEstimator
from my_tracker import *
from Look_CNN import look_cnn
from Preprocessing import *
from img_draw import imgMake
from Haar_Like import Haar_like
from make_test_case import my_make_test_case
from Evaluation import evaluation
import matplotlib.pyplot as plt
from Eye import *


def flatten_array_remove_item(array, itemToPop):
    array_flat = np.ndarray.flatten(array)
    array_toPop = np.array(itemToPop)
    array_refined = np.setdiff1d(array_flat, array_toPop)
    return array_refined


def threshold(frame, quantile=0.95, maxValue=255, borderType=cv2.THRESH_BINARY_INV):
    ED = cv2.erode(frame, np.ones((9, 9), np.uint8), 2)
    gb = cv2.GaussianBlur(ED, (31, 31), 50)
    frame_values = flatten_array_remove_item(gb, 255)
    thres = np.quantile(frame_values, quantile)
    _, frame_thold = cv2.threshold(frame, thres, maxValue, borderType)
    return frame_thold


def my_threshold(img, quantile=0.4, maxValue=255, borderType=cv2.THRESH_BINARY_INV):
    EROD = cv2.erode(img, np.ones((9, 9), np.uint8), 2)
    gaussian = cv2.GaussianBlur(EROD, (31, 31), 50)
    img_values = flatten_array_remove_item(gaussian, 255)
    thres = np.quantile(img_values, quantile)
    _, img_thold = cv2.threshold(img, thres, maxValue, borderType)
    return img_thold


def search(img):
    print(img.shape)
    search_dir = [[-1, -1], [-1, 0], [-1, 1],
                  [ 0, -1],          [ 0, 1],
                  [ 1, -1], [ 1, 0], [ 1, 1]]
    visited = np.full((img.shape[0], img.shape[1]), 0)
    area_number = 0

    big_area_idx = 0
    big_area_size = 0
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if visited[row][col] == 0 and img[row][col] == 255:
                area_number += 1
                cur_area_size = 0
                queue = [[row, col]]
                while len(queue) > 0:
                    now_row = queue[0][0]
                    now_col = queue[0][1]
                    del queue[0]
                    cur_area_size += 1
                    for d in search_dir:
                        next_row = now_row + d[0]
                        next_col = now_col + d[1]
                        if next_row < 0 or next_row >= img.shape[0] or next_col < 0 or next_col >= img.shape[1]:    continue
                        if visited[next_row][next_col] > 0 or img[next_row][next_col] == 0:
                            continue
                        queue.append([next_row, next_col])
                        visited[next_row][next_col] = area_number
                if big_area_size < cur_area_size:
                    big_area_idx = area_number
                    big_area_size = cur_area_size
    result = np.full((img.shape[0], img.shape[1]), 0)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if visited[row][col] == big_area_idx:
                result[row][col] = 255
    # result = np.expand_dims(result, axis=-1)
    result = np.array(result, dtype=np.uint8)
    return result



print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
SKYBLUE = (255, 255, 0)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

PIXELS = [(1280, 720), (640, 480), (256, 144), (320, 240), (480, 360)]
PIXEL_NUMBER = 0
RES_W, RES_H = PIXELS[PIXEL_NUMBER][0], PIXELS[PIXEL_NUMBER][1]

FaceDetector = FaceDetector()  # 얼굴 인식 관련
MarkDetector = MarkDetector(save_model="../assets/shape_predictor_68_face_landmarks.dat")  # 랜드마크 관련
# cv2.matchTemplate()도 해보자
Tracker = Tracker()  # 트래킹 관련
haar_like = Haar_like()
iMake = imgMake()
mtc = my_make_test_case()
eva = evaluation()
HCBC = HaarCascadeBlobCapture()
total_frame = 0
hit = 0
acc = 0
Y = "center"

pred_dir = [0, 0, 0]
crop = None
cap = None
try:
    # 카메라 or 영상
    path_name = "D:/Dataset/test_video.mp4"
    num = path_name[path_name.rfind("/") - 1]
    if num == "1" or num == "4":
        Y = "right"
    elif num == "2" or num == "5":
        Y = "center"
    else:
        Y = "left"
    print(Y)
    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("D:/mystudy/pholythec/Project/DMS/WIN_20220520_16_13_04_Pro.mp4")
    # cap = cv2.VideoCapture("D:/JEON/Polytechnics/Project/DMS/dataset/WIN_20220526_15_33_19_Pro.mp4")
    # cap = cv2.VideoCapture("D:/JEON/dataset/look_direction/vid/4/03-4.mp4")
    cap = cv2.VideoCapture(path_name)


except:
    print("Error opening video stream or file")
while cap.isOpened():
    perv_time = time.time()
    ret, frame = cap.read(1)
    if ret:
        total_frame += 1
        frame = np.array(imutils.resize(frame, width=RES_W, height=RES_H))  # imutils cv2 경량화 보조 패키지
        # frame = img_Preprocessing_v3(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # (H, W)
        if gray is None:
            break

        if Tracker.track_number == 0:
            Tracker.find_tracker(gray, FaceDetector)
        else:  # 트레킹 할 얼굴이 1명 이상 있다면(지금은 1명만 트레킹 하도록 작성함)
            box_rect = None
            if Tracker.frame_counter == 60:  # 60 프레임마다 한번씩 트래커 이탈 방지용 refaceDetection
                box_rect = Tracker.find_tracker(gray, FaceDetector, re=True)
            else:
                box_rect = Tracker.tracking(gray)  # 네모 박스 처준다 원한다면 rectangle타입 반환도 가능

            if box_rect is not None:
                landmarks = MarkDetector.get_marks(gray, box_rect)
                if landmarks is not None:
                    landmarks_ndarray = MarkDetector.full_object_detection_to_ndarray(landmarks)


                    eyes = [eye_crop_none_border(gray, landmarks_ndarray[36:42]),
                            eye_crop_none_border(gray, landmarks_ndarray[42:48])]
                    for i in range(2):
                        for _ in range(2):
                            eyes[i] = cv2.pyrUp(eyes[i])
                        key_points = HCBC.blob_track(eyes[i], HCBC.previous_blob_area[i])
                        kp = key_points or HCBC.previous_keypoints[i]
                        # if kp is not None:
                            # print(f"{i}: x:{kp[0].pt[0]} y:{kp[0].pt[1]}")
                        eyes[i] = HCBC.draw(eyes[i], kp, frame)
                        # cv2.imshow(f"eyes[{i}]", eyes[i])
                        HCBC.previous_keypoints[i] = kp
                    cv2.imshow("left", eyes[0])
                    cv2.imshow("right", eyes[1])

                    # check = HCBC.eye_direction_process(gray, landmarks_ndarray)
                    # cv2.putText(frame, f"{check}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)


                    MarkDetector.draw_marks(frame, landmarks, color=GREEN)  # 랜드마크 점 그려주기

                    # 랜드마크 type=full_object_detection --> .part().x, .part().x 형식으로 뽑아내기
                    # landmarks = MarkDetector.full_object_detection_to_ndarray(landmarks)

        cv2.putText(frame, f"fps:{int(1. / (time.time() - perv_time))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED,
                    2)
        cv2.imshow('show_frame', frame)
        # cv2.imshow("frame_preprocessing", frame)
    else:
        break
    # print(total_frame)
    # if the `esc` key was pressed, break from the loop
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
cap.release()
eva.total_frame = total_frame
eva.measurement_result()
