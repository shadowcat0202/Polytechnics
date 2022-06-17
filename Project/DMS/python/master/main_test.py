import cv2
import dlib
import numpy as np
import imutils
import time

from face_detector import * # 얼굴 detector
from mark_detector import * # 랜드마크 detector
from tracker import *       # 트래킹
from pose_estimator import PoseEstimator

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
Tracker = Tracker()  # 트래킹 관련


total_frame = 0

try:
    # 카메라 or 영상
    path_name = "D:/JEON/dataset/look_direction/vid/4/02-4.mp4"
    # cap = cv2.VideoCapture(0) # 카메라
    cap = cv2.VideoCapture(path_name)   # 영상으로
except:
    print("Error opening video stream or file")
while cap.isOpened():
    perv_time = time.time()
    ret, frame = cap.read()
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
                landmarks_ndarray = MarkDetector.full_object_detection_to_ndarray(landmarks)
                # v1 ============================================
                eyes = [eye_crop_none_border(gray, landmarks_ndarray[36:42]),
                        eye_crop_none_border(gray, landmarks_ndarray[42:48])]

                for i, eye in enumerate(eyes):
                    # eyes[i] = cv2.resize(eye, dsize=(eye.shape[1] * 3, eye.shape[0] * 3))
                    for _ in range(1):
                        eyes[i] = cv2.pyrUp(eyes[i])
                    eyes[i] = haar_like.threshold(eyes[i])

                pred = haar_like.eye_dir(eyes)
                cv2.imshow("left", eyes[0])
                cv2.imshow("right", eyes[1])
                cv2.putText(frame, f"fps:{pred}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, RED, 2)
                eva.measurement(pred)

                # eyes[0] = cv2.pyrUp(np.mean(eyes[0], axis=2).astype("uint8"))
                # eyes[0] = np.expand_dims(eyes[0], axis=-1)


                # v2 ============================================
                # left_eye = eye_crop_none_border(gray, landmarks_ndarray[36:42])
                # right_eye = eye_crop_none_border(gray, landmarks_ndarray[42:48])
                # cv2.imshow("left", left_eye)
                # cv2.imshow("right", right_eye)

                MarkDetector.draw_marks(frame, landmarks, color=GREEN)  # 랜드마크 점 그려주기

                # 랜드마크 type=full_object_detection --> .part().x, .part().x 형식으로 뽑아내기
                # landmarks = MarkDetector.full_object_detection_to_ndarray(landmarks)

        cv2.putText(frame, f"fps:{int(1. / (time.time() - perv_time))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
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