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


def eye_crop(img, eye_landmark):
    """
    :argument
        img: 한 프레임(이미지)
        eye_landmark: 눈의 랜드마크 좌표
    :return
        result: img에서 눈 영역의 이미지를 잘라서 반환
        border: 잘라낸 눈의 영역의 [(p1.x, p1.y), (p2.x, p2.y)]
    """
    W = [i[0] for i in eye_landmark]
    H = [i[1] for i in eye_landmark]

    W_min, W_max = min(W), max(W)
    H_min, H_max = min(H), max(H)

    W_btw = W_max - W_min
    H_btw = H_max - H_min

    W_border_size = W_btw * 0.2
    H_border_size = H_btw * 0.5

    p1_W_border = int(W_min - W_border_size)
    p1_H_border = int(H_min - H_border_size)
    p2_W_border = int(W_max + W_border_size)
    p2_H_border = int(H_max + H_border_size)

    border = [
        (p1_W_border, p1_H_border),
        (p2_W_border, p2_H_border)
    ]
    result = np.array(img[p1_H_border:p2_H_border, p1_W_border:p2_W_border])
    return result, border


def eye_crop_none_border(img, eye_landmark):
    """
    :argument
        img: 한 프레임(이미지)
        eye_landmark: 눈의 랜드마크 좌표
    :return
        result: img에서 눈 영역의 이미지를 잘라서 반환
        border: 잘라낸 눈의 영역의 [(p1.x, p1.y), (p2.x, p2.y)]
    """
    W = [i[0] for i in eye_landmark]
    H = [i[1] for i in eye_landmark]

    W_min, W_max = min(W), max(W)
    H_min, H_max = min(H), max(H)

    result = img[H_min:H_max, W_min:W_max]
    result = np.expand_dims(result, axis=-1)
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
PIXEL_NUMBER = 1
RES_W, RES_H = PIXELS[PIXEL_NUMBER][0], PIXELS[PIXEL_NUMBER][1]

FaceDetector = FaceDetector()  # 얼굴 인식 관련
MarkDetector = MarkDetector(save_model="../assets/shape_predictor_68_face_landmarks.dat")  # 랜드마크 관련
# cv2.matchTemplate()도 해보자
Tracker = Tracker()  # 트래킹 관련
haar_like = Haar_like()
iMake = imgMake()
eye = Eye

crop = None
cap = None
try:
    # 카메라 or 영상
    cap = cv2.VideoCapture(1)
    # cap = cv2.VideoCapture("D:/JEON/dataset/haar-like/WIN_20220601_15_48_25_Pro.mp4")
    # cap = cv2.VideoCapture("D:/JEON/Polytechnics/Project/DMS/dataset/WIN_20220526_15_33_19_Pro.mp4")
    # cap = cv2.VideoCapture("D:/JEON/dataset/look_direction/vid/5/04-5.mp4")


except:
    print("Error opening video stream or file")
while cap.isOpened():
    perv_time = time.time()
    ret, frame = cap.read()
    if ret:
        frame = np.array(imutils.resize(frame, width=RES_W, height=RES_H))  # imutils cv2 경량화 보조 패키지
        view = frame
        # frame = img_Preprocessing_v3(frame)
        frame = img_gray_Preprocessing(frame)

        if frame is None:   break

        if Tracker.track_number == 0:
            Tracker.find_tracker(frame, FaceDetector)
        else:  # 트레킹 할 얼굴이 1명 이상 있다면(지금은 1명만 트레킹 하도록 작성함)
            box_rect = None
            if Tracker.frame_counter == 60:  # 60 프레임마다 한번씩 트래커 이탈 방지용 refaceDetection
                box_rect = Tracker.find_tracker(frame, FaceDetector, re=True)
            else:
                box_rect = Tracker.tracking(frame)  # 네모 박스 처준다 원한다면 rectangle타입 반환도 가능

            if box_rect is not None:
                # 랜드마크 type=full_object_detection --> .part().x, .part().x 형식으로 뽑아내기
                landmarks = MarkDetector.get_marks(frame, box_rect)
                # MarkDetector.draw_marks(frame, landmarks, color=GREEN)    # 랜드마크 점 그려주기

                landmarks = MarkDetector.full_object_detection_to_ndarray(landmarks)
                if landmarks is not None or landmarks == ():
                    # # ==================== # # 랜드마크 받는 부분 =============================================================== # #
                    for i, land in enumerate([landmarks[36:42], landmarks[42:48]]):
                        cimg = eye_crop_none_border(frame, land)

                else:
                    print("랜드마크 없어서 예외처리가 나와야하지만 뭔가 이상하네요 에러 났을때 안나오네요")
            cv2.putText(view, f"fps:{int(1. / (time.time() - perv_time))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        WHITE,
                        2)
            cv2.imshow('show_frame', view)
    # if the `esc` key was pressed, break from the loop
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
cap.release()
