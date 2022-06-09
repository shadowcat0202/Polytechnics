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


class HaarCascadeBlobCapture:
    def __init__(self):
        self.previous_blob_area = [1, 1]
        self.previous_keypoints = [None, None]
        self.minArea = 100
        detector_params = cv2.SimpleBlobDetector_Params()
        detector_params.filterByArea = True
        detector_params.minArea = 300
        detector_params.filterByColor = True
        detector_params.blobColor = 255
        # detector_params.maxArea = 1500
        self.blob_detector = cv2.SimpleBlobDetector_create(detector_params)

    def blob_track(self, img, prev_area=None, quantile=0.2, typ=cv2.THRESH_BINARY_INV):
        if img is None:
            return None
        flat = np.ndarray.flatten(img)
        threshold = np.quantile(flat, quantile)
        _, img = cv2.threshold(img, threshold, 255, typ)
        img = cv2.erode(img, None, iterations=5)
        img = cv2.dilate(img, np.ones((5, 5), np.uint8), iterations=7)
        img = cv2.medianBlur(img, 5)
        cv2.imshow("t_e_d_mB", img)

        keypoints = self.blob_detector.detect(img)

        ans = None
        if keypoints and len(keypoints) > 1:
            tmp = 1000
            for keypoint in keypoints:  # filter out odd blobs
                if abs(keypoint.size - prev_area) < tmp:
                    ans = keypoint
                    tmp = abs(keypoint.size - prev_area)

            keypoints = (ans,)
        return keypoints

    def draw(self, img, keypoints, dest=None):
        if dest is None:
            dest = img
        if img is None:
            return None
        d = cv2.drawKeypoints(
            img,
            keypoints,
            dest,
            (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        cv2.imshow("eye", d)

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


def eye_crop_border_to_center(img, eye_landmark):
    W = [i[0] for i in eye_landmark]
    H = [i[1] for i in eye_landmark]

    W_min, W_max = min(W), max(W)
    H_min, H_max = min(H), max(H)

    H_btw = H_max - H_min
    W_btw = W_max - W_min

    center = (round(H_btw / 2), round(W_btw / 2))

    H_border = round(H_btw * 0.5)
    W_border = round(W_btw * 0.2)

    result = img[H_min - H_border:H_max + H_border, W_min - W_border:W_max + W_border]
    result = np.expand_dims(result, axis=-1)
    return result


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
    path_name = "D:/JEON/dataset/look_direction/vid/4/04-4.mp4"
    num = path_name[path_name.rfind("/") - 1]
    if num == "1" or num == "4":
        Y = "right"
    elif num == "2" or num == "5":
        Y = "center"
    else:
        Y = "left"
    print(Y)
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("D:/mystudy/pholythec/Project/DMS/WIN_20220520_16_13_04_Pro.mp4")
    # cap = cv2.VideoCapture("D:/JEON/Polytechnics/Project/DMS/dataset/WIN_20220526_15_33_19_Pro.mp4")
    # cap = cv2.VideoCapture("D:/JEON/dataset/look_direction/vid/4/03-4.mp4")
    # cap = cv2.VideoCapture(path_name)


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
                if landmarks is not None:
                    landmarks_ndarray = MarkDetector.full_object_detection_to_ndarray(landmarks)
                    eyes = [eye_crop_none_border(gray, landmarks_ndarray[36:42]),
                            eye_crop_none_border(gray, landmarks_ndarray[42:48])]
                    for i in range(2):
                        eyes[i] = cv2.pyrUp(eyes[i])
                        key_points = HCBC.blob_track(eyes[i], HCBC.previous_blob_area[i])
                        kp = key_points or HCBC.previous_keypoints[i]
                        # if kp is not None:
                            # print(f"{i}: x:{kp[0].pt[0]} y:{kp[0].pt[1]}")
                        HCBC.draw(eyes[i], kp, frame)
                        # cv2.imshow(f"eyes[{i}]", eyes[i])
                        HCBC.previous_keypoints[i] = kp

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
