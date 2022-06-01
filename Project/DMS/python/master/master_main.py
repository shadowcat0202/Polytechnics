import cv2
import dlib
import numpy as np
import imutils
import time


from my_face_detector import *
from my_mark_detector import *
from pose_estimator import PoseEstimator
from my_tracker import *
from EYE import eye_Net
from Preprocessing import *
from img_draw import imgMake
img_size = (100,100)

def make_input_shape(img):
    # print(f"make_input_shape(img){type(img)}, img.shape={img.shape}, len(img.shape)={len(img.shape)}")
    result = None
    if len(img.shape) == 2:
        # 차원 증가 (a, b) --> (a, b, 1)
        # img = cv2.resize(img, img_size)
        # img = np.expand_dims(img, axis=-1)
        # img = np.expand_dims(img, axis=0)
        # result = img.reshape(1, img_size[0], img_size[1], 1)
        # result = img.reshape(img_size[0], img_size[1], 1)
        result = np.expand_dims(img, axis=-1)
    elif len(img.shape) == 3:
        if img.shape[2] == 3:
            img, _, _ = cv2.split(img)  # 흑백만 남긴다
            # img = cv2.resize(img, (img_size[0], img_size[1]))
            # img = np.expand_dims(img, axis=-1)
            # result = np.expand_dims(img, axis=0)
            # result = img.reshape(1, img_size[0], img_size[1], 1)
            # result = img.reshape(img_size[0], img_size[1], 1)
            result = np.expand_dims(img, axis=-1)
    # (1, 90, 90, 1)로 만들어주기 위해서 reisze(90,90)
    # -> 차원 추가 뒤(axis=-1)(90, 90, 1)
    # -> 차원 추가 앞(axis= 0)(1, 90, 90, 1)
    return result


def eye_img_split(img, eye_landmark):
    row = [i[0] for i in eye_landmark]
    col = [i[1] for i in eye_landmark]

    row_min, row_max = min(row), max(row)
    col_min, col_max = min(col), max(col)

    btw_row = row_max - row_min
    btw_col = col_max - col_min
    per = 0.5
    border_size_row = btw_row * per
    border_size_col = btw_col * per

    row_mid = row_max - row_min
    col_mid = col_max - col_min

    row_border1 = int(row_min - border_size_row)
    col_border1 = int(col_min - border_size_col)
    row_border2 = int(row_max + border_size_row)
    col_border2 = int(col_max + border_size_col)

    result = np.array(img[col_border1:col_border2, row_border1:row_border2])
    return result, [row_border1, col_border1, row_border2, col_border2]
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



FaceDetector = FaceDetector()   # 얼굴 인식 관련
MarkDetector = MarkDetector(save_model="../assets/shape_predictor_68_face_landmarks.dat")   # 랜드마크 관련
# cv2.matchTemplate()도 해보자
Tracker = Tracker() # 트래킹 관련
# eye = eye_Net() # 눈 깜빡임 관련 + 모델
# eye.eye_predictor("D:/JEON/dataset/eye/model/keras_eye_trained_model_wow.h5")  # 모델
iMake = imgMake()

cap = None
try:
    # 카메라 or 영상
    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("D:/JEON/dataset/haar-like/WIN_20220601_15_48_25_Pro.mp4")
    cap = cv2.VideoCapture("D:/JEON/Polytechnics/Project/DMS/dataset/WIN_20220526_15_33_19_Pro.mp4")

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
        # frame = np.array(imutils.resize(frame, width=RES_W, height=RES_H))

        if frame is None:   break

        if Tracker.track_number == 0:
            Tracker.find_tracker(frame, FaceDetector)
        else:   # 트레킹 할 얼굴이 1명 이상 있다면(지금은 1명만 트레킹 하도록 작성함)
            box_rect = None
            if Tracker.frame_counter == 60:  # 60 프레임마다 한번씩 트래커 이탈 방지용 refaceDetection
                box_rect = Tracker.find_tracker(frame, FaceDetector, re=True)
            else:
                box_rect = Tracker.tracking(frame)  # 네모 박스 처준다 원한다면 rectangle타입 반환도 가능

            if box_rect is not None:
                # 랜드마크 type=full_object_detection --> .part().x, .part().x 형식으로 뽑아내기
                landmarks = MarkDetector.get_marks(frame, box_rect)
                # MarkDetector.draw_marks(frame, landmarks, color=GREEN)

# 눈 계산 + 예측(분명 학습은 잘 한거같은데 왜 직접 사용하면 뭔가 이상함)=========================================================
                landmarks = MarkDetector.full_object_detection_to_ndarray(landmarks)
                eye_pred_val = ""
                # close_open_check = [None, None]
                for i, land in enumerate([landmarks[36:42], landmarks[42:48]]):
                    # close_open_check[i], bpnt = eye.eye_predict(frame, land)
                    if i == 0:
                        spl, border_pnt = eye_img_split(frame, land)
                        img_reshape = make_input_shape(spl)
                        print(f"{img_reshape.shape}", end=", ")
                        cv2.imshow("left", img_reshape)
                        # eye_pred_val = str(close_open_check[i]) + ", "
                        # cv2.putText(view, f"{close_open_check[i]}" if close_open_check[i] > 0.9 else "close",
                        #             (400, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
                        # cv2.putText(view, "open" if close_open_check[i] > 0.9 else "close",
                        #             (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
                        # cv2.imshow("left", frame[bpnt[1]:bpnt[3], bpnt[0]:bpnt[2]])
                    else:
                        spl, border_pnt = eye_img_split(frame, land)
                        img_reshape = make_input_shape(spl)
                        print(f"{img_reshape.shape}")
                        cv2.imshow("right", img_reshape)
                        # eye_pred_val += str(close_open_check[i])
                        # cv2.putText(view, f"{close_open_check[i]}" if close_open_check[i] > 0.9 else "close",
                        #             (400, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
                        # cv2.putText(view, "open" if close_open_check[i] > 0.9 else "close",
                        #             (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
                        # cv2.imshow("right", frame[bpnt[1]:bpnt[3], bpnt[0]:bpnt[2]])

                    # if close_open_check[i] > 0.9:
                    #     close_open_check[i] = 1
                    # cv2.rectangle(view, (bpnt[0], bpnt[1]), (bpnt[2], bpnt[3]), WHITE, 1)

                # if close_open_check[0] == 0 and close_open_check[1] == 0:
                #     cv2.putText(view, "SLEEP!!!", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 3)
                # print(eye_pred_val)



        cv2.putText(view, f"fps:{int(1. / (time.time() - perv_time))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        cv2.imshow('show_frame', view)
    # if the `esc` key was pressed, break from the loop
    key = cv2.waitKey(1)
    if key == 27:
        break
            

cv2.destroyAllWindows()
cap.release()
