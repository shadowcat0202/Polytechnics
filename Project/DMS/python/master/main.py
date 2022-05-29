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

cap = cv2.VideoCapture(1)

FaceDetector = FaceDetector()   # 얼굴 인식 관련
MarkDetector = MarkDetector(save_model="../assets/shape_predictor_68_face_landmarks.dat")   # 랜드마크 관련
Tracker = Tracker() # 트래킹 관련
eye = eye_Net() # 눈 깜빡임 관련 + 모델
eye.eye_predictor("D:/Dataset/eye/model/cnn_eye_open_close(0.0918 0.9684).h5")  # 모델

try:
    cap = cv2.VideoCapture(0)
except:
    print("Error opening video stream or file")

while cap.isOpened():
    perv_time = time.time()
    ret, frame = cap.read()
    if ret:
        frame = np.array(imutils.resize(frame, width=RES_W, height=RES_H))  # imutils cv2 경량화 보조 패키지
        if frame is None:   break
        # cv2.imshow("1", frame)
        if Tracker.track_number == 0:
            Tracker.find_tracker(frame, FaceDetector)
        else:   # 트레킹 할 얼굴이 1명 이상 있다면(지금은 1명만 트레킹 하도록 작성함)
            box_rect = None
            if Tracker.frame_counter == 30:  # 60 프레임마다 한번씩 트래커 이탈 방지용 refaceDetection
                box_rect = Tracker.find_tracker(frame, FaceDetector, re=True)
            else:
                box_rect = Tracker.tracking(frame)  # 네모 박스 처준다 원한다면 rectangle타입 반환도 가능

            if box_rect is not None:
                # 랜드마크 type=full_object_detection --> .part().x, .part().x 형식으로 뽑아내기
                landmarks = MarkDetector.get_marks(frame, box_rect)
                # MarkDetector.draw_marks(frame, landmarks, color=GREEN)

# 눈 계산 + 예측(분명 학습은 잘 한거같은데 왜 직접 사용하면 뭔가 이상함)=========================================================
                landmarks = MarkDetector.full_object_detection_to_ndarray(landmarks)
                close_open_check = [1, 1]
                for i, land in enumerate([landmarks[36:42], landmarks[42:48]]):
                    close_open_check[i], bpnt = eye.eye_predict(frame, land)
                    if i == 0:
                        cv2.putText(frame, "open" if close_open_check[i] > 0.7 else "close",
                                    (bpnt[0], bpnt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
                    else:
                        cv2.putText(frame, "open" if close_open_check[i] > 0.7 else "close",
                                    (bpnt[2], bpnt[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
                    if close_open_check[i] < 0.5:
                        close_open_check[i] = 0
                    cv2.rectangle(frame, (bpnt[0], bpnt[1]), (bpnt[2], bpnt[3]), RED, 1)

                if close_open_check[0] == 0 and close_open_check[1] == 0:
                    cv2.putText(frame, "SLEEP!!!", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 1)



    cv2.putText(frame, f"fps:{int(1. / (time.time() - perv_time))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
    cv2.imshow('frame', frame)
    # if the `esc` key was pressed, break from the loop
    key = cv2.waitKey(1)
    if key == 27:
        break
            

cv2.destroyAllWindows()
cap.release()

