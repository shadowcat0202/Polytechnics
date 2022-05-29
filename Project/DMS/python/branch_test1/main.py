import numpy as np
import cv2
import time
from my_mark_detector import MarkDetector, FaceDetector
from branch_test1.pose_estimator import PoseEstimator
import calculation as calc
from branch_test1.img_draw import *
import Preprocessing as ps
import tensorflow as tf
from EYE import eye_Net
import matplotlib.pyplot as plt

from functools import wraps

lastsave = 0

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
MARK_INDEX = RIGHT_EYE + LEFT_EYE + MOUTH_INLINE

total_frame = 0
detection_frame = 0
frame_sum = 0
frame_avg = 0


total_frame1 = 0
detection_frame1 = 0
frame_sum1 = 0

trackers = []

# video_capture = cv2.VideoCapture("D:/JEON/Polytechnics/Project/DMS/dataset/WIN_20220520_16_13_04_Pro.mp4")
# video_capture = cv2.VideoCapture("D:/mystudy/pholythec/Project/DMS/WIN_20220520_16_13_04_Pro.mp4")
video_capture = cv2.VideoCapture(1)  # 카메라

width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

pose_estimator = PoseEstimator(img_size=(height, width), model_path="../assets/model.txt")

# 3. Introduce a mark detector to detect landmarks.
face_detector = FaceDetector()
mark_detector = MarkDetector(save_model="../assets/shape_predictor_68_face_landmarks.dat")  # 경로 확인 필수!!

eye_calc = calc.eye_calculation()

eye_predictor = eye_Net()
# eye_model = eye_predictor.eye_predictor("D:/Dataset/eye/model/keras_eye_trained_model.h5")
eye_model = eye_predictor.eye_predictor("D:/Dataset/eye/model/cnn_eye_open_close(0.0918 0.9684).h5")

RES_W = 1280  # 1280 # 640 # 256 # 320 # 480 # pixels
RES_H = 720  # 720 # 480 # 144 # 240 # 360 # pixels

img_resize_rate = 0.8 # 원본 이미지에서 50%로 줄여도 detection에서는 크게 문제가 없다 640x400
if video_capture.isOpened():
    print("camera is ready")
    while True:
        total_frame += 1
        start_t = time.time()
        ret, img = video_capture.read()
        key = cv2.waitKey(1)
        if key == 27 or not ret:  # ESC
            break

        # img = cv2.resize(img,
        #                  (round(img.shape[1] * img_resize_rate), round(img.shape[0] * img_resize_rate)),
        #                  cv2.INTER_AREA)
        # print(type(img))
        img = cv2.resize(img, (RES_W, RES_H), cv2.INTER_AREA)
        # img, g, r = cv2.split(img)
        # b = b // 3

        # img = cv2.flip(img, 1) # 시나리오 측정할때는 좌우 반전 풀고 실행
        # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # 적외선 카메라 사용시

        rows, cols = len(img[0]), len(img)
        # img = ps.img_Preprocessing_v3(img)  # 프레임별 이미지 전처리
        # tholdc = ps.thresHold(cmpos)
        # tholdo = ps.thresHold(img)
        # IM = imgMake()
        # dstimage = IM.create_image_multiple(rows, cols, 3, 2, 2)
        # IM.showMultiImage(dstimage, cmpos, rows, cols, 1, 0, 0)
        # IM.showMultiImage(dstimage, tholdc, rows, cols, 1, 0, 1)
        # IM.showMultiImage(dstimage, tholdo, rows, cols, 1, 1, 0)

        face_box = face_detector.get_faceboxes(img)  # 전처리한 이미지에서 얼굴 검출

        if face_box is not None:
            detection_frame += 1
            landmark = mark_detector.get_marks(img, face_box)  # 얼굴에서 랜드마크 추출 type:list
            close_open_check = [1, 1]

            if landmark is not None:
                for i, land in enumerate([landmark[36:42], landmark[42:48]]):
                    eye_img_split, bpnt = eye_predictor.eye_img_split(img, land)
                    img_reshape = eye_predictor.make_input_shape(eye_img_split)
                    close_open_check[i] = eye_model.predict(img_reshape)[0][0]
                    if i == 0:
                        print(close_open_check[i], end=" ")
                    else:
                        print(close_open_check[i])
                    if close_open_check[i] < 0.7:
                        close_open_check[i] = 0
                    # cv2.rectangle(img, (bpnt[0], bpnt[1]), (bpnt[2], bpnt[3]), RED, 1)
                    # if close_open_check[0][0] < 0.7:
                    #     text = "close"
                    # else:
                    #     text = "open"
                    # cv2.putText(img, f"{text}", (bpos[0], bpos[1]),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

                if close_open_check[0] == 0 and close_open_check[1] == 0:
                    cv2.putText(img, "SLEEP!!!", (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 1)
            # pred = eye_predictor.my_predict(img[bdx1:bdx2, bdy1:bdy2])
            # cv2.rectangle(img,(bdx1, bdy1),(bdx2, bdy2), WHITE, 2)
            # cv2.putText(img, f"{pred}", (bdx1, bdy1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

            landmark_ndarray = np.array(landmark, dtype=np.float32)  # 파라미터 타입은 numpy.ndarray으로 해야함
            # 추가로 solve_pose_by_68_points 내부 함수 중 cv2.solvePnP의 인자중 랜드마크는 np.float32로 해주어야 한다
            pose = pose_estimator.solve_pose_by_68_points(landmark_ndarray)

            # dst = calc.get_rotate_border_box(img, face_box, landmark)
            # cv2.imshow("dst", dst)
            # cmpos1 = ps.img_Preprocessing_v2(dst)
            # face_box1 = face_detector.get_faceboxes(dst)

            # # 눈 감는거 + 자는거(부 정확함) 판단=======================================================
            # eye_close_status = eye_calc.eye_close(landmark[36:42], landmark[42:48])
            # if eye_close_status:
            #     eye_calc.close(img, color=WHITE)
            #     if eye_calc.close.count >= 10:
            #         cv2.putText(img, "SLEEPING!!!", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
            # # 눈 감김 판단 끝 =======================================================================

            # # 시각화 ===============================================================================
            # # 육면체를 보고싶다면?
            # pose_estimator.draw_annotation_box(img, pose[0], pose[1], color=GREEN)

            # # x, y, z 축 보고 싶다면?
            # axis = pose_estimator.get_axis(img, pose[0], pose[1])  # 축에 대한 데이터를 쓰고 싶은데 어디있는지 모름 ㅎ
            # axis = [[[RED_x RED_y]],[[GREEN_x GREEN_y]],[[BLUE_x BLUE_y]],[[CENTER_x CENTER_y]]]
            # --> BLUE(정면) GREEN(아래) RED(좌측) CENTER(중심)
            # pose_estimator.draw_axes(img, pose[0], pose[1])

            # # 얼굴 랜드마크 보고 싶다면?
            # mark_detector.draw_marks(img, landmark[36:48], color=RED)

            # # 얼굴 detect box 보고 싶다면?
            # detection_box = [face_box.left(), face_box.top(),
            #                  face_box.right(), face_box.bottom()]
            # mark_detector.draw_box(img, [detection_box], box_color=BLUE)

            # frame_sum += int(1. / (time.time() - start_t))
            # if frame_cnt == 3:
            #     frame_avg = round(frame_sum / frame_cnt)
            #     frame_cnt = 0
            #     frame_sum = 0
            # cv2.putText(img, f"FPS:{frame_avg}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
            # cv2.imshow("img", img)
            # cv2.putText(cmpos, "detection!!!", (250, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        cv2.putText(img, f"cmpos rate :{round((detection_frame / total_frame) * 100, 2)}", (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        cv2.imshow("cmpos", img)

cv2.destroyAllWindows()
video_capture.release()