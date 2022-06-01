import glob

import cv2
import dlib
import numpy as np
import imutils
import time

from my_face_detector import *
from my_mark_detector import *
from my_tracker import *
from EYE import eye_Net
from Preprocessing import *

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
    border_size_row = btw_row * 0.2
    border_size_col = btw_col * 0.5

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

FaceDetector = FaceDetector()  # 얼굴 인식 관련
MarkDetector = MarkDetector(save_model="../assets/shape_predictor_68_face_landmarks.dat")  # 랜드마크 관련
# cv2.matchTemplate()도 해보자
Tracker = Tracker()  # 트래킹 관련
# eye = eye_Net()  # 눈 깜빡임 관련 + 모델

vid_root_path = "D:/JEON/dataset/look_direction/vid/"
save_root_path = "D:/JEON/dataset/look_direction/cnn_eyes/"
save_look_number = [str(i) for i in range(1, 7)]
cap = None
for look_num in save_look_number:
    print(f"{vid_root_path + look_num} spliting...")
    _path = vid_root_path + look_num + "/"
    vid_name_list = glob.glob(_path + "*.mp4")

    img_name_counter = 0
    for vid_name in vid_name_list:
        vid_name = vid_name.replace("\\\\", "/")
        print(f"{vid_name} working...")
        try:
            # 카메라 or 영상
            # cap = cv2.VideoCapture(0)
            cap = cv2.VideoCapture(vid_name)
        except:
            print("Error opening video stream or file")

        RES_W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        RES_H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        while cap.isOpened():
            perv_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print(f"{vid_name} finish")
                break
            if ret:
                frame = np.array(frame)  # imutils cv2 경량화 보조 패키지
                frame = img_gray_Preprocessing(frame)
                # frame = np.array(imutils.resize(frame, width=RES_W, height=RES_H))

                if frame is None:
                    print(f"{vid_name} finish")
                    break

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
                        # MarkDetector.draw_marks(frame, landmarks, color=GREEN)

                        # 눈 계산 + 예측(분명 학습은 잘 한거같은데 왜 직접 사용하면 뭔가 이상함)=========================================================
                        landmarks = MarkDetector.full_object_detection_to_ndarray(landmarks)
                        for i, land in enumerate([landmarks[42:48], landmarks[36:42]]):
                            # close_open_check[i], bpnt = eye.eye_predict(frame, land)
                            if i == 0:  # 사람 기준 왼쪽눈
                                spl_img, _ = eye_img_split(frame, land)
                                spl_img = np.expand_dims(spl_img, axis=-1)
                                save_name = save_root_path + look_num + "/" + '{0:05}'.format(img_name_counter) + ".png"
                                # print(save_name)
                                cv2.imwrite(save_name,spl_img)
                                img_name_counter += 1
                                # eye_pred_val = str(close_open_check[i]) + ", "
                                # cv2.putText(view, f"{close_open_check[i]}" if close_open_check[i] > 0.9 else "close",
                                #             (400, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
                                # cv2.putText(view, "open" if close_open_check[i] > 0.9 else "close",
                                #             (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
                                # cv2.imshow("left", frame[bpnt[1]:bpnt[3], bpnt[0]:bpnt[2]])
                            else:   # 사람 기준 오른쪽 눈
                                spl_img, _ = eye_img_split(frame, land)
                                spl_img = np.expand_dims(spl_img, axis=-1)
                                save_name = save_root_path + look_num + "/" + '{0:05}'.format(img_name_counter) + ".png"
                                # print(save_name)
                                cv2.imwrite(save_name,spl_img)
                                img_name_counter += 1
                                # eye_pred_val += str(close_open_check[i])
                                # cv2.putText(view, f"{close_open_check[i]}" if close_open_check[i] > 0.9 else "close",
                                #             (400, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
                                # cv2.putText(view, "open" if close_open_check[i] > 0.9 else "close",
                                #             (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
                                # cv2.imshow("right", frame[bpnt[1]:bpnt[3], bpnt[0]:bpnt[2]])
                            print(img_name_counter)
                            # cv2.rectangle(view, (bpnt[0], bpnt[1]), (bpnt[2], bpnt[3]), WHITE, 1)
                cv2.imshow("wow",frame)
            # if the `esc` key was pressed, break from the loop
            key = cv2.waitKey(1)
            if key == 27:
                break

        cv2.destroyAllWindows()
        cap.release()

