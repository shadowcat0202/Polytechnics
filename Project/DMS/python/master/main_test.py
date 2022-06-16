import cv2
import dlib
import numpy as np
import time

from visualization import Camera  # 카메라 관련 (Color도 여기에서)
from tracker import *  # 트래킹
from face_detector import *  # 얼굴 detector
from mark_detector import *  # 랜드마크 detector
from Eye import *

print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))

# cm = Camera()
cm = Camera(path="D:/JEON/dataset/look_direction/vid/1/02-1.mp4")  # path를 안하면 카메라 하면 영상
tk = Tracker()
fd = FaceDetector()
md = MarkDetector()

while cm.cap.isOpened():
    ret, frame = cm.cap.read()  # 영상 프레임 받기
    key = cv2.waitKey(1)
    if key == 27:
        break
    # md.changeMarkIndex(key)  # 랜드마크 점 종류를 바꾸고 싶다면 활성화 (미완성)

    if ret:
        # 전처리 구간 ==========================================================
        imgNdarray = cm.getFrameResize2ndarray(frame)  # frame to ndarray
        gray = cv2.cvtColor(imgNdarray, cv2.COLOR_BGR2GRAY)  # (H, W)
        # ==================================================================

        rect = tk.getRectangle(gray, fd)  # 트래킹 하는것과 동시에 fd facedetect 반환
        if rect is not None:
            landmarks = md.get_marks(gray, rect)
            md.draw_marks(gray, landmarks, color=cm.getRed())
            landmarks = md.full_object_detection_to_ndarray(landmarks)

            eyes = [eye_crop_none_border(gray, landmarks[36:42]),
                    eye_crop_none_border(gray, landmarks[42:48])]

        cv2.imshow("gray", gray)
    else:
        break

cv2.destroyAllWindows()
cm.cap.release()
