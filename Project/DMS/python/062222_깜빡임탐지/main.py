# import seaborn as sns
import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
import time
from SYblink_detector import *
from visualization import Camera  # 카메라 관련 (Color도 여기에서 get함수로 받아올 수 있다)
from tracker import *  # 트래킹
from face_detector import *  # 얼굴 detector
from mark_detector import *  # 랜드마크 detector
from Eye import *
from SYpupil_detector import *
from SYcommon_calculator import *

print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))

# cm = Camera()
cm = Camera(path="D:/JEON/dataset/dataset_ver1.1.mp4")  # path를 안하면 카메라 하면 영상
tk = Tracker()
fd = FaceDetector()
md = MarkDetector()

###
arr_minlmax = np.array([[1000.0, -1000.0],[1000.0, -1000.0],[1000.0, -1000.0],[1000.0, -1000.0] ])
arr_ratios = np.array([[], [], [], []])
# [min, max]
# [ L_facetoEye, L_eyeHeightToWidth, R_facetoEye, R_eyeHeightToWidth]
###

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

        rect = tk.getRectangle(gray, fd)  # 트래킹 하는것과 동시에 face detect 반환
        if rect is not None:
            landmarks = md.get_marks(gray, rect)
            md.draw_marks(imgNdarray, landmarks, color=cm.getRed())
            landmarks = md.full_object_detection_to_ndarray(landmarks)

            """
            Blinking Test Starts Here
            """
            blk = BlinkDetector(imgNdarray, gray, landmarks) # 블링크 디텍트 클래스 가동
            ratios_minlmax, ratios_NM = blk.update_min_max_then_get_normalized_ratios(arr_minlmax) # 4개의 ratio의 min/max 범위값, 정규화된 ratio 값 산출
            """
                   ratios_minlmax = [ [min1/max1], [min2/max2], [min3/max3], [min4, max4] ]
                       * 1,2,3,4 = ratio1, ratio2, ratio3, ratio4
                           * ratio1 = 눈 면적(L) / 얼굴면적
                           * ratio2 = 눈 면적(R) / 얼굴면적
                           * ratio3 = 눈 높이(L) / 눈 너비(L)
                           * ratio4 = 눈 눞이(R) / 눈 너비(R)
            """
            # 각 ratio의 정규화 값이 0.4를 초과하는지 여부 확인.
            # 0.4 초과 시 1 (눈을 뜸), 0.4 이하(눈을 감음)
            results, status = blk.is_open(ratios_NM)
            # results = [1,1,1,0] 3개의 지표에서 떴다고 판단, 1개의 지표는 감았다고 판단
            # result 내 1의 갯수가 0 갯수 초과일 경우 1을 산출.
            # status = 0 (눈을 감음), 1 (눈을 뜸)
            msg = "OPEN" if status == 1 else "CLOSED" # 이미지에 표기할 메시지

            cv2.putText(imgNdarray,f"status:{msg}",(620,300), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=cm.getBlue(), thickness=1 )
            cv2.putText(imgNdarray,f"results: {results}",(620,330), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=cm.getRed(), thickness=1 )

            """
            Blinking Test Ends Here
            """



    else:
        break
    cv2.imshow("original", imgNdarray)

cv2.destroyAllWindows()
cm.cap.release()
"""
Modification
-Deleted
eyes = [eye_crop_none_border(gray, landmarks[36:42]),
eye_crop_none_border(gray, landmarks[42:48])]

"""
