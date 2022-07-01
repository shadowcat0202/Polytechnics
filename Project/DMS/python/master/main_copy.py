# import timeit
import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
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
# cm = Camera(0)  # path를 안하면 카메라 하면 영상
cm = Camera(path="C:/Dropbox/PC/Documents/project_DMS/draft_paper/dataset_ver1.1.mp4")  # path를 안하면 카메라 하면 영상
# cm = Camera(path = 1)  # path를 안하면 카메라 하면 영상
# cm = Camera(1)  # path를 안하면 카메라 하면 영상
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

            # blk.time_it(blk.__init__())
            # 각 ratio의 정규화 값이 0.4를 초과하는지 여부 확인.
            # 0.4 초과 시 1 (눈을 뜸), 0.4 이하(눈을 감음)
            arr_minlmax, arr_nmRatios = blk.new_minlmax_and_normalized_ratios(arr_minlmax)

            """
                arr_minlmax = [ [min1/max1], [min2/max2], [min3/max3], [min4, max4] ]
                     * 1,2,3,4 = ratio1, ratio2, ratio3, ratio4
                     * ratio1 = 눈 면적(L) / 얼굴면적
                     * ratio2 = 눈 면적(R) / 얼굴면적
                     * ratio3 = 눈 높이(L) / 눈 너비(L)
                     * ratio4 = 눈 눞이(R) / 눈 너비(R)
                arr_NMratios = [ ratio1, ratio2, ratio3, ratio4 ]

            """
            # # results, status = blk.is_open(ratios_NM)
            # # results = [1,1,1,0] 3개의 지표에서 떴다고 판단, 1개의 지표는 감았다고 판단
            # # result 내 1의 갯수가 0 갯수 초과일 경우 1을 산출.
            results, status = blk.eye_status_open(arr_nmRatios)

            blk.display_eye_status(results, status)
            # # status = 0 (눈을 감음), 1 (눈을 뜸)
            """
            Blinking Test Ends Here
            """

            """
            Gaze Estimation Starts Here
            """
            if status == 0: pass
            else:
                results_pred, dirctn = blk.eye_gaze_estimation(show=True) #동공표시 True
                blk.display_gaze_estimation(results_pred, dirctn)
            """
            Gaze Estimation Ends Here
            """

    else:
        break
    cv2.imshow("original", imgNdarray)

cm.cap.release()
cv2.destroyAllWindows()
"""
Modification
-Deleted
eyes = [eye_crop_none_border(gray, landmarks[36:42]),
eye_crop_none_border(gray, landmarks[42:48])]

"""
