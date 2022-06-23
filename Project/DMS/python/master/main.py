import cv2
import dlib
import numpy as np
import time

from visualization import Camera  # 카메라 관련 (Color도 여기에서 get함수로 받아올 수 있다)
from tracker import *  # 트래킹
from face_detector import *  # 얼굴 detector
from mark_detector import *  # 랜드마크 detector
from Eye import *

from SYblink_detector import *
from SYpupil_detector import *

from pose_estimator import *
from myHead import *
from myMouth import *


def testPreprocessing(img):
    size = (35,35)
    # 커널 모양
    kernel = []
    kernel.append(cv2.getStructuringElement(cv2.MORPH_RECT, size))  # 네모(avg)
    kernel.append(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size))  # 타원(best!!
    kernel.append(cv2.getStructuringElement(cv2.MORPH_CROSS, size)) # 십자가(worst)
    thold = np.min(img) + np.std(img)
    img = cv2.medianBlur(img, 3, 3)
    img = np.where(img < thold, 255, 0).astype("uint8")
    # img = cv2.erode(img, (5, 5), iterations=9)
    a = []
    a.append(img)
    a.append(cv2.erode(img, kernel[0]))
    a = np.hstack(a)
    b = []
    b.append(cv2.erode(img, kernel[1]))
    b.append(cv2.erode(img, kernel[2]))
    b = np.hstack(b)
    return np.vstack([a,b])

def Preprocessing(img):
    size = (30,30)
    # 커널 모양
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)  # 네모(avg)
    thold = np.min(img) + np.std(img)
    img = cv2.medianBlur(img, 3, 3)
    img = np.where(img < thold, 255, 0).astype("uint8")
    result = cv2.erode(img, kernel)
    cv2.imshow("prep", result)
    return result

print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))

# cm = Camera()
# cm = Camera(path="D:/JEON/dataset/dataset_ver1.1.mp4")  # path를 안하면 카메라 하면 영상
cm = Camera(path="D:/JEON/dataset/dataset_ver1.1.mp4")  # path를 안하면 카메라 하면 영상
tk = Tracker()
fd = FaceDetector()
md = MarkDetector()
ey = HaarCascadeBlobCapture()
pe = PoseEstimator()
head = myHead()
mouth = myMouth()

###
arr_minlmax = np.array([[1000.0, -1000.0], [1000.0, -1000.0], [1000.0, -1000.0], [1000.0, -1000.0]])
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
            landmarks = md.full_object_detection_to_ndarray(landmarks)
            md.draw_marks(imgNdarray, landmarks, color=cm.getRed())


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

            """
            Gaze Estimation Starts Here
            """
            if status ==0: # 눈이 감겨있다면 패스
                pass
            else: # 눈이 떠져있을 경우
                # 동공의 좌표, 각 눈의 판단값 출력
                xy_pupls, est_dir_LR = blk.get_pupil_location_and_direction()
                """
                xy_pupls = [ (x1, y1), (x2, y2) ]
                    * x1, y1 = 왼쪽 눈 중앙의 x,y 좌표
                    * x2, y2 = 오른쪽 눈 중앙의 x,y 좌표
                    
                est_dir_LR [ (idx1), (idx2) ]
                    * idx1 = 왼쪽 눈의 판단 값 (0: 좌, 1: 중, 2:우) 
                    * idx2 = 오른쪽 눈의 판단 값 (0: 좌, 1: 중, 2:우) 
                """
                # 최종 판단 값 출력
                blk.decide_on_gaze_direction(est_dir_LR) #Return 값 X, view 이미지에 판단 값 puttext로 표현
                """
                방향 출력 값: LEFT, RIGHT, CENTRE, ERR (모두 만족하지 않는 경우)
                """
            """
            Gaze Estimation Ends Here
            """


            """
            headPose estimation start
            """
            head.lowerHeadText(landmarks, gray)  # 수정했습니다 (class pose_estimator file_name= 부분)
            mouth.openMouthText(landmarks, gray)
            # x, y, z 축 보고 싶다면?
            landmarks_32 = np.array(landmarks, dtype=np.float32)
            pose = pe.solve_pose_by_68_points(landmarks_32)
            axis = pe.get_axis(gray, pose[0], pose[1])
            # axis = [[[RED_x RED_y]],[[GREEN_x GREEN_y]],[[BLUE_x BLUE_y]],[[CENTER_x CENTER_y]]]
            # --> BLUE(정면) GREEN(아래) RED(좌측) CENTER(중심)
            # >>> head ===========================================================================
            if axis is not None:
                pe.draw_axes(gray, pose[0], pose[1])
                head.directionText(axis, imgNdarray)
            # cv2.imshow("img_gray", gray)
            # >>> test ==========================================================================
            # if axis is not None:
            #     pe.draw_axes(imgNdarray, pose[0], pose[1])
            #     head.directionText(axis, imgNdarray)
            # cv2.imshow("imgNdarray", imgNdarray)
            # # =============================================================================
            cv2.imshow("output", imgNdarray)
    else:
        cv2.destroyAllWindows()
        cm.cap.release()
        break


