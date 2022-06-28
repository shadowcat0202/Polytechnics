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

from pose_estimator import *
from myHead import *
from myMouth import *

from DecisionModel_v2 import *


def testPreprocessing(img):
    size = (35, 35)
    # 커널 모양
    kernel = []
    kernel.append(cv2.getStructuringElement(cv2.MORPH_RECT, size))  # 네모(avg)
    kernel.append(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size))  # 타원(best!!
    kernel.append(cv2.getStructuringElement(cv2.MORPH_CROSS, size))  # 십자가(worst)
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
    return np.vstack([a, b])


def Preprocessing(img):
    size = (30, 30)
    # 커널 모양
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)  # 세모(avg)
    thold = np.min(img) + np.std(img)
    img = cv2.medianBlur(img, 3, 3)
    img = np.where(img < thold, 255, 0).astype("uint8")
    result = cv2.erode(img, kernel)
    cv2.imshow("prep", result)
    return result


def drawCornerRect(img, r, sleep, color):
    cy = img.copy()
    if sleep == "Sleep":
        cy[r.top():r.bottom(), r.left():r.right(), 2] += 50
        color = (0, 0, 255)
    elif sleep == "Drows":
        cy[r.top():r.bottom(), r.left():r.right(), [0,2]] += 50
        color = (255, 0, 255)
    else:
        color = (0, 255, 0)

    cy = cv2.line(cy, (r.left(), r.top()), (r.left(), r.top() + 30), color, 2)
    cy = cv2.line(cy, (r.left(), r.top()), (r.left() + 30, r.top()), color, 2)

    cy = cv2.line(cy, (r.right(), r.bottom()), (r.right() - 30, r.bottom()), color, 2)
    cy = cv2.line(cy, (r.right(), r.bottom()), (r.right(), r.bottom() - 30), color, 2)

    cy = cv2.line(cy, (r.right(), r.top()), (r.right() - 30, r.top()), color, 2)
    cy = cv2.line(cy, (r.right(), r.top()), (r.right(), r.top() + 30), color, 2)

    cy = cv2.line(cy, (r.left(), r.bottom()), (r.left() + 30, r.bottom()), color, 2)
    cy = cv2.line(cy, (r.left(), r.bottom()), (r.left(), r.bottom() - 30), color, 2)
    return cy

def ROIinWindow(r, h, w):
    if r.left() < 0 or r.top() < 0 or r.right() >= w or r.bottom() >= h:
        return False
    return True


print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))
path = "D:/JEON/dataset/drive-download-20220627T050141Z-001/"
filename = ["WIN_20220624_15_58_44_Pro.mp4", "WIN_20220624_15_49_03_Pro.mp4", "WIN_20220624_15_40_21_Pro.mp4",
            "WIN_20220624_15_29_33_Pro.mp4"]
loop = 0
key = None
while True:
    video = path + filename[loop]
    loop += 1
    if loop % len(filename) == 0:
        loop = 0
    color = (0, 255, 0)
    # cm = Camera()  # path를 안하면 카메라 하면 영상
    cm = Camera(path=video)  # path를 안하면 카메라 하면 영상
    # cm = Camera(
    #     path="D:/JEON/dataset/drive-download-20220627T050141Z-001/WIN_20220624_15_40_21_Pro.mp4")  # path를 안하면 카메라 하면 영상

    tk = Tracker()
    fd = FaceDetector()
    md = MarkDetector()
    ey = HaarCascadeBlobCapture()
    pe = PoseEstimator()
    head = myHead()
    mouth = myMouth()
    dm2 = DecisionModel()

    ###
    arr_minlmax = np.array([[1000.0, -1000.0], [1000.0, -1000.0], [1000.0, -1000.0], [1000.0, -1000.0]])
    arr_ratios = np.array([[], [], [], []])
    # [min, max]
    # [ L_facetoEye, L_eyeHeightToWidth, R_facetoEye, R_eyeHeightToWidth]
    ###
    while cm.cap.isOpened():
        ret, frame = cm.cap.read()  # 영상 프레임 받기
        key = cv2.waitKey(1)
        if key == ord('n') or key == 27:
            break
        # md.changeMarkIndex(key)  # 랜드마크 점 종류를 바꾸고 싶다면 활성화 (미완성)

        if ret:
            imgNdarray = cm.getFrameResize2ndarray(frame)  # frame to ndarray
            gray = cv2.cvtColor(imgNdarray, cv2.COLOR_BGR2GRAY)  # (H, W)

            rect = tk.getRectangle(gray, fd)  # 트래킹 하는것과 동시에 face detect 반환
            if rect is not None and ROIinWindow(rect, cm.RES_H, cm.RES_W):
                landmarks = md.get_marks(gray, rect)
                landmarks = md.full_object_detection_to_ndarray(landmarks)
                # md.draw_marks(imgNdarray, landmarks, color=cm.getRed())

                """
                Blinking Test Starts Here
                """
                blk = BlinkDetector(imgNdarray, gray, landmarks)  # 블링크 디텍트 클래스 가동

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

                # blk.display_eye_status(results, status)
                # # status = 0 (눈을 감음), 1 (눈을 뜸)

                """
                Gaze Estimation Starts Here
                """
                dirctn = ""
                if status == 0:
                    pass
                else:
                    results_pred, dirctn = blk.eye_gaze_estimation(show=True)  # 동공표시 True
                    # blk.display_gaze_estimation(results_pred, dirctn)

                """
                headPose estimation start
                """
                lowerheadRasio, _ = head.lowerHeadCheck(landmarks)
                sleepHead = head.lowerHeadText(landmarks, gray)  # 수정했습니다 (class pose_estimator file_name= 부분)
                mouth.openMouthText(landmarks, gray)
                # x, y, z 축 보고 싶다면?
                landmarks_32 = np.array(landmarks, dtype=np.float32)
                pose = pe.solve_pose_by_68_points(landmarks_32)
                axis = pe.get_axis(gray, pose[0], pose[1])
                # axis = [[[RED_x RED_y]],[[GREEN_x GREEN_y]],[[BLUE_x BLUE_y]],[[CENTER_x CENTER_y]]]
                # --> BLUE(정면) GREEN(아래) RED(좌측) CENTER(중심)
                if axis is not None:
                    pe.draw_axes(gray, pose[0], pose[1])
                    headDirection = head.directionText(axis, imgNdarray)
                    cv2.putText(imgNdarray, f"{headDirection}",
                                (rect.right() + 30, rect.top() + 30),
                                cv2.FONT_HERSHEY_PLAIN,
                                fontScale=2, color=(0, 0, 255), thickness=3)

                """
                analyze_v2
                """
                sleeping = dm2.Update(imgNdarray, status, lowerheadRasio, 0)
                imgNdarray = drawCornerRect(imgNdarray, rect, sleeping, color)

                cv2.putText(imgNdarray, f"{sleeping}",
                            (rect.right() + 30, rect.top()), cv2.FONT_HERSHEY_PLAIN,
                            fontScale=2, color=(0, 0, 255), thickness=3)
                cv2.putText(imgNdarray, f"{dirctn}",
                            (rect.right() + 30, rect.top() + 60), cv2.FONT_HERSHEY_PLAIN,
                            fontScale=2, color=(0, 0, 255), thickness=3)
            else:
                cv2.putText(imgNdarray, f"not found detection",
                            (20, 80), cv2.FONT_HERSHEY_PLAIN,
                            fontScale=2, color=(0, 0, 255), thickness=3)
            cv2.imshow("output", imgNdarray)

        else:
            cv2.destroyAllWindows()
            cm.cap.release()
            break
    if key == 27:
        break
