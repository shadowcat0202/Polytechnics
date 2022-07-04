import timeit

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
# path = "D:/JEON/dataset/"
filename = ["WIN_20220624_15_58_44_Pro", "WIN_20220624_15_49_03_Pro", "WIN_20220624_15_40_21_Pro",
            "WIN_20220624_15_29_33_Pro"]    # , "dataset_ver1.1"
TF = [14134, 13257, 12778, 10281]



loop = 3
key = None
while True:
    test_y = path + filename[loop] + "._세환.txt"
    video = path + filename[loop] + ".mp4"

    if loop % len(filename) == 0:
        loop = 0
    color = (0, 255, 0)
    # cm = Camera()  # path를 안하면 카메라 하면 영상
    cm = Camera(path=video)  # path를 안하면 카메라 하면 영상
    # cm = Camera(
    #     path="D:/JEON/dataset/drive-download-20220627T050141Z-001/WIN_20220624_15_40_21_Pro.mp4")  # path를 안하면 카메라 하면 영상


    frame_control = 150


    tk = Tracker()
    fd = FaceDetector()
    md = MarkDetector()
    ey = HaarCascadeBlobCapture()
    pe = PoseEstimator()
    head = myHead(frame_control)
    mouth = myMouth()
    dm2 = DecisionModel()

    start_time = None
    end_time = None
    total_frame = 0
    detection_frame = 0
    dfps = []
    hit = 0
    miss = 0
    acc = 0
    file = open(test_y, "r")
    y1 = []
    y2 = []

    ###
    arr_minlmax = np.array([[1000.0, -1000.0], [1000.0, -1000.0], [1000.0, -1000.0], [1000.0, -1000.0]])
    arr_ratios = np.array([[], [], [], []])
    head_log = {}
    headStatus_log = np.array([])
    arr_headRatios = np.array([])
    arr_headThold = np.array([])

    # [min, max]
    # [ L_facetoEye, L_eyeHeightToWidth, R_facetoEye, R_eyeHeightToWidth]
    ###
    cnt_frm = 0
    head_down = None
    while cm.cap.isOpened():
        ret, frame = cm.cap.read()  # 영상 프레임 받기
        key = cv2.waitKey(1)
        if key == ord('n') or key == 27:
            cv2.destroyAllWindows()
            cm.cap.release()
            break
        # md.changeMarkIndex(key)  # 랜드마크 점 종류를 바꾸고 싶다면 활성화 (미완성)
        if ret:
            total_frame += 1
            start_time = timeit.default_timer()
            line = file.readline()
            # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            imgNdarray = cm.getFrameResize2ndarray(frame)  # frame to ndarray
            gray = cv2.cvtColor(imgNdarray, cv2.COLOR_BGR2GRAY)  # (H, W)

            rect = tk.getRectangle(gray, fd)  # 트래킹 하는것과 동시에 face detect 반환
            if rect is not None and ROIinWindow(rect, cm.RES_H, cm.RES_W):
                detection_frame += 1
                landmarks = md.get_marks(gray, rect)
                landmarks = md.full_object_detection_to_ndarray(landmarks)
                # md.draw_marks(imgNdarray, landmarks, color=cm.getGreen())

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
                eye_gaze = None
                if status == 0:
                    pass
                else:
                    results_pred, dirctn, eye_gaze = blk.eye_gaze_estimation(show=True)  # 동공표시 True
                    # blk.display_gaze_estimation(results_pred, dirctn)

                """
                headPose estimation start
                """
                # lowerheadRasio, _ = head.lowerHeadCheck(landmarks)
                # sleepHead = head.lowerHeadText(landmarks, gray)  # 수정했습니다 (class pose_estimator file_name= 부분)
                # mouth.openMouthText(landmarks, gray)
                # # x, y, z 축 보고 싶다면?
                landmarks_32 = np.array(landmarks, dtype=np.float32)
                pose = pe.solve_pose_by_68_points(landmarks_32)
                axis = pe.get_axis(gray, pose[0], pose[1])
                # # axis = [[[RED_x RED_y]],[[GREEN_x GREEN_y]],[[BLUE_x BLUE_y]],[[CENTER_x CENTER_y]]]
                # # --> BLUE(정면) GREEN(아래) RED(좌측) CENTER(중심)
                headDirection = None
                if axis is not None:
                    # pe.draw_axes(gray, pose[0], pose[1])
                    headDirection = head.directionText(axis, imgNdarray)
                #     cv2.putText(imgNdarray, f"{headDirection}",
                #                 (rect.right() + 30, rect.top() + 30),
                #                 cv2.FONT_HERSHEY_PLAIN,
                #                 fontScale=2, color=(0, 0, 255), thickness=3)
                """
                시영 코드
                """
                """ SY - HEAD NORMALIZED STARTS HERE """
                lowerheadRasio, _ = head.lowerHeadCheck(landmarks)
                # 두 자리 정수로 ratio 변환
                headRatio = round(lowerheadRasio * 100, 0).astype('int8')

                # ratio 관련 정보 저장/계산
                head_log, headRatio_mode, headThold = head.updatedLog_modeRatio_tholdRatio(head_log, headRatio)
                # head_log = ratio값과 1(카운트)을 저장하는 딕셔너리
                # head_headRatio_mode = 딕셔너리 내 최빈값 (발생빈도가 가장 높은 ratio)
                # headThold = 고개 숙였다를 판단하는 기준척도. (최빈값 이후 카운트가 가장 낮아지는 지점)

                # 고개 숙였는지 판단
                head_down = head.head_down(headRatio, headThold)

                # headLog_sorted = sorted(head_log.items(),key=lambda x:x[0]) #ToDo - 지워도 됨. 딕셔너리 확인용
                # print(f"headLog = {headLog_sorted}") #ToDo - 지워도 됨. 딕셔너리 확인용

                # headStatus_log = head.updated_statusLog(headStatus_log, headThold, headRatio) #ToDo - 지워도 됨. 그래프 확인용
                # arr_headRatios = np.append(arr_headRatios, headRatio) #ToDo-Delete 그래프 확인용
                # arr_headThold = np.append(arr_headThold, headThold) #ToDo-Delete 그래프 확인용

                # cv2.putText(imgNdarray, f"Down: {head_down}", (500, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
                # cv2.putText(imgNdarray, f"Ratio: {headRatio}", (500, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
                # cv2.putText(imgNdarray, f"Thold: {headThold}", (500, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
                # cv2.putText(imgNdarray, f"Mode: {headRatio_mode}", (500, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
                """ SY - HEAD CEHCK ENDS HERE """

                """
                analyze_v2
                """
                warning = dm2.Update(imgNdarray, status, head_down, eye_gaze, headDirection)
                # imgNdarray = drawCornerRect(imgNdarray, rect, sleeping, color)
                end_time = timeit.default_timer()
                dfps.append(int(1./(end_time-start_time)))
                y_value = None
                if int(line.split(",")[1][1]) == 1:
                    y_value = True
                else:
                    y_value = False
                # print(f"y_value:{y_value}, warning: {warning}")
                if y_value == warning:
                    hit += 1
                else:
                    miss += 1

                # cv2.putText(imgNdarray, f"warning: {warning}",
                #             (100, 40), cv2.FONT_HERSHEY_PLAIN,
                #             fontScale=2, color=(0, 0, 255), thickness=3)

            else:
                # cv2.putText(imgNdarray, f"not found detection",
                #             (20, 80), cv2.FONT_HERSHEY_PLAIN,
                #             fontScale=2, color=(0, 0, 255), thickness=3)
                pass
            if total_frame % 500 == 0:
                print(f"{round(total_frame/TF[loop] * 100, 2)}%")


        else:
            cv2.destroyAllWindows()
            cm.cap.release()
            print(f"{video}\n"
                  f"total_frame:{total_frame}, detection_frame:{detection_frame}\n"
                  f"avg FPS:{sum(dfps) // detection_frame} acc:{round((hit / detection_frame) * 100, 4)}")
            break
        # cv2.imshow("output", imgNdarray)
    if key == 27:
        break
    loop += 1
    if loop == 4:
        break
