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

from output_signal_model import OutputSignalModel


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
cm = Camera()  # path를 안하면 카메라 하면 영상
# cm = Camera(path="D:/JEON/dataset/dataset_ver1.1.mp4")  # path를 안하면 카메라 하면 영s상

tk = Tracker()
fd = FaceDetector()
md = MarkDetector()
ey = HaarCascadeBlobCapture()
pe = PoseEstimator()
head = myHead()
mouth = myMouth()
outputModel = OutputSignalModel()

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
        imgNdarray = cm.getFrameResize2ndarray(frame)  # frame to ndarray
        gray = cv2.cvtColor(imgNdarray, cv2.COLOR_BGR2GRAY)  # (H, W)

        rect = tk.getRectangle(gray, fd)  # 트래킹 하는것과 동시에 face detect 반환
        if rect is not None:
            # print(rect)
            landmarks = md.get_marks(gray, rect)
            landmarks = md.full_object_detection_to_ndarray(landmarks)
            # md.draw_marks(imgNdarray, landmarks, color=cm.getRed())
            # cv2.rectangle(imgNdarray, (rect.left(), rect.top()),(rect.right(), rect.bottom()), (0, 255, 0), 3)
            landmarks_32 = np.array(landmarks, dtype=np.float32)
            pose = pe.solve_pose_by_68_points(landmarks_32)
            axis = pe.get_axis(gray, pose[0], pose[1])
            if axis is not None:
                pe.draw_annotation_box(imgNdarray, pose[0], pose[1], color=(0, 255, 0))
                pe.draw_axes(imgNdarray, pose[0], pose[1])
            cv2.imshow("rect", imgNdarray)

    else:
        cv2.destroyAllWindows()
        cm.cap.release()
        break


