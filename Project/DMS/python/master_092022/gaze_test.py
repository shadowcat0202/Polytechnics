import cv2
import dlib
import numpy as np
import time

from visualization import Camera  # 카메라 관련 (Color도 여기에서 get함수로 받아올 수 있다)
from tracker import *  # 트래킹
from face_detector import *  # 얼굴 detector
from mark_detector import *  # 랜드마크 detector
from Eye import *
from gaze_sub.eye_traiking import GazeTracking


def testPreprocessing(img):
    size = (35, 35)
    # 커널 모양
    kernel = []
    kernel.append(cv2.getStructuringElement(cv2.MORPH_RECT, size))  # 네모(avg)
    kernel.append(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size))  # 타원(best!!)
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


def landMarkPutOnlyRectangle(img, rect, md):
    """
    :param img: 원본 이미지
    :param rect: 얼굴 detection : dlib._dlib_pybind11.rectangle
    :return: rect에 roi된 이미지, rect roi에 맞춘 랜드마크
    """

    landmark = md(img, rect)
    rectImg = img[rect.top():rect.bottom(), rect.left():rect.right()]
    # print(x1, y1)

    for i in range(68):
        landmark.part(i).x -= rect.left()
        landmark.part(i).y -= rect.top()

    return rectImg, landmark


def draw_marks(image, marks, color=(225, 255, 255)):
    for i in range(36, 48):
        cv2.circle(image, (marks.part(i).x, marks.part(i).y), 1, color, -1, cv2.LINE_AA)


print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))

# cm = Camera()
cm = Camera(path="D:/JEON/dataset/dataset_ver1.1.mp4")  # path를 안하면 카메라 하면 영상
# cm = Camera(path="D:/Dataset/dataset_ver1.1.mp4")  # path를 안하면 카메라 하면 영상
tk = Tracker()
fd = FaceDetector()
md = dlib.shape_predictor("./assets/shape_predictor_68_face_landmarks.dat")
# ey = HaarCascadeBlobCapture()
gaze = GazeTracking(fd, md)  # 응시 트래킹

facedetector_test = dlib.get_frontal_face_detector()

while cm.cap.isOpened():
    ret, frame = cm.cap.read()  # 영상 프레임 받기
    key = cv2.waitKey(1)
    if key == 27:
        break
    # md.changeMarkIndex(key)  # 랜드마크 점 종류를 바꾸고 싶다면 활성화 (미완성)

    if ret:
        frame = cm.getFrameResize2ndarray(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rect = tk.getRectangle(gray, fd)  # 트래킹 하는것과 동시에 face detect 반환

        if rect is not None:
            # gray, landmarks = landMarkPutOnlyRectangle(gray, rect, md)
            landmarks = md(gray, rect)
            gaze.refresh(gray, landmarks)
            gray = gaze.annotated_frame()

        # frame = gaze.annotated_frame()
        # text = ""
        #
        # if gaze.is_blinking():
        #     text = "BLINKING"
        # elif gaze.is_right():
        #     text = "RIGHT"
        # elif gaze.is_left():
        #     text = "LEFT"
        # elif gaze.is_center():
        #     text = "CENTER"
        #
        # cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
        # left_pupil = gaze.pupil_left_coords()
        # right_pupil = gaze.pupil_right_coords()
        # cv2.putText(frame, "L pupil: " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        # cv2.putText(frame, "R pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        #
        # cv2.imshow("Demo", frame)
    else:
        cv2.destroyAllWindows()
        cm.cap.release()
        break
