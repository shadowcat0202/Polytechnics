import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils


def saturate_contrast2(p, num):
    pic = p.copy()
    pic = pic.astype('int32')
    pic = np.clip(pic + (pic - 128) * num, 0, 255)
    pic = pic.astype('uint8')
    return pic


def extract_eye(image, eye_landmark):
    lower_bound = max([pos[0] for pos in eye_landmark])
    upper_bound = min([pos[1] for pos in eye_landmark])

    eye = image[upper_bound - 3:lower_bound + 3, left[0] - 3:right[0] + 3]
    image[upper_bound - 3:lower_bound + 3, left[0] - 3:right[0] + 3] = eye


def find_pupil(right_eye):
    global gray_right_eye
    global threshold
    gray_right_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
    gray_right_eye = cv2.GaussianBlur(gray_right_eye, (25, 25), 0)
    _, threshold = cv2.threshold(gray_right_eye, 30, 255, cv2.THRESH_BINARY_INV)
    _, contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    return contours


def test_something_idea():
    # # file_path = "D:\JEON\dataset"
    # file_path = "D:\JEON\dataset"
    # img = cv2.imread(hl.win_path_to_python_path(file_path) + "/" + "images.png")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../assets/shape_predictor_68_face_landmarks.dat')

    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        # img = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.imshow("orgin", img)
        rects = detector(img, 0)
        for (i, rect) in enumerate(rects):
            landmarks = predictor(img, rect)
            landmarks = face_utils.shape_to_np(landmarks)
            right_eye = imutils.resize(extract_eye(img, landmarks[42:48]), width=200, height=100)
            img = cv2.bilateralFilter(img, -1, 10, 5)  # 이미지 or 영상 잡음 제거 (양방향 필터 - Bilateral filter)
            img = saturate_contrast2(img, 1)  # 위에서 사용했던 명암비 조절 함수, 명암비 2

        cv2.imshow("test", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
