import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils


class Haar_Like:
    def __init__(self):
        pass

    def landmarks_to_np(self, shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)
        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        # return the list of (x, y)-coordinates
        return coords

    def eye_on_mask(self, ret, mask, landmark, side):
        """        
        :param ret: face detector result
        :param mask:  눈 부분을 흰색으로 변경할 이미지 배열
        :param landmark: 랜드 마크 좌표 묶음
        :param side: 어느쪽 눈
        :return: 눈 부분 흰색으로 변경한 이미지 배열
        """
        points = [[landmark[i][0] - ret.left(), landmark[i][1] - ret.top()] for i in side]
        points = np.array(points, dtype=np.int32)
        mask = cv2.fillConvexPoly(mask, points, 255)  # 흰색으로 변경
        return mask

    def contouring(self, thresh, mid, img, right=False):
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        try:
            cnt = max(cnts, key=cv2.contourArea)
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            if right:
                cx += mid
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        except:
            pass

    def img_cut_rect(self, img, rect):
        result = img[rect.top():rect.bottom(), rect.left():rect.right()]
        return result

    def get_shape(self, rect):
        """
        얼굴 detection box shape 반환
        :param rect:
        :return: height, width
        """
        return rect.bottom() - rect.top(), rect.right() - rect.left()

    def win_path_to_python_path(self, _path):
        return _path.replace("\\", "/")

    def img_rotate(self, img, degree):
        h, w = img.shape[:-1]

        crossLine = int(((w * h + h * w) ** 0.5))
        centerRotatePT = int(w / 2), int(h / 2)
        new_h, new_w = h, w

        rotatefigure = cv2.getRotationMatrix2D(centerRotatePT, degree, 1)
        result = cv2.warpAffine(img, rotatefigure, (new_w, new_h))
        return result


left = [i for i in range(36, 42)]
right = [i for i in range(42, 48)]


def test():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../assets/shape_predictor_68_face_landmarks.dat')
    hl = Haar_Like()
    file_path = "D:\JEON\dataset"
    img = cv2.imread(hl.win_path_to_python_path(file_path) + "/" + "images.png", cv2.COLOR_BGR2GRAY)
    kernel = np.ones((9, 9), np.uint8)
    rects = detector(img, 0)
    for rect in rects:
        img_cut_rect = hl.img_cut_rect(img, rect)

        landmarks = predictor(img, rect)
        landmarks = hl.landmarks_to_np(landmarks)

        mask = np.zeros((hl.get_shape(rect)), dtype=np.uint8)
        mask = hl.eye_on_mask(rect, mask, landmarks, left)
        mask = hl.eye_on_mask(rect, mask, landmarks, right)
        mask = cv2.dilate(mask, kernel, 3)

        btw_and = cv2.bitwise_and(mask, mask, mask=mask)
        cv2.imshow("test", btw_and)
    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break


def something_function():
    pass

def wow():
    kernel = np.ones((9, 9), np.uint8)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../assets/shape_predictor_68_face_landmarks.dat')
    hl = Haar_Like()
    file_path = "D:\JEON\dataset\look_direction\cnn_eyes_number_sort"
    img = cv2.imread(hl.win_path_to_python_path(file_path) + "/" + "2_13796.png")
    cv2.imshow("orin", img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype("uint8")[:,:,0]
    img = cv2.GaussianBlur(img, (7,7), 0)
    img = cv2.equalizeHist(img)
    print(img.shape)


    _, src_bin = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow("wow", src_bin)
    src_bin = cv2.erode(src_bin, kernel, iterations=1)
    contours, _ = cv2.findContours(src_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = np.expand_dims(contours, axis=-1)
    # contours = np.expand_dims(contours, axis=0)
    print(contours.shape)

    contours = sorted(contours, key=lambda x: cv2.contourArea(x),
                      reverse=True)  # Print the contour with the biggest area first.
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(src_bin, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(src_bin, (x + int((w / 2)), 0), (x + int((w / 2)), src_bin.shape[0]), (0, 255, 0), 2)
        cv2.line(src_bin, (0, y + int(h / 2)), (src_bin.shape[1], y + int(h / 2)), (0, 255, 0), 2)

        break  # After printing contour with the biggest area, draw no more contours

    src_bin = cv2.resize(src_bin, (500, 300))

    # print(contours)
    cv2.imshow("src_bin", src_bin)
    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break

def main():
    HL = Haar_Like()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../assets/shape_predictor_68_face_landmarks.dat')
    LEFT = list(range(36, 42))
    RIGHT = list(range(42, 48))

    cap = cv2.VideoCapture(1)
    ret, img = cap.read()
    img_copy = img.copy()

    cv2.namedWindow("image")

    kernel = np.ones((9, 9), np.uint8)

    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.imshow("gray", gray)
        face_detect = detector(gray, 0)
        if face_detect is not None:
            for rect in face_detect:
                shape = predictor(gray, rect)
                shape = HL.landmarks_to_np(shape)
                mask = np.zeros((HL.get_shape(rect)), dtype=np.uint8)
                mask = HL.eye_on_mask(img, left)
                mask = HL.eye_on_mask(img, right)
                mask = cv2.dilate(mask, kernel, 5)
                cut_img_rect = HL.img_cut_rect(img, rect)
                # cv2.imshow("cut", cut_img_rect)
                eyes = cv2.bitwise_and(cut_img_rect, cut_img_rect, mask=mask)
                cv2.imshow("eyes", eyes)
        if cv2.waitKey(1) & 0xFF == 27:
            break


wow()
