import cv2
import dlib
import numpy as np

class Haar_Like:
    def __init__(self):
        pass

    def shape_to_np(self, shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)
        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        # return the list of (x, y)-coordinates
        return coords

    def eye_on_mask(self, mask, side):
        points = [shape[i] for i in side]
        points = np.array(points, dtype=np.int32)
        mask = cv2.fillConvexPoly(mask, points, 255)
        return mask

    def contouring(self,thresh, mid, img, right=False):
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
    def img_cutting(self, img, rect):
        result = img[rect.right()- rect.left(), rect.bottom()- rect.top()]

HL = Haar_Like()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../assets/shape_predictor_68_face_landmarks.dat')
LEFT = list(range(36, 42))
RIGHT = list(range(42, 48))

cap = cv2.VideoCapture(0)
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
            shape = HL.shape_to_np(shape)
            mask = np.zeros((rect.right()- rect.left(), rect.bottom()- rect.top()), dtype=np.uint8)
            mask = HL.eye_on_mask(mask, LEFT)
            mask = HL.eye_on_mask(mask, RIGHT)
            mask = cv2.dilate(mask, kernel, 5)
            eyes = cv2.bitwise_and(img, img, mask=mask)
            cv2.imshow("img", img)
            cv2.imshow("mask", mask)
            cv2.imshow("eyes", eyes)






    if cv2.waitKey(1) & 0xFF == ord('q'):
        break