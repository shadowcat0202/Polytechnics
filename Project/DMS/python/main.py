# 학습 모델 다운로드
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2   반디집으로 풀 수 있다
import functools

import cv2, dlib
import numpy as np
import time  # 시간 측정

file_path = './dataset/close_test1.mp4'

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INLINE = list(range(51, 68))
FACE_OUTLINE = list(range(0, 17))


# (유클리드 거리 계산)
def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1 / 2)


# 눈의 비율을 이용해 눈 감김을 확인
def calculate_EAR(eye):
    # 얼굴 특징점 번호 사진 참조
    A = distance(eye[1], eye[5])
    B = distance(eye[2], eye[4])
    C = distance(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def counter(func):
    @functools.wraps(func)
    def tmp(*args, **kwargs):
        tmp.count += 1
        time.sleep(0.05)
        lastsave = time.time()
        if time.time() - lastsave > 5.0:
            lastsave = time.time()
            tmp.count = 0
        return func(*args, **kwargs)

    tmp.count = 0
    return tmp


@counter
def close():
    cv2.putText(img_frame, "CLOSE", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, RED, 4)


# dlib에서 기본적으로 제공하는 face_object_detector 객체 생성
detector = dlib.get_frontal_face_detector()
# 얼굴에 랜드마크를 찍기위해서 shape_predictor_68_face_landmarks.dat를 불러온다
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

# 동영상 or 카메라
video = cv2.VideoCapture(file_path)
# cam = cv2.VideoCapture(0)   # 나중에 카메라로 셋팅하고 싶다면 이렇게

# while True:
#     img, frame = cam.read()
#     face = detactor(frame)
#
#     # dectector는 얼굴의 어떤 위치 순서대로 반환? 이거 뭐지?
#     for f in face:
#         cv2.rectangle(frame, (f.left(), f.top()), (f.right(), f.bottom()), (0, 0, 255), 2)
#         land = sp(frame, f)
#         land_list = []
#         print(land)
#
#         for l in land.parts():
#             land_list.append([l.x, l.y])
#             cv2.circle(frame, (l.x, l.y), 3, (255, 0, 0), -1)
#
#     cv2.imshow("face diectactor", frame)
#
#     if cv2.waitKey(1) == 'q':
#         break

index = LEFT_EYE + RIGHT_EYE

while True:
    ret, img_frame = video.read()
    # 입력받은 영상으로부터 gray 스케일로 변환
    img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)

    # 업 셈플링 횟수(이미지를 확대)
    faces = detector(img_gray, 1)  # detector(img_gray) 도 사용 가능

    # 검출된 얼굴 갯수만큼 반복
    for face in faces:
        shape = predictor(img_frame, face)  # 얼굴에서 68개 점 찾기

        # 검출된 랜드마크를 리스트에 저장
        list_points = []
        for p in shape.parts():
            list_points.append([p.x, p.y])

        list_points = np.array(list_points)

        for i, pt in enumerate(list_points[index]):
            pt_pos = (pt[0], pt[1])
            cv2.circle(img_frame, pt_pos, 2, GREEN, -1)

        cv2.rectangle(img_frame, (face.left(), face.top()), (face.right(), face.bottom()), BLUE, 3)

        # cv2.imshow("result", img_frame)

        left_EAR = calculate_EAR(list_points[LEFT_EYE])
        right_EAR = calculate_EAR(list_points[RIGHT_EYE])

        EAR = (left_EAR + right_EAR) / 2
        EAR = round(EAR, 2)

        if EAR < 0.19:
            print(EAR)
            close()
            # print(f"close count:{close.count}")
            if close.count == 15:
                print("Driver is sleeping")

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord("1"):
            index = ALL
        elif key == ord("2"):
            index = LEFT_EYEBROW + RIGHT_EYEBROW
        elif key == ord("3"):
            index = LEFT_EYE + RIGHT_EYE
        elif key == ord("4"):
            index = NOSE
        elif key == ord("5"):
            index = MOUTH_OUTLINE + MOUTH_INLINE
        elif key == ord("6"):
            index = FACE_OUTLINE

        # 아래 3줄은 중간에 멈춰서 보기 위한 라인
        # print(calculate_EAR(list_points[index]))
        # if input() == 'a':
        #     continue
        cv2.imshow("test", img_frame)


video.release()
cv2.destroyAllWindows()
