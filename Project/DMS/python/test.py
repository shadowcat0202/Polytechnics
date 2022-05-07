# 학습 모델 다운로드
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2   반디집으로 풀 수 있다
import functools

import cv2, dlib
import numpy as np
import time  # 시간 측정
now_status = "Active"
status = {"sleep": 0, "drowsy": 0, "active": 0}
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
NOTHING = list(range(0, 0))




# (두 점 사이의 유클리드 거리 계산)
def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1 / 2)


# 양쪽 눈의 비율을 이용해 눈 감김을 확인
def calculate_EAR(eye):
    # 얼굴 특징점 번호 사진 참조
    A = distance(eye[1], eye[5])
    B = distance(eye[2], eye[4])
    C = distance(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def light_removing(frame) :
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    med_L = cv2.medianBlur(L, 99)   # median filter
    invert_L = cv2.bitwise_not(med_L)   # invert lightness
    composed = cv2.addWeighted(gray, 0.75, invert_L, 0.25, 0)
    return L, composed


def counter(func):
    @functools.wraps(func)
    def tmp(*args, **kwargs):
        lastsave = time.time()
        print(f"tmp 도는 중 lastsave:{lastsave}")
        tmp.count += 1
        time.sleep(0.05)
        if time.time() - lastsave > 5.0:
            print("time.time() - lastsave에 들어옴")
            lastsave = time.time()
            tmp.count = 0
        return func(*args, **kwargs)

    tmp.count = 0
    return tmp


@counter
def close():
    cv2.putText(img_frame, "CLOSE", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, RED, 4)


# https://github.com/infoaryan/Driver-Drowsiness-Detection/blob/master/driver_drowsiness.py
def eyeStatusCheck(left, right, st):
    if left == 0 or right == 0:
        status["sleep"] += 1
        status["drowsy"] = 0
        status["active"] = 0
        if status["sleep"] > 5.0:
            st = "Sleeping"
    elif left == 1 or right == 1:
        status["sleep"] = 0
        status["drowsy"] += 1
        status["active"] = 0
        if status["drowsy"] > 5.0:
            st = "Drowsy"
    else:
        status["sleep"] = 0
        status["drowsy"] = 0
        status["active"] += 1
        if status["active"] > 2.0:
            st= "Active"

    cv2.putText(img_frame, st, (face.left(), face.top()), cv2.FONT_HERSHEY_SIMPLEX, 1.2, RED, 2)

# start ==================================================================================
print("loading facial landmark predictor...")
# dlib에서 기본적으로 제공하는 face_object_detector 객체 생성
detector = dlib.get_frontal_face_detector()
# 얼굴에 랜드마크를 찍기위해서 shape_predictor_68_face_landmarks.dat를 불러온다
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

# 동영상 or 카메라
video = cv2.VideoCapture('./dataset/move_close_test.mp4')
# cam = cv2.VideoCapture(0)   # 나중에 카메라로 셋팅하고 싶다면 이렇게

# 영상 크기 변환 -> 작동이 안됨
# print(f"bf{video.get(cv2.CAP_PROP_FRAME_WIDTH)}X{video.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
# 
# video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# print(f"af{video.get(cv2.CAP_PROP_FRAME_WIDTH)}X{video.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

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

index = ALL    # 초기값
# https://github.com/woorimlee/drowsiness-detection # 좀더 자세한 부분
while True:
    ret, img_frame = video.read()  # 동영상 or 웹캠을 프레임 단위로 자름
    # 읽을 프레임이 없는 경우 종료
    if not ret: break

    # 입력받은 영상으로부터 gray 스케일로 변환
    # img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    L, gray = light_removing(img_frame)

    # resize : 이미지 크기 변환
    # 1) 변환할 이미지
    # 2) 변환할 이미지 크기(가로, 세로)
    # - interpolation : 보간법 지정
    #   - 보간법 : 알려진 데이터 지점 내에서 새로운 데이터 지점을 구성하는 방식
    #   - cv2.INTER_NEAREST : 최근방 이웃 보간법
    #   - cv2.INTER_LINEAR(default) : 양선형 보간법(2x2 이웃 픽셀 참조)
    #   - cv2.INTER_CUBIC : 3차 회선 보간법(4x4 이웃 픽셀 참조)
    #   - cv2.INTER_LANCZOS4 : Lanczos 보간법(8x8 이웃 픽셀 참조)
    #   - cv2.INTER_AREA : 픽셀 영역 관계를 이용한 resampling 방법으로 이미지 축소시 효과적
    # img_gray = cv2.resize(img_gray, (720, 720), interpolation=cv2.INTER_CUBIC)


    # 업 셈플링 횟수(이미지를 증가?)
    faces = detector(gray, 0)  # detector(img_gray) 도 사용 가능

    # 검출된 얼굴 갯수만큼 반복
    for face in faces:
        shape = predictor(gray, face)  # 얼굴에서 68개 점 찾기

        # 검출된 랜드마크를 리스트에 저장
        list_points = []
        for p in shape.parts():
            list_points.append([p.x, p.y])

        list_points = np.array(list_points)

        # cv2.imshow("result", img_frame)

        left_EAR = calculate_EAR(list_points[LEFT_EYE])
        right_EAR = calculate_EAR(list_points[RIGHT_EYE])


        # version 1==================================================
        # EAR = (left_EAR + right_EAR) / 2
        # EAR = round(EAR, 2)
        # if EAR < 0.19:
        #     print(EAR)
        #     close()
        #     # print(f"close count:{close.count}")
        #     if close.count == 15:
        #         print("Driver is sleeping")

        # version 2==================================================
        # print(f"{left_EAR}, {right_EAR}")
        eyeStatusCheck(left_EAR, right_EAR, now_status)

        key = cv2.waitKey(1)
        if key == ord('q'):
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
        elif key == ord("7"):
            index = NOTHING

        # 아래 3줄은 중간에 멈춰서 보기 위한 라인
        # print(calculate_EAR(list_points[index]))
        # if input() == 'a':
        #     continue

        # 점 찍어주기
        for i, pt in enumerate(list_points[index]):
            pt_pos = (pt[0], pt[1])
            cv2.circle(img_frame, pt_pos, 1, GREEN, -1)

        # 얼굴 네모
        cv2.rectangle(img_frame, (face.left(), face.top()), (face.right(), face.bottom()), GREEN, 1)

        cv2.imshow("test", img_frame)

video.release()
cv2.destroyAllWindows()
