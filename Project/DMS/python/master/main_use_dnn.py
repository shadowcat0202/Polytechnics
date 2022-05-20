import cv2
import dlib
import numpy as np
import time
from functools import wraps

from master.mark_detector import MarkDetector
from master.pose_estimator import PoseEstimator

print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
SKYBLUE = (255, 255, 0)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

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
MARK_INDEX = RIGHT_EYE + LEFT_EYE + MOUTH_INLINE

lastsave = 0


# (두 점 사이의 유클리드 거리 계산)
def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1 / 2)


# 눈 비율 값 계산
def ER_ratio(eye_point):
    # 얼굴 특징점 번호 사진 참조
    A = distance(eye_point[1], eye_point[5])
    B = distance(eye_point[2], eye_point[4])
    C = distance(eye_point[0], eye_point[3])
    return (A + B) / (2.0 * C)


def eye_close(left_eye, right_eye):
    left_ER = ER_ratio(left_eye)
    right_ER = ER_ratio(right_eye)

    avg = round((left_ER + right_ER) / 2, 2)
    cv2.putText(img, f"{avg}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
    return avg < 0.3


# https://wjh2307.tistory.com/21
# 눈 감기(함수) + 자는거 판단(밑에 코드)
def close_counter(func):
    @wraps(func)
    def tmp(*args, **kwargs):
        tmp.count += 1
        global lastsave
        # print(f"during close: {time.time() - lastsave}")
        if time.time() - lastsave > 5:
            lastsave = time.time()
            tmp.count = 0
        return func(*args, **kwargs)

    tmp.count = 0
    return tmp


@close_counter
def close():
    cv2.putText(img, "close", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)


# detection한 네모중에 가장 큰(카메라와 가장 가깝다고 판단) 얼굴을 찾아낸다
def front_detection(squares):
    most_front_face_index = 0
    max_size_area = 0
    for i, sq in enumerate(squares):
        curr_area = (sq.right() - sq.left()) * (sq.bottom() - sq.top())
        if curr_area > max_size_area:
            max_size_area = curr_area
            most_front_face_index = i
    return most_front_face_index


clahe = cv2.createCLAHE(clipLimit=2.0,
                        tileGridSize=(8, 8))


def img_Preprocessing(img_frame):
    re_size = cv2.resize(img_frame, (400, 400), cv2.INTER_AREA)
    gray = cv2.cvtColor(re_size, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(re_size, cv2.COLOR_BGR2LAB)

    gray = clahe.apply(gray)
    # lab = clahe.apply(lab)

    L = lab[:, :, 0]
    med_L = cv2.medianBlur(L, 99)  # median filter  # 뭔지 모르겠음
    invert_L = cv2.bitwise_not(med_L)  # invert lightness   # 빛 제거
    composed = cv2.addWeighted(gray, 0.75, invert_L, 0.25, 0)
    return re_size, composed


face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("../assets/shape_predictor_68_face_landmarks.dat")
print("stub loading facial landmark predictor...")
# video_capture = cv2.VideoCapture("./branch_test1.mp4")  # 사진
video_capture = cv2.VideoCapture(0)  # 카메라

width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

pose_estimator = PoseEstimator(img_size=(height, width))

# 3. Introduce a mark detector to detect landmarks.
mark_detector = MarkDetector()

# 4. Measure the performance with a tick meter.
tm = cv2.TickMeter()

if video_capture.isOpened():
    print("camera is ready")

    while True:
        start_t = time.time()
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        ret, img = video_capture.read()

        img = cv2.flip(img, 1)

        """
        cv2.resize((fx,fy),interpilation )
        1. cv2.INTER_NEAREST - 최근방 이웃 보간법
         가장 빠르지만 퀄리티가 많이 떨어집니다. 따라서 잘 쓰이지 않습니다.     
        2. cv2.INTER_LINEAR - 양선형 보간법(2x2 이웃 픽셀 참조)
         4개의 픽셀을 이용합니다.
         효율성이 가장 좋습니다. 속도도 빠르고 퀄리티도 적당합니다.

        3. cv2.INTER_CUBIC - 3차회선 보간법(4x4 이웃 픽셀 참조)
         16개의 픽셀을 이용합니다.
         cv2.INTER_LINEAR 보다 느리지만 퀄리티는 더 좋습니다.        
        4. cv2.INTER_LANCZOS4 - Lanczos 보간법 (8x8 이웃 픽셀 참조)
         64개의 픽셀을 이용합니다.
         좀더 복잡해서 오래 걸리지만 퀄리티는 좋습니다.         
        5. cv2.INTER_AREA - 영상 축소시 효과적
         영역적인 정보를 추출해서 결과 영상을 셋팅합니다.
         영상을 축소할 때 이용합니다."""
        img, cmpos = img_Preprocessing(img)
        detection = face_detector(cmpos, 0)

        if detection:
            d_index = 0
            if len(detection) >= 2:  # 인식된 얼굴이 2명 이상인 경우
                d_index = front_detection(detection)  # 카메라에서 가장 가까운 얼굴 찾기
            print(type(detection[d_index]))
            shape = shape_predictor(cmpos, detection[d_index])  # 그 얼굴에서 랜드마크 추출
            landmarks = list([p.x, p.y] for p in shape.parts())  # 리스트화
            landmarks = np.array(landmarks, dtype=np.float32)  # 파라미터 타입은 numpy.ndarray으로 해야함
            # 추가로 solve_pose_by_68_points 내부 함수 중 cv2.solvePnP의 인자중 랜드마크는 np.float32로 해주어야 한다
            pose = pose_estimator.solve_pose_by_68_points(landmarks)

            # # 육면체를 보고 싶다면?
            pose_estimator.draw_annotation_box(img, pose[0], pose[1], color=GREEN)

            # # x, y, z 축 보고 싶다면?
            axis = pose_estimator.get_axis(img, pose[0], pose[1]) # 축에 대한 데이터를 쓰고 싶은데 어디있는지 모름 ㅎ
            # axis = [[[RED_x RED_y]],[[GREEN_x GREEN_y]],[[BLUE_x BLUE_y]],[[CENTER_x CENTER_y]]]
            # --> BLUE(정면) GREEN(아래) RED(좌측) CENTER(중심)
            pose_estimator.draw_axes(img, pose[0], pose[1])
            # 얼굴 랜드마크 보고 싶다면?
            mark_detector.draw_marks(img, landmarks[MARK_INDEX], color=GREEN)

            # 얼굴 detect box 보고 싶다면?
            # detection_box = [detection[d_index].left(), detection[d_index].top(),
            #                  detection[d_index].right(), detection[d_index].bottom()]
            # mark_detector.draw_box(img, [detection_box], box_color=BLUE)

            eye_close_status = eye_close(landmarks[42:48], landmarks[36:42])
            if eye_close_status:
                close()
                # print(f'close count : {close.count}')
                if close.count >= 15:
                    # waring_flag += 1
                    cv2.putText(img, "SLEEPING!!!", (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

            # for i in INDEX:
            #     cv2.circle(img, (landmarks[i][0], landmarks[i][1]), 1, GREEN, -1)

        cv2.putText(img, f"FPS:{int(1. / (time.time() - start_t))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

        cv2.imshow("img", img)

cv2.destroyAllWindows()
video_capture.release()
