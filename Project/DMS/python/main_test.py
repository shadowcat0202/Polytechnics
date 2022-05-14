import pprint
import timeit

import cv2
import dlib
import numpy as np
import time

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
LANDMARK_INDEX = RIGHT_EYE + LEFT_EYE
# LANDMARK_INDEX = ALL

eye_close_record = [0, 0]
eye_record_size = 50

ER_cnt = 0
ER_max = [0, 0]
ER_avg = [0, 0]
ER_sum = [0, 0]

eye_state = 1


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


def rotate(brx, bry):
    crx = brx - middle_x
    cry = bry - middle_y
    arx = np.cos(-angle) * crx - np.sin(-angle) * cry
    ary = np.sin(-angle) * crx + np.cos(-angle) * cry
    rx = int(arx + middle_x)
    ry = int(ary + middle_y)
    return rx, ry


def sleep_check(eyes_ER, cnt, st):
    for i, eye in enumerate(eyes_ER):
        if eye < ER_max[i] * 0.8:  # 감았다고 판단 하면 비율 뭘로 할건지 정해야함
            if eye_close_record[i] == eye_record_size:
                pass  # 예외처리
            else:
                eye_close_record[i] += 1
        else:
            if eye_close_record[i] == 0:
                pass
            else:
                eye_close_record[i] -= 1
    check_state = (sum(eye_close_record) / 2) / eye_record_size
    if check_state > 0.7:
        cv2.putText(img, "SLEEP!!", (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
        st = 0
    elif check_state > 0.4:
        cv2.putText(img, "DROWSY!", (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
        st = 1
    else:
        cv2.putText(img, "ACTIVE", (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
        st = 2

    ER_sum[0] += ER_left
    ER_sum[1] += ER_right
    if cnt == 30:
        for i in range(2):
            ER_avg[i] = round((ER_avg[i] + ER_sum[i]) / (ER_cnt + 1), 3)
            if ER_avg[i] > ER_max[i]:
                ER_max[i] = ER_avg[i]
            else:
                if st != 0:  # 눈을 감은 상태가 길어질수록 최대 값이 감소하면서 눈을 뜨는 걸로 판단해버림 이것을 방지 하기위해서
                    ER_max[i] = (ER_avg[i] + ER_max[i]) * 0.5
            ER_sum[i] = 0
        return 0, st
    return cnt + 1, st


face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
print("stub loading facial landmark predictor...")
video_capture = cv2.VideoCapture("./test1.mp4")  # 사진
# video_capture = cv2.VideoCapture(0)  # 카메라

if video_capture.isOpened():
    print("camera is ready")
    while True:
        start_t = time.time()
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        ret, img = video_capture.read()
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

        img = cv2.resize(img, (500, 500), cv2.INTER_AREA)
        img = cv2.flip(img, 1)  # cv2.flip(frame, [0 | 1]) 0 상하, 1 좌우 반전
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0,
                                tileGridSize=(8, 8))
        clahe_image = clahe.apply(gray)
        detection = face_detector(clahe_image, 0)
        # print(detection)

        if detection:
            for d in detection:  # 얼굴을 감지한것이 여러명일 경우도 있기때문에 for문으로 작성
                shape = shape_predictor(clahe_image, d)
                landmarks = list([p.x, p.y] for p in shape.parts())

                # 얼굴 점 찍어주고 싶다면
                # for pos in landmarks[LANDMARK_INDEX]:
                #     cv2.circle(img, (pos[0], pos[1]), 2, RED, -1)

                # 감지한 얼굴 좌표
                d_x1 = d.left()
                d_y1 = d.top()
                d_x2 = d.right()
                d_y2 = d.bottom()
                # 여유 공간 확보
                border_x1 = d_x1 - (d_x2 - d_x1) / 2
                border_y1 = d_y1 - (d_y2 - d_y1) / 2
                border_x2 = d_x2 + (d_x2 - d_x1) / 2
                border_y2 = d_y2 + (d_y2 - d_y1) / 2

                # 감지한 얼굴 상자에서 중앙 좌표 계산
                center_x = (d_x1 + d_x2) / 2
                center_y = (d_y1 + d_y2) / 2

                # 양 눈의 끝점
                right_eye_x = shape.part(45).x
                right_eye_y = shape.part(45).y
                left_eye_x = shape.part(36).x
                left_eye_y = shape.part(36).y

                # 양 끝점을 활용한 눈 사이의 점
                middle_x = int(left_eye_x + (right_eye_x - left_eye_x) / 2)
                middle_y = int(left_eye_y + (right_eye_y - left_eye_y) / 2)

                tan = (left_eye_y - middle_y) / (middle_x - left_eye_x)
                angle = np.arctan(tan)

                rd1 = rotate(d_x1, d_y1)
                rd2 = rotate(d_x2, d_y1)
                rd3 = rotate(d_x1, d_y2)
                rd4 = rotate(d_x2, d_y2)
                d2b_1 = rotate(border_x1, border_y1)
                d2b_2 = rotate(border_x2, border_y1)
                d2b_3 = rotate(border_x1, border_y2)
                d2b_4 = rotate(border_x2, border_y2)

                # 얼굴 detect한걸로 하니까 여유 공간 없어서 인식 불가
                # pts1 = np.float32([[rd1[0], rd1[1]], [rd2[0], rd2[1]], [rd3[0], rd3[1]],[rd4[0], rd4[1]]])
                pts1 = np.float32(
                    [[d2b_1[0], d2b_1[1]], [d2b_2[0], d2b_2[1]], [d2b_3[0], d2b_3[1]], [d2b_4[0], d2b_4[1]]])
                pts2 = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst_frame = cv2.warpPerspective(img, M, (
                    400, 400))  # 원근 변환 cv2.warpPerspective(origin_frame, 변환 프레임, (width, height))
                gray2 = cv2.cvtColor(dst_frame, cv2.COLOR_RGB2GRAY)
                clahe_image2 = clahe.apply(gray2)
                detection2 = face_detector(clahe_image2)
                # 재 정렬한 이미지에서 판단
                for d2 in detection2:
                    xx1 = d2.left()
                    yy1 = d2.top()
                    xx2 = d2.right()
                    yy2 = d2.bottom()

                    shape2 = shape_predictor(clahe_image2, d2)
                    cv2.rectangle(dst_frame, (xx1, yy1), (xx2, yy2), GREEN, 1)
                    landmarks2 = np.array(list([p.x, p.y] for p in shape2.parts()))

                    ER_left = ER_ratio(landmarks2[LEFT_EYE])
                    ER_right = ER_ratio(landmarks2[RIGHT_EYE])
                    ER_cnt, eye_state = sleep_check([ER_left, ER_right], ER_cnt, eye_state)

                    for p in landmarks2[LANDMARK_INDEX]:
                        cv2.circle(dst_frame, (p[0], p[1]), 2, RED, -1)

                    cv2.putText(img, f"cnt:{ER_cnt}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
                    for i, ea in enumerate(ER_avg):
                        cv2.putText(img, f"avg[{i}]:{round(ER_avg[i], 2)}", (10, 120 + i * 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
                    for i, erm in enumerate(ER_max):
                        cv2.putText(img, f"max[{i}]:{round(ER_max[i], 2)}", (10, 160 + i * 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

                    rex = shape2.part(45).x
                    rey = shape2.part(45).y
                    lex = shape2.part(36).x
                    ley = shape2.part(36).y

                    # for p in landmarks2[LANDMARK_INDEX]:
                    #     cv2.circle(dst_frame, (p[0], p[1]), 2, RED, -1)

                    cv2.putText(img, f"ER_L:{round(ER_left * 100, 2)}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED,
                                2)
                    cv2.putText(img, f"ER_R:{round(ER_right * 100, 2)}%", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED,
                                2)
                cv2.imshow("dst", dst_frame)
        cv2.putText(img, f"FPS:{int(1. / (time.time() - start_t))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
        cv2.imshow("test", img)
        # cv2.imshow("gray", gray)

cv2.destroyAllWindows()
video_capture.release()
