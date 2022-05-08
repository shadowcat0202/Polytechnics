# http://www.ntrexgo.com/archives/36038
import random

import cv2, dlib
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

# 100프레임을 미리 저장해서 눈의 비율의 최대치등을 계산
ER_array_ready = False
ER_array_ready_size = 100
left_eye_ER_array = []
right_eye_ER_array = []
MAX_ER_left = 0
MAX_ER_right = 0
eye_ratio_limit = 0.00

count_time = [0, 0]
program_switch = False

open_eye = False
eye_close_count = 0
driving_state_step = [15, 35]


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


# 얼굴이 기울었을때 인식률을 높이기 위한 detection_img의 재 정렬을 위한 좌표 계산 함수
def rotate(brx, bry):
    crx = brx - midx
    cry = bry - midy
    arx = np.cos(-angle) * crx - np.sin(-angle) * cry
    ary = np.sin(-angle) * crx + np.cos(-angle) * cry
    rx = int(arx + midx)
    ry = int(ary + midy)
    return rx, ry


# 눈 비율 계산
def calculate_EAR(eye):
    # 얼굴 특징점 번호 사진 참조
    A = distance(eye[1], eye[5])
    B = distance(eye[2], eye[4])
    C = distance(eye[0], eye[3])
    return (A + B) / (2.0 * C)


face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
print("stub loading facial landmark predictor...")
video_capture = cv2.VideoCapture(0)  # 카메라
index = NOTHING

if video_capture.isOpened():
    print("camera is ready")
    while True:
        ret, frame = video_capture.read()
        if not ret: break
        
        # 얼굴 랜드마크 종류별로 변경
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

        frame = cv2.flip(frame, 1)  # cv2.flip(frame, [0 | 1]) 0 상하, 1 좌우 반전
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0,
                                tileGridSize=(8, 8))
        clahe_image = clahe.apply(gray)
        detection = face_detector(clahe_image, 0)

        if detection:  # 얼굴을 detection 했을때
            for d in detection:
                shape = shape_predictor(clahe_image, d)
                landmarks = np.array(list([p.x, p.y] for p in shape.parts()))

                # 점 or (선 그어주기 미완선)
                # for i, pt in enumerate(landmarks[index]):
                #     cv2.circle(frame, (pt[0], pt[1]), 1, GREEN, -1)
                #     # 선으로 하고 싶은데 모르겠다
                #     # bf = i % len(landmarks[index])
                #     # af = (i + 1) % len(landmarks[index])
                #     # cv2.line(frame,
                #     #          (pt[bf][0], pt[bf][1]),
                #     #          (pt[af][0], pt[af][1]),
                #     #          GREEN, 1)
                # 점 찍어주기
                for pos in landmarks[index]:
                    cv2.circle(frame, (pos[0], pos[1]), 1, GREEN, -1)

                left_eye_landmark = landmarks[LEFT_EYE]
                right_eye_landmark = landmarks[RIGHT_EYE]

                # ER_array, eye_ratio_limit 갱신
                left_eye_ER_array.append(ER_ratio(left_eye_landmark))
                right_eye_ER_array.append(ER_ratio(right_eye_landmark))
                if ER_array_ready:
                    # 갱신하는 방법1. 머리좀 굴려본다
                    # if left_eye_ER_array[0] == MAX_ER_left:
                    #     del left_eye_ER_array[0]
                    #     MAX_ER_left = max(left_eye_ER_array)
                    # else:
                    #     del left_eye_ER_array[0]
                    #     if left_eye_ER_array[-1] > MAX_ER_left:
                    #         MAX_ER_left = left_eye_ER_array[-1]
                    #
                    # if right_eye_ER_array[0] == MAX_ER_right:
                    #     del right_eye_ER_array[0]
                    #     MAX_ER_right = max(right_eye_ER_array)
                    # else:
                    #     del right_eye_ER_array[0]
                    #     if right_eye_ER_array[-1] > MAX_ER_right:
                    #         MAX_ER_right = right_eye_ER_array[-1]
                    # 갱신하는 방법2. 막해본다
                    del left_eye_ER_array[0]
                    del right_eye_ER_array[0]
                    MAX_ER_left = max(left_eye_ER_array)
                    MAX_ER_right = max(right_eye_ER_array)
                    eye_ratio_limit = (MAX_ER_left + MAX_ER_right) / 2 * 0.65  # 이 수치는 어떤식으로 구했는지는 모름

                else:
                    if len(left_eye_ER_array) == ER_array_ready_size:
                        ER_array_ready = True
                        print("ER data ready!")

                # face detection square 값 저장
                x = d.left()
                y = d.top()
                x1 = d.right()
                y1 = d.bottom()
                # face detection square border 값 저장(여유 공간 좌표)
                bdx = x - (x1 - x) / 2
                bdy = y - (y1 - y) / 2
                bdx1 = x1 + (x1 - x) / 2
                bdy1 = y1 + (y1 - y) / 2
                # face detection square의 가운데 값
                midx = (x + x1) / 2
                midy = (y + y1) / 2

                # 눈의 양 끝점
                rex = shape.part(45).x
                rey = shape.part(45).y
                lex = shape.part(36).x
                ley = shape.part(36).y

                # 눈의 양끝점 좌표 설정 및 눈 사이 가운데 점 설정
                mex = int(lex + (rex - lex) / 2)
                mey = int(ley + (rey - ley) / 2)
                # tan 값 계산
                tan = (ley - mey) / (mex - lex)
                # 각도 계산
                angle = np.arctan(tan)
                degree = np.degrees(angle)

                # detection 좌표 회전
                rsd_1 = rotate(x, y)
                rsd_2 = rotate(x1, y)
                rsd_3 = rotate(x, y1)
                rsd_4 = rotate(x1, y1)
                # border 좌표 회전
                d2_1 = rotate(bdx, bdy)
                d2_2 = rotate(bdx1, bdy)
                d2_3 = rotate(bdx, bdy1)
                d2_4 = rotate(bdx1, bdy1)

                # 기존 좌표를 회전된 좌표에 매칭, 새로운 창으로 프린트
                pts1 = np.float32([[d2_1[0], d2_1[1]], [d2_2[0], d2_2[1]], [d2_3[0], d2_3[1]], [d2_4[0], d2_4[1]]])
                pts2 = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                # 원근 변환 cv2.warpPerspective(origin_frame, 변환 프레임, (width, height))
                dst_frame = cv2.warpPerspective(frame, M, (400, 400))

                d2gray = cv2.cvtColor(dst_frame, cv2.COLOR_RGB2GRAY)
                d2clahe_image = clahe.apply(d2gray)
                d2detection = face_detector(d2clahe_image)

                # 재 정렬한 이미지에서 판단
                if d2detection:
                    for d2 in d2detection:
                        xx = d2.left()
                        yy = d2.top()
                        xx1 = d2.right()
                        yy1 = d2.bottom()

                        # 얼굴 랜드마크 찾기
                        d2shape = shape_predictor(d2clahe_image, d2)
                        cv2.rectangle(dst_frame, (xx, yy), (xx1, yy1), GREEN, 1)
                        d2landmarks = np.array(list([p.x, p.y] for p in d2shape.parts()))

                        ER_left = ER_ratio(d2landmarks[LEFT_EYE])
                        ER_right = ER_ratio(d2landmarks[RIGHT_EYE])
                        # limit 비율로 측정 눈감으면 open_eye = False 뜨면 True
                        if ER_left <= eye_ratio_limit and ER_right <= eye_ratio_limit:
                            open_eye = False
                        if ER_left > eye_ratio_limit and ER_right > eye_ratio_limit:
                            open_eye = True

                        # 1. time()을 가지고 비교 하는 방법
                        # if open_eye:
                        #     count_time[0] = time.time()
                        # if time.time() - count_time[0] > 2.5:
                        #     cv2.putText(dst_frame, "SLEEP!!", (xx, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

                        # 2. 빈도수(eye_close_count)를 가지고 비율을 계산하는 방법
                        if ER_array_ready:
                            if open_eye:
                                if eye_close_count > 0:
                                    eye_close_count -= 1
                            else:
                                if eye_close_count <= 50:
                                    eye_close_count += 1
                                    cv2.putText(dst_frame, "close", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
                            # print(f"close_count:{eye_close_count}")
                        
                        # 졸음의 정도를  driving_state_step 수치를 조정해야한다 [0]활성화 임계점 [1]자는임계점
                        if eye_close_count >= driving_state_step[1]:
                            cv2.putText(dst_frame, "SLEEP!!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
                        elif driving_state_step[0] < eye_close_count < driving_state_step[1]:
                            cv2.putText(dst_frame, "DROWSY", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
                        else:
                            cv2.putText(dst_frame, "ACTIVE", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

                    cv2.imshow("dst_frame", dst_frame)

                # 수치 보고 싶을때
                cv2.putText(frame, "left:{:.2f}".format(MAX_ER_left), (450, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, RED, 2)
                cv2.putText(frame, "right:{:.2f}".format(MAX_ER_right), (450, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, RED, 2)
                cv2.putText(frame, "limit:{:.2f}".format(eye_ratio_limit), (450, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, RED, 2)
                # cv2.putText(frame, "degree:{:.2f}".format(degree), (10, 430), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.7, RED, 2)

        else:  # 얼굴을 detection 못했을때
            pass
        # 점 찍어주기

        cv2.imshow("test", frame)

cv2.destroyAllWindows()
video_capture.release()
