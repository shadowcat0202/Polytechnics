# 학습 모델 다운로드
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2   반디집으로 풀 수 있다

import functools
import cv2, dlib
import numpy as np

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


# 눈 비율 값 계산
def ER_ratio(eye_point):
    # 얼굴 특징점 번호 사진 참조
    A = distance(eye_point[1], eye_point[5])
    B = distance(eye_point[2], eye_point[4])
    C = distance(eye_point[0], eye_point[3])
    return (A + B) / (2.0 * C)


# 얼굴이 돌아갔을때 인식불가 방지
def rotate(brx, bry):
    crx = brx - midx
    cry = bry - midy
    arx = np.cos(-angle) * crx - np.sin(-angle) * cry
    ary = np.sin(-angle) * crx + np.cos(-angle) * cry
    rx = int(arx + midx)
    ry = int(ary + midy)
    return (rx, ry)


# 눈 비율 계산
def calculate_EAR(eye):
    # 얼굴 특징점 번호 사진 참조
    A = distance(eye[1], eye[5])
    B = distance(eye[2], eye[4])
    C = distance(eye[0], eye[3])
    return (A + B) / (2.0 * C)


face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
print("loading facial landmark predictor...")
video_capture = cv2.VideoCapture(0)  # 카메라
# video_capture = cv2.VideoCapture("./dataset/close_test2.mp4") # 동영상

# http://www.ntrexgo.com/archives/36038
# main start ==================================================================
if video_capture.isOpened():
    print("video_capture is ready")
    while True:
        ret, frame = video_capture.read()  # 동영상 or 웹캠을 프레임 단위로 자름
        if not ret: break
        if cv2.waitKey(1) and 0xFF == ord('q'): break  # cv2.watiKey(1) 없으면(종료 하는 부분 없으면) cv2.imshow() 안나옴

        # frame = cv2.flip(frame, 1)  # cv2.flip(frame, [1(좌우) | 0(상하)])   # 지금은 딱히 필요 없으니 패스
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 흑백 전환
        # createCLAHE를 사용함으로서 정확한 명도 변환
        clahe = cv2.createCLAHE(clipLimit=2.0,
                                tileGridSize=(8, 8))  # CLAHE 객체 생성 # https://m.blog.naver.com/samsjang/220543360864
        clahe_image = clahe.apply(gray)  # 밝기가(0~255)를 균등하게 분포하도록 변경 후 적용
        detection = face_detector(clahe_image)

        for d in detection:
            shape = shape_predictor(clahe_image, d)  # 가공된 프레임에서 얼굴 랜드마크 작성
            # 랜드마크 추출
            # landmarks = []
            # for p in shape.parts():
            #     landmarks.append([p.x, p.y])
            # landmarks = np.array(landmarks)

            # 랜드마크 점 그리기
            # for pt in landmarks:
            #     landmark = (pt[0], pt[1])
            #     cv2.circle(frame, landmark, 2, GREEN, -1)

            # cv2.rectangle(원하는frame, (p1.x, p1.y), (p2.x, p2.y), 색상, 두께)
            # cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), GREEN, 1)
            # cv2.imshow("frame", frame)

            # left_eye = landmarks[LEFT_EYE]
            # right_eye = landmarks[RIGHT_EYE]
            #
            # left_eye_ER = ER_ratio(left_eye)
            # right_eye_ER = ER_ratio(right_eye)

            # 얼굴을 인식한 사각형의 테두리 좌표
            x = d.left()
            y = d.top()
            x1 = d.right()
            y1 = d.bottom()

            # 얼굴 박스보다 더 넓은 박스
            bdx = x - (x1 - x) / 2
            bdy = y - (y1 - y) / 2
            bdx1 = x1 + (x1 - x) / 2
            bdy1 = y1 + (y1 - y) / 2

            # 박스에서 가운데 점
            midx = (x + x1) / 2
            midy = (y + y1) / 2

            # cv2.rectangle(frame, (int(bdx), int(bdy)), (int(bdx1), int(bdy1)), RED, 1)
            # cv2.imshow("big", frame)
            # 눈 양 끝점 좌표
            left_eye_x = shape.part(36).x
            left_eye_y = shape.part(36).y
            right_eye_x = shape.part(45).x
            right_eye_y = shape.part(45).y

            # 양 끝점 선 긋기(영상을 보고 싶을때 하는 라인)
            # cv2.line(frame, (left_eye_x, left_eye_y),(right_eye_x, right_eye_y), RED, 2)
            # cv2.imshow("line", frame)

            # 눈 양 끝점 중앙값
            min_eye_x = int(left_eye_x + (right_eye_x - left_eye_x) / 2)
            min_eye_y = int(left_eye_y + (right_eye_y - left_eye_y) / 2)

            # tan 값 계산
            tan_x = min_eye_x - left_eye_x
            tan_y = left_eye_y - min_eye_y
            # 각도 계산
            angle = np.arctan(tan_y / tan_x)
            degree = np.degrees(angle)

            # 좌표 회전(앞에서 구한 각도 만큼 회전)
            rsd_1 = rotate(x, y)
            rsd_2 = rotate(x1, y)
            rsd_3 = rotate(x, y1)
            rsd_4 = rotate(x1, y1)
            d2_1 = rotate(bdx, bdy)
            d2_2 = rotate(bdx1, bdy)
            d2_3 = rotate(bdx, bdy1)
            d2_4 = rotate(bdx1, bdy1)

            # 회전된 좌표를 이용하여 새로운 창으로 프린트
            pts1 = np.float32([[d2_1[0], d2_1[1]], [d2_2[0], d2_2[1]], [d2_3[0], d2_3[1]], [d2_4[0], d2_4[1]]])
            pts2 = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])
            M = cv2.getPerspectiveTransform(pts1, pts2)  # 기하학적 변환
            # 원근 변환 cv2.warpPerspective(origin_frame, 변환 프레임, (width, height))
            dst_frame = cv2.warpPerspective(frame, M, (400, 400))
            d2gray = cv2.cvtColor(dst_frame, cv2.COLOR_BGR2GRAY)
            d2clahe_image = clahe.apply(d2gray)
            d2detections = face_detector(d2clahe_image)

            # 회전시킨 d2에서 얼굴 인식 진행
            for d2 in d2detections:
                xx = d2.left()
                yy = d2.top()
                xx1 = d2.right()
                yy1 = d2.bottom()
                d2shape = shape_predictor(d2clahe_image, d2)

                cv2.rectangle(dst_frame, (xx, yy), (xx1, yy1), GREEN, 1)
                # 눈 선따라 이어주기
                # for i in range(36, 41):
                #     cv2.line(dst_frame, (d2shape.part(i).x, d2shape.part(i).y),
                #              (d2shape.part(i + 1).x, d2shape.part(i + 1).y),
                #              GREEN, 1)
                # for i in range(42, 47):
                #     cv2.line(dst_frame, (d2shape.part(i).x, d2shape.part(i).y),
                #              (d2shape.part(i + 1).x, d2shape.part(i + 1).y),
                #              GREEN, 1)
                landmarks = []
                for p in d2shape.parts():
                    landmarks.append([p.x, p.y])
                landmarks = np.array(landmarks)

                left_eye = landmarks[LEFT_EYE]
                right_eye = landmarks[RIGHT_EYE]
                left_ER = ER_ratio(left_eye)
                right_ER = ER_ratio(right_eye)

            cv2.imshow("dst_frame", dst_frame)

    video_capture.release()
    cv2.destroyAllWindows()
