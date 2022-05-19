import cv2
from functools import wraps
import time


class eye_calculation:
    def __init__(self):
        self.both_ER_ratio_avg = 0
        self.clahe = cv2.createCLAHE(clipLimit=2.0,
                                     tileGridSize=(8, 8))
        pass

    # (두 점 사이의 유클리드 거리 계산)
    def distance(self, p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1 / 2)

    # 눈 비율 값 계산
    def ER_ratio(self, eye_point):
        # 얼굴 특징점 번호 사진 참조
        A = self.distance(eye_point[1], eye_point[5])
        B = self.distance(eye_point[2], eye_point[4])
        C = self.distance(eye_point[0], eye_point[3])
        return (A + B) / (2.0 * C)

    def eye_close(self, left_eye, right_eye):
        left_ER = self.ER_ratio(left_eye)
        right_ER = self.ER_ratio(right_eye)

        avg = round((left_ER + right_ER) / 2, 2)
        self.both_ER_ratio_avg = avg
        return avg < 0.3

    def draw_eye_close_ratio(self, img, color=(0, 255, 0)):
        cv2.putText(img, f"{self.both_ER_ratio_avg}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # https://wjh2307.tistory.com/21
    # 눈 감기(함수) + 자는거 판단(밑에 코드)
    def close_counter(self, func):
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
    def close(self, img, color=(0, 0, 255)):
        cv2.putText(img, "close", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


class ImgPreprocessing:
    def img_Preprocessing(self, img_frame):
        re_size = cv2.resize(img_frame, (400, 400), cv2.INTER_AREA)
        gray = cv2.cvtColor(re_size, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(re_size, cv2.COLOR_BGR2LAB)

        gray = self.clahe.apply(gray)
        # lab = clahe.apply(lab)

        L = lab[:, :, 0]
        med_L = cv2.medianBlur(L, 99)  # median filter  # 뭔지 모르겠음
        invert_L = cv2.bitwise_not(med_L)  # invert lightness   # 빛 제거
        composed = cv2.addWeighted(gray, 0.75, invert_L, 0.25, 0)
        return re_size, composed
