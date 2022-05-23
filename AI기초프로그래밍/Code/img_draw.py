import cv2
import numpy as np


class imgMake:
    def __init__(self):
        pass

    # 검정색 이미지를 생성
    # h : 높이
    # w : 넓이
    # d : 깊이 (1 : gray, 3: bgr)
    def create_image(self, h, w, d):
        image = np.zeros((h, w, d), np.uint8)
        color = tuple(reversed((0, 0, 0)))
        image[:] = color
        return image

    # 이미지 여러장을 한개의 창에 출력하기 위한 base창 설정
    # hcout : 높이 배수(2: 세로로 2배)
    # wcount : 넓이 배수 (2: 가로로 2배)
    def create_image_multiple(self, h, w, d, hcout, wcount):
        image = np.zeros((h * hcout, w * wcount, d), np.uint8)
        color = tuple(reversed((0, 0, 0)))
        image[:] = color
        return image

    # 통이미지 하나에 원하는 위치로 복사(표시)
    # dst : create_image_multiple 함수에서 만든 통 이미지
    # src : 복사할 이미지
    # h : 높이
    # w : 넓이
    # d : 깊이
    # col : 행 위치(0부터 시작)
    # row : 열 위치(0부터 시작)
    def showMultiImage(self, dst, src, h, w, d, col, row):
        # 3 color
        if d == 3:
            dst[(col * h):(col * h) + h, (row * w):(row * w) + w] = src[0:h, 0:w]
        # 1 color
        elif d == 1:
            dst[(col * h):(col * h) + h, (row * w):(row * w) + w, 0] = src[0:h, 0:w]
            dst[(col * h):(col * h) + h, (row * w):(row * w) + w, 1] = src[0:h, 0:w]
            dst[(col * h):(col * h) + h, (row * w):(row * w) + w, 2] = src[0:h, 0:w]
