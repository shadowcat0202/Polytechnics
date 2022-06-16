import cv2
import numpy as np


def img_Preprocessing(img_frame):
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

    gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img_frame, cv2.COLOR_BGR2LAB)

    # # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    lab = clahe.apply(lab)

    # # 빛 제거? 무슨 원리인지는 모르겠다
    L = lab[:, :, 0]
    med_L = cv2.medianBlur(L, 99)  # median filter  # 뭔지 모르겠음
    invert_L = cv2.bitwise_not(med_L)  # invert lightness   # 빛 제거?
    composed = cv2.addWeighted(gray, 0.75, invert_L, 0.25, 0)

    # # Histograms Equalization
    equ = cv2.equalizeHist(gray)  # 무조건 색상이 1차원이여야 한다 = gray
    result = np.hstack((gray, equ))

    # cv2.imshow("Histograms Equalization", composed)
    return img_frame, composed, result


def img_Preprocessing_v2(img_frame):
    # Gaussian Pyramid: An image pyramid is a collection of images - all arising from a single original image - that are successively downsampled until some desired stopping point is reached.
    # https://docs.opencv.org/3.4/d4/d1f/tutorial_pyramids.html
    re_size = 0
    re_size = cv2.pyrDown(np.mean(img_frame, axis=2).astype("uint8"))
    # re_size = cv2.pyrDown(img_frame)

    # Tophat: The top-hat filter is used to enhance bright objects of interest in a dark background.
    # https://www.geeksforgeeks.org/top-hat-and-black-hat-transform-using-python-opencv/
    filterSize = (150, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
    topHat = cv2.morphologyEx(re_size, cv2.MORPH_TOPHAT, kernel)  # 밝기가 높은것을 부각 시켜준다

    # HE: It is a method that improves the contrast in an image, in order to stretch out the intensity range (see also the corresponding Wikipedia entry).
    # gray = cv2.cvtColor(topHat, cv2.COLOR_BGR2GRAY)
    result = cv2.equalizeHist(topHat)
    return result


def img_Preprocessing_v3(img_frame):
    # Gaussian Pyramid: An image pyramid is a collection of images - all arising from a single original image - that are successively downsampled until some desired stopping point is reached.
    re_size = cv2.pyrDown(np.mean(img_frame, axis=2).astype("uint8"))

    # Tophat: The top-hat filter is used to enhance bright objects of interest in a dark background.
    # filterSize = (320, 200)
    filterSize = (160, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
    topHat = cv2.morphologyEx(re_size, cv2.MORPH_TOPHAT, kernel)  # 밝기가 높은것을 부각 시켜준다

    # CLAHE : # Equalizes the histogram of a grayscale image using Contrast Limited Adaptive Histogram Equalization.
    clahe = cv2.createCLAHE(clipLimit = 1.0, tileGridSize = (16,16))
    clahe = clahe.apply(topHat)

    histEqual = cv2.equalizeHist(clahe)

    result = histEqual
    return result

def img_gray_Preprocessing(img_frame):
    result = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    return result

def thresHold(img):
    # _, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # _, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    # _, th = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    # _, th = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO) # 이거 생각보다 잘나옴 ㅋ
    # _, th = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
    # th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)  # 이상하게 잘나오는듯? 아닌듯?
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)

    return th