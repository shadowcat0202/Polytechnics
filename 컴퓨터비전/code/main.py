import cv2
import numpy as np
import matplotlib.pyplot as plt

import Image_Processing as ipp


def face_roi(_img, _img_show=False):
    # face = _img[240:400, 217:365]
    face = _img[240:400, 217:365].copy()  # Deep copy
    face[:, :, 0] = 255
    cv2.imshow("lena_face", face)
    cv2.waitKey(0)
    cv2.destroyWindow("lena_face")


def practice_1(_img):
    b = _img[:, :, 0]
    g = _img[:, :, 1]
    r = _img[:, :, 2]
    b = ipp.equalization_img(b)
    g = ipp.equalization_img(g)
    r = ipp.equalization_img(r)
    eq_img = cv2.merge((b, g, r))
    cv2.imshow('img', np.hstack([_img, eq_img]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def class_c_b():
    src_img = cv2.imread('./c_b.png')
    gray = ipp.color_img_to_gray(src_img)

    ipp.my_Threshold(gray)

    # _, bin_img_bin = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    _, bin_img_otsu = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)
    # cv2.imshow('th',bin_img_bin)

    bin_img_bin = cv2.cvtColor(bin_img_otsu, cv2.COLOR_GRAY2BGR)
    masked_img = cv2.bitwise_and(src_img, bin_img_bin)
    cv2.imshow('mask', masked_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
def class_20220928():
    src_img = cv2.imread('./Lenna.png')
    bgr_to_gray = ipp.color_img_to_gray(src_img)
    rows, cols = bgr_to_gray.shape
    gradient = np.zeros((rows-2, cols-2), dtype=np.uint8)
    k_size = 3

    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])

    for y in range(rows - 2):
        for x in range(cols - 2):
            roi = bgr_to_gray[y:y+k_size, x:x+k_size]
            sx = np.sum(kernel_x * roi)
            sy = np.sum(kernel_y * roi)
            gradient[y, x] = np.sqrt(sx**2 + sy**2)

    cv2.imshow('gradient', gradient)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    lena = cv2.imread("./Lenna.png")
    yard = cv2.imread('./Unequalized_Hawkes_Bay_NZ.jpg')
    # change_raw_col(lena)
    # face_roi(lena)
    # img_split_merge(lena)
    # color_img_to_gray(lena)
    # mean_filter(lena, kernel_size=100)
    # custom_filter(lena)
    # sharpening(lena)

    # make_histogram(lena)
    # make_histogram_use_cvlib(lena)

    # norm_min_max_img(lena)

    # equalization_img(lena, _show_img=True)
    # color_equalize(lena)
    # practice_1(yard)

    class_20220928()
