import cv2
import numpy as np
import matplotlib.pyplot as plt

blue = [255, 0, 0]
green = [0, 255, 0]
red = [0, 0, 255]


def face_roi(_img, _img_show=False):
    # face = _img[240:400, 217:365]
    face = _img[240:400, 217:365].copy()  # Deep copy
    face[:, :, 0] = 255
    cv2.imshow("lena_face", face)
    cv2.waitKey(0)
    cv2.destroyWindow("lena_face")


def change_raw_col(_img):
    _img[200, :] = green
    _img[:, 100] = blue


def img_split_merge(_img, _show_img=False):
    b = _img[:, :, 0]
    g = _img[:, :, 1]
    r = _img[:, :, 2]

    if _show_img:
        hstack = np.hstack([b, g, r])
        cv2.imshow("bgr_split", hstack)

        merge_img = cv2.merge((r, g, b))
        cv2.imshow("mrege(r,g,b)", merge_img)

        cv2.waitKey(0)
        cv2.destroyWindow("bgr_split")
        cv2.destroyWindow("mrege(r,g,b)")
    return b, g, r

def color_img_to_gray(_img):
    """
    컬러 이미지를 gray로 바꾸고 다시 컬러로 바꾸는 방법
    :param _img:    원본 이미지
    :return: 
    """
    # rgb의 값을 평균내 한개의 채널로 변경
    # [10, 6, 12] -> [9]
    color_to_gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("color_to_gray", color_to_gray)
    # 한개의 채널을 rgb 모든 채널에 똑같이 넣는다. color 정보를 잃어 버린다.
    gray_to_bgr = cv2.cvtColor(color_to_gray, cv2.COLOR_GRAY2BGR)
    cv2.imshow("gray_to_bgr", gray_to_bgr)
    cv2.waitKey(0)
    cv2.destroyWindow("color_to_gray")
    cv2.destroyWindow("gray_to_bgr")


def mean_filter(_img, kernel_size=3):
    """
    :param _img:    원본 이미지
    :param kernel_size:     커널 사이즈(홀수 추천)
    :return:    None
    """
    kernel = np.ones((kernel_size, kernel_size), np.float32) / kernel_size ** 2
    dst = cv2.filter2D(_img, -1, kernel)

    hstack = np.hstack([_img, dst])
    cv2.imshow("mean_filter(original, mean)", hstack)
    cv2.waitKey(0)
    cv2.destroyWindow("mean_filter(original, mean)")


def custom_filter(_img):
    kernel_1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])  # 자기 자신
    kernel_2 = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])  # 왼쪽으로 옮겨간다
    kernel_3_Gaussian = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 3, 1, 0],
        [1, 3, 5, 3, 1],
        [0, 1, 3, 1, 0],
        [0, 0, 1, 0, 0]
    ]) / 9
    kernel_4_Gaussian = np.array([
        [5, 3, 1, 3, 5],
        [0, 1, 0, 1, 3],
        [1, 0, 0, 0, 1],
        [3, 1, 0, 1, 3],
        [5, 3, 1, 3, 5]
    ]) / 9

    dst_1 = cv2.filter2D(_img, -1, kernel_1)
    dst_2 = cv2.filter2D(_img, -1, kernel_2)
    dst_3 = cv2.filter2D(_img, -1, kernel_3_Gaussian)
    dst_4 = cv2.filter2D(_img, -1, kernel_4_Gaussian)

    dst_1[265, :] = green
    dst_2[265, :] = green

    hstack = np.hstack([dst_1, dst_2, dst_3, dst_4])
    cv2.imshow("custom_filter", hstack)
    cv2.waitKey(0)
    cv2.destroyWindow("custom_filter")


def sharpening(_img):
    kernel = np.array([[-1, -1, -1], [-1, 45, -1], [-1, -1, -1]]) / 9  # 선을 선명하게

    dst = cv2.filter2D(_img, -1, kernel)

    cv2.imshow("sharpening", np.hstack([_img, dst]))
    cv2.waitKey(0)
    cv2.destroyWindow("sharpening")


def make_histogram(_img, roi=None):
    """
    이미지를 넣으면
    :param roi: 원하는 구역
    :param _img: image
    :return: None
    """
    if len(_img.shape) == 3:  # color
        histo = [[0] * 256 for _ in range(3)]

        for y in range(_img.shape[0]):
            for x in range(_img.shape[1]):
                histo[0][_img[y, x, 0]] += 1
                histo[1][_img[y, x, 1]] += 1
                histo[2][_img[y, x, 2]] += 1

        fig, axes = plt.subplots(1, 3)
        fig.suptitle(f'image histogram', fontsize=15)  # 제목

        for i in range(3):
            axes[i].plot(histo[i])

        plt.tight_layout()
        plt.show()

    elif len(_img.shape) == 2:  # gray
        histo = [0] * 256
        for y in range(_img.shape[0]):
            for x in range(_img.shape[1]):
                histo[_img[y, x]] += 1
        print(histo)
        plt.plot(histo)
        plt.show()


def make_histogram_use_cvlib(_img):
    """
    opencv에서 제공하는 라이브러리를 사용
    :param _img: image
    :return: none
    """
    gray = cv2.cvtColor(_img, cv2.COLOR_RGB2GRAY)
    histo = cv2.calcHist(images=[gray],
                         channels=[0],
                         mask=None,
                         histSize=[256],
                         ranges=[0, 256])
    print(histo)

    histo = histo.flatten()
    histo = histo.astype(np.int32)
    print(histo)
    plt.plot(histo)
    plt.show()


def norm_min_max_img(_img, _opencv=False, _show=False):
    """
    normalize
    :param _img:
    :param _opencv: opencv lib 사용 여부
    :param _show: 이미지와 그래프를 보고 싶은 경우
    :return: gray, normalized img, 누적 histogram
    """
    norm_img = None
    histo = None
    if len(_img.shape) == 3:
        gray = cv2.cvtColor(_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = _img

    # opencv lib 사용
    if _opencv:
        norm_opencv = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        if _show:
            cv2.imshow('norm_opencv', norm_opencv)

    # 직접 구현
    else:
        norm_img = gray.astype(np.float32)
        norm_img = (norm_img - norm_img.min()) / (norm_img.max() - norm_img.min())

        # 영상을 시각화 할려면
        # 픽셀의 값이 0~1 사이로 정규화 되므로 255를 곱해줘야 한다.
        norm_img *= 255

        # gray 영상으로 보기 위해 unit8 로 바꾼다.
        norm_img = norm_img.astype(np.uint8)

        histo = np.bincount(norm_img.ravel(), minlength=256)

        if _show:
            plt.plot(histo)
            plt.show()
            cv2.imshow('norm_img', norm_img)
    if _show:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return gray, norm_img, histo


def equalization_img(_img, _show_hist=False, _show_img=False):
    """
    이미지 평탄화
    :param _img: 
    :param _show_hist: 
    :param _show_img: 
    :return: 한 채널에 대한 평탄화 이미지
    """
    gray, norm_img, norm_histo = norm_min_max_img(_img)

    # histo equal
    cdf = norm_histo.cumsum()

    # cdf_min 은 반드시 0이 아닌 값이다.
    # 우리는 min max 정규화를 했으므로 0이 없을 것이다. 확인!
    # print(cdf.min())

    H = norm_img.shape[0]
    W = norm_img.shape[1]
    h_v = (cdf - cdf.min()) / (H * W - cdf.min()) * 255

    he_img = norm_img.copy()
    for y in range(H):
        for x in range(W):
            he_img[y, x] = h_v[norm_img[y, x]]

    # 사실상 위의 작업들은 아래와 같은 opencv lib 코드 한줄로 가능
    # opencv_eq = cv2.equalizeHist(norm_img)

    if _show_hist:
        subpolt_raw = 1
        subpolt_col = 3
        plt.subplot(subpolt_raw, subpolt_col, 1)
        plt.plot(norm_histo)
        plt.subplot(subpolt_raw, subpolt_col, 2)
        plt.plot(cdf)
        plt.subplot(subpolt_raw, subpolt_col, 3)
        plt.plot(h_v)
        plt.show()
    if _show_img:
        # h1 = np.hstack([gray, norm_img])
        # h2 = np.hstack([he_img, opencv_eq])
        # stack_img = np.vstack([h1, h2])
        stack_img = np.hstack([gray, norm_img, he_img])
        cv2.imshow('gray, Noralize, he_img, opencv', stack_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return he_img


def color_equalize(_img):
    """
    opencv lib 사용해서 분리 -> merge 해서 사용
    :param _img: 이미지
    :return: none
    """
    b = _img[:, :, 0]
    g = _img[:, :, 1]
    r = _img[:, :, 2]
    b = cv2.equalizeHist(b)
    g = cv2.equalizeHist(g)
    r = cv2.equalizeHist(r)
    eq_img = cv2.merge((b, g, r))
    cv2.imshow('eq_img', eq_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def practice_1(_img):

    b = _img[:, :, 0]
    g = _img[:, :, 1]
    r = _img[:, :, 2]
    b = equalization_img(b)
    g = equalization_img(g)
    r = equalization_img(r)
    eq_img = cv2.merge((b,g,r))
    cv2.imshow('img', np.hstack([_img, eq_img]))
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
    practice_1(yard)
