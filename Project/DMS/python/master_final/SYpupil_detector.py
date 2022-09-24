import statistics


import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
# import seaborn as sns
from scipy.signal import correlate
from SYeye_caculation import *

class PupilDetector:
    def __init__(self, view, frame, landmark):
        self.img_org = view
        self.img_ipt = frame
        self.ldmk1, self.ldmk2, self.ldmk3, self.ldmk4, self.ldmk5, self.ldmk6 = self.ldmks = landmark
        self.loc_x, self.loc_y = self.loc_xy()
        self.xy_min, self.xy_max = self.loc_xy_min_max()
        self.px_wth, self.px_hgt, self.size_eye = self.eye_width_height_with_padding()
        self.xy_mid = self.loc_xy_eye_center()
        self.img_crp = self.img_eye_cropped()
        self.img_prcd = self.img_eye_processed()

    def loc_xy(self):
        loc_x = [i[0] for i in self.ldmks]
        loc_y = [i[1] for i in self.ldmks]

        return loc_x, loc_y

    def loc_xy_min_max(self):
        loc_x = [i[0] for i in self.ldmks]
        loc_y = [i[1] for i in self.ldmks]

        xy_min = min(loc_x), min(loc_y)
        xy_max = max(loc_x), max(loc_y)

        return xy_min, xy_max

    def eye_width_height_with_padding(self):
        x_min, y_min = self.xy_min
        x_max, y_max = self.xy_max
        wth = x_max - x_min
        hgt = y_max - y_min
        rate_pd = 1.3

        return wth * rate_pd, hgt * rate_pd, (wth, hgt)

    def loc_xy_eye_center(self):
        xy_mid = sum(self.loc_x) / len(self.loc_x), sum(self.loc_y) / len(self.loc_y)

        return xy_mid

    def img_eye_cropped(self):
        x_mid, y_mid = self.loc_xy_eye_center()
        x_adj, y_adj = self.px_wth, self.px_hgt

        x_from, x_to = round(x_mid - x_adj), round(x_mid + x_adj)
        y_from, y_to = round(y_mid - y_adj), round(y_mid + y_adj)

        img_cropped = self.img_ipt[y_from:y_to, x_from:x_to]

        return img_cropped

    def img_eye_processed(self):
        kernel = (3,3)
        img = self.img_crp
        img = cv2.pyrUp(cv2.pyrUp(img))
        thold = np.min(img)+np.std(img)
        img = np.where(img < thold, 255-img*4, 0)
        thold2 = np.max(img)-np.std(img)
        img = np.where(img > thold2, 255, 0).astype('uint8')


        # img = np.where(img)
        # _, img = cv2.threshold(img, thold, 255, cv2.THRESH_OTSU)
        # print(f" max = {np.max(img)}")
        # mean = np.mean(np.where(img<255))
        # print(f"mean = {mean}")
        # cv2.imshow

        # img = cv2.GaussianBlur(img, kernel, 50 )
        # img = np.where(img < thold, 255-img, 0).astype('uint8')
        # img = cv2.erode(img, kernel, 3)
        # _, img = cv2.threshold(img, np.median(img), 255, cv2.THRESH_BINARY)
        # img = cv2.erode(img, kernel, 3)
        # img = cv2.dilate(img, kernel, 1)
        # img = cv2.medianBlur(img, 3,3)
        # img = cv2.medianBlur(img, 3,3)
        # _, img = cv2.threshold(img, np.mean(img), 255, cv2.THRESH_TRUNC)


        img_prcd = img

        return img_prcd
    #
    # def img_eye_processed(self):
    #     kernel_size = (3,3)
    #     img = self.img_crp
    #     thold = np.min(img) + np.std(img)
    #     img = np.where(img < thold, 0, 255).astype('uint8')
    #     # img = np.where(img < thold, 255-img, 0).astype('uint8')
    #     _, img = cv2.threshold(img, thold, 255, cv2.THRESH_OTSU)
    #     img = cv2.dilate(img, kernel_size, iterations=3)
    #     img = cv2.erode(img, kernel_size, iterations=3)
    #     # img = cv2.blur(img, (3, 3)).astype('uint8')
    #     # img = cv2.filter2D(img, (3, 3)).astype('uint8')
    #     # mode = statistics.mode(img.reshape(-1))
    #
    #     img_prcd = img
    #
    #     return img_prcd

