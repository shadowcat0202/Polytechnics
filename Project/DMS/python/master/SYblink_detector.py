import cv2
import numpy as np
import matplotlib.pyplot as plt
from mark_detector import *
from SYcommon_calculator import *


class BlinkDetector():
    def __init__(self, view, img_ip, allLandmarks):
        self.view = view
        self.img_ip = img_ip
        self.ldmk_all = allLandmarks
        self.ldmk_face_outl = allLandmarks[0:17]
        self.ldmk_lftE0, self.ldmk_lftE1, self.ldmk_lftE2, self.ldmk_lftE3, self.ldmk_lftE4, self.ldmk_lftE5, = self.ldmk_lftE_outl = allLandmarks[
                                                                                                                                      36:42]
        self.ldmk_rytE0, self.ldmk_rytE1, self.ldmk_rytE2, self.ldmk_rytE3, self.ldmk_rytE4, self.ldmk_rytE5, = self.ldmk_rytE_outl = allLandmarks[
                                                                                                                                      42:48]

    def get_xy_min_max_ofLandmark(self, landmarks):
        x_ldmks = [loc[0] for loc in landmarks]
        y_ldmks = [loc[1] for loc in landmarks]
        range_x = (np.min(x_ldmks), np.max(x_ldmks))
        range_y = (np.min(y_ldmks), np.max(y_ldmks))

        return range_x, range_y

    def crop_image_ofLandmark(self, image, landmarks):

        (x_min, x_max), (y_min, y_max) = range_x, range_y = self.get_xy_min_max_ofLandmark(landmarks)
        px_pd = 5

        y_from, y_to = y_min - px_pd, y_max + px_pd
        x_from, x_to = x_min - px_pd, x_max + px_pd

        img_crop = image[y_from:y_to, x_from:x_to]
        xy_topLft = (x_from, y_from)

        return img_crop, xy_topLft

    def get_area_ofLandmark(self, landmarks):
        # mask_blackout = np.zeros(, dtype=np.uint8)
        image = self.img_ip
        mask_blackout = np.zeros_like(image, dtype=np.uint8)
        mask_shape = cv2.fillConvexPoly(mask_blackout, landmarks, 255)
        mask_crop, _ = self.crop_image_ofLandmark(mask_shape, landmarks)
        contours, _ = cv2.findContours(mask_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x),
                          reverse=True)  # Print the contour with the biggest area first.
        area = cv2.contourArea(contours[0])
        # Visualization
        # for cnt in contours:
        #     (x, y, w, h) = cv2.boundingRect(cnt)
        #     cv2.drawContours(mask_crop, [cnt], -1, (0, 255, 0), -1)
        #     break
        # cv2.imshow("CONTOURS", mask_crop)
        # cv2.imshow("CONTOURS", mask_shape)

        return area

    def get_ratio_eye_to_face(self):
        area_lftE = self.get_area_ofLandmark(self.ldmk_lftE_outl)
        area_rytE = self.get_area_ofLandmark(self.ldmk_rytE_outl)
        area_face = self.get_area_ofLandmark(self.ldmk_face_outl)
        # print(f"area eye(L/R) = {ar_lftE}/{ar_rytE}")

        ratio_L_E2F = area_lftE / area_face
        ratio_R_E2F = area_rytE / area_face
        # print(f"ratio E2F(L/R) = {ratio_L}/{ratio_R}")

        return ratio_L_E2F, ratio_R_E2F

    def get_ratio_eye_height_to_width(self):
        (x_min_L, x_max_L), (y_min_L, y_max_L) = range_x_lftE, range_y_lftE = self.get_xy_min_max_ofLandmark(
            self.ldmk_lftE_outl)
        (x_min_R, x_max_R), (y_min_R, y_max_R) = range_x_rytE, range_y_rytE = self.get_xy_min_max_ofLandmark(
            self.ldmk_rytE_outl)

        ratio_L_H2W = (y_max_L - y_min_L) / (x_max_L - x_min_L)
        ratio_R_H2W = (y_max_R - y_min_R) / (x_max_R - x_min_R)

        return ratio_L_H2W, ratio_R_H2W

    def get_two_ratios_from_LR_eyes(self):
        L_E2F, R_E2F = self.get_ratio_eye_to_face()
        L_H2W, R_H2W = self.get_ratio_eye_height_to_width()

        multiply1 = 10000
        L_E2F, R_E2F = L_E2F * multiply1, R_E2F * multiply1

        multiply2 = 100
        L_H2W, R_H2W = L_H2W * multiply2, R_H2W * multiply2

        list_ratios = [L_E2F, R_E2F, L_H2W, R_H2W]
        list_ratios = [round(ratio, 4) for ratio in list_ratios]
        # print(f"list_ratio = {list_ratios}")

        array_ratios = np.array(list_ratios)
        # print(f"array_ratio = {array_ratios}")
        # array_ratios = np.round(array_ratios, 3)
        # print(f"array_ratios = {array_ratios}")

        return array_ratios

    def update_min_max_then_get_normalized_ratios(self, ratios_minlmax):
        ratios = self.get_two_ratios_from_LR_eyes()
        # print(f"loaded ratios = {ratios}")
        minlmax = ratios_minlmax
        # print(f"loaded min/max = {minlmax}")

        ratios_normalized = []
        for idx in range(len(ratios)):
            if ratios[idx] < minlmax[idx][0]:
                minlmax[idx][0] = ratios[idx]
                # print(f"min updated as {ratios[idx]}/{minlmax[idx][0]}(check:{minlmax[idx][0] == ratios[idx]})")
                # print(f"min updated as {ratios[idx]}/{minlmax[idx][0]}(check:{minlmax[idx][0]==ratios[idx]})")
            if ratios[idx] > minlmax[idx][1]:
                minlmax[idx][1] = ratios[idx]
                # print(f"max updated as {ratios[idx]}/{minlmax[idx][1]}(check:{minlmax[idx][1] == ratios[idx]})")

            # print(f"minlmax[{idx}] = {minlmax[idx]}")
            # print(f"isSame = ")
            if (minlmax[idx][1] == minlmax[idx][0]) or (minlmax[idx][0] == ratios[idx]):
                ratios_normalized.append(np.NAN)
            else:
                range_ratio = round(minlmax[idx][1] - minlmax[idx][0], 4)
                # print(f"range_ratio = {range_ratio} = ({ratios[idx]}")
                # print(f"ratio[{idx}] = {ratios[idx]}")
                ratio_NM = round((ratios[idx] - minlmax[idx][0]) / range_ratio, 4)
                ratios_normalized.append(ratio_NM)
                # print(f"ratio_NM[{idx}] == {ratio_NM} appended")
        # print(f"updated min/max = {minlmax}")
        # print(f"ratios_normalized = {ratios_normalized}")
        return minlmax, ratios_normalized

    def is_open(self, ratios_NM):
        ratios_NM = ratios_NM
        thold = 0.4
        results = [1 if ratio > thold else 0 for ratio in ratios_NM]

        output = 1 if results.count(1) > results.count(0) else 0

        return results, output

    def preprocess_img(self, image):

        img = image
        thold = np.min(img) + np.std(img)
        img = cv2.GaussianBlur(img, (5, 5), 2)
        img = np.where(img < thold, 255 - img * np.std(img) * 1, 0)
        _, img = cv2.threshold(img, np.median(img), 255, cv2.THRESH_BINARY)
        img = cv2.erode(img, None, 2)
        img = cv2.dilate(img, None, 1)
        img_prcd = img
        return img_prcd

    def classify_gaze_direcetion(self, x_min, x_max, x_pupil):
        min = x_min
        mid = (x_max + x_min) / 2
        max = x_max

        distance_toMin = abs(min - x_pupil)
        distance_toMid = abs(mid - x_pupil)
        distance_toMax = abs(max - x_pupil)

        arr_distance = np.array([distance_toMin, distance_toMid, distance_toMax])

        index_direction = np.argmin(arr_distance)
        # print(f"min/max/pupl = {x_min}/{x_max}/{x_pupil}")
        # print(f"list_distance = {arr_distance}")
        # print(f"index_direction = {index_direction}")

        return index_direction

    def get_pupil_location_and_direction(self, show=True):

        # lftE =  self.crop_image_ofLandmark(self.img_ip, self.ldmk_lftE_outl)
        # rytE =  self.crop_image_ofLandmark(self.img_ip, self.ldmk_rytE_outl)

        landmarks = [self.ldmk_lftE_outl, self.ldmk_rytE_outl]

        ### Contour 계산 및
        xy_pupil_ORG = []
        indice_direction = []
        for side in landmarks:
            eye = self.crop_image_ofLandmark(self.img_ip, side)
            img_eye = self.preprocess_img(eye[0]).astype('uint8')
            x_topLft, y_topLft = eye[1]
            contours, _ = cv2.findContours(img_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x),
                              reverse=True)  # Print the contour with the biggest area first.
            for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                x_pupl_CRP, y_pupl_CRP = (2 * x + w) / 2, (2 * y + h) / 2
                xy_pupil = (round(x_topLft + x_pupl_CRP, 0), round(y_topLft + y_pupl_CRP, 0))

                if show is True:
                    # print(f"xy_pupil[0]/[1] = {xy_pupil[0]}/{xy_pupil[1]}")
                    cv2.drawContours(self.view, [cnt], -1, (0, 255, 0), 1)
                    cv2.circle(self.view, (int(xy_pupil[0]), int(xy_pupil[1])), int(h / 2), (255, 0, 0), -1)
                xy_pupil_ORG.append(xy_pupil)

                x_minlmax, _ = self.get_xy_min_max_ofLandmark(side)
                x_min, x_max = x_minlmax
                # print(f"x_min/max = {x_minlmax}")
                idx_direction = self.classify_gaze_direcetion(x_min, x_max, xy_pupil[0])
                indice_direction.append(idx_direction)
                # print("ONESIDE DONE")
                break



        return xy_pupil_ORG, indice_direction


    def decide_on_gaze_direction(self, indice_direction):
        # print(f"indice_direction = {indice_direction}")

        if len(indice_direction)==2:
            prd_L, prd_R = indice_direction[0], indice_direction[1]

            if (prd_L == 0 and prd_R == 0) or (prd_L == 0 and prd_R == 1) or (prd_L == 1 and prd_R == 0):
                direction = "LEFT"
            elif (prd_L == 2 and prd_R == 2) or (prd_L == 1 and prd_R == 2) or (prd_L == 2 and prd_R == 1):
                direction = "RIGHT"
            elif (prd_L == 1 and prd_R == 1):
                direction = "CENTRE"
            else:
                direction = "Err"

            # cv2.putText(self.view, f"{direction}", self.ldmk_all[33], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(self.view, f"{direction}", (620, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

"""
# for eye in [lftE, rytE]:
#     img_eye = self.preprocess_img(eye[0]).astype('uint8')
#     x_topLft, y_topLft = eye[1]
#     contours, _ = cv2.findContours(img_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key=lambda x: cv2.contourArea(x),
#                       reverse=True)  # Print the contour with the biggest area first.
#     for cnt in contours:
#         (x, y, w, h) = cv2.boundingRect(cnt)
#         x_pupl_CRP, y_pupl_CRP = (2 * x + w) / 2, (2 * y + h) / 2
#         xy_pupil = (round(x_topLft+x_pupl_CRP,0), round(y_topLft+y_pupl_CRP,0))
#         VISUALIZATION
#         # Cropped Image
#         plt.imshow(img_eye)
#         plt.scatter(xy_ctr[0], xy_ctr[1])
#         plt.show()
#
#         # Original Image
#         plt.imshow(self.view)
#         plt.scatter(xy_pupil[0], xy_pupil[1])
#         plt.show()
#
#         # print(f"xy_pupil[0]/[1] = {xy_pupil[0]}/{xy_pupil[1]}")
#         cv2.drawContours(self.view, [cnt], -1, (0, 255, 0), 1)
#         cv2.circle(self.view, (int(xy_pupil[0]), int(xy_pupil[1])), int(h/2), (255, 0, 0), -1)
#         xy_pupil_ORG.append(xy_pupil)
#         break
"""
