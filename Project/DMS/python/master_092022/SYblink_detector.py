import cv2
import numpy as np
import matplotlib.pyplot as plt
from mark_detector import *
from SYcommon_calculator import *
import timeit
from visualization import*

cm = Camera()

class BlinkDetector():
    def __init__(self, view, img_ip, allLandmarks):
        self.view = view
        self.img_ip = img_ip
        self.ldmk_EEF = self.ldmk_lftE, self.ldmk_rytE, self.ldmk_face = [allLandmarks[36:42], allLandmarks[42:48], allLandmarks[0:17]]
        self.ldmk_Eyes = [self.ldmk_lftE, self.ldmk_rytE]
        self.topLft_EEF = self.xy_topLeft_ofEEF()
        self.crp_lftE, self.crp_rytE, self.crp_face = self.crp_EEF = self.cropped_images_ofEEF()
        self.msk_lftE, self.msk_rytE, self.msk_face = self.msk_EEF = self.cropped_mask_ofEEF()
        self.ratios = self.ratios_E2Fx2_H2W()

    def compare_functions(self, f1, f2):
        time_f1 = timeit.timeit(f1)
        time_f2 = timeit.timeit(f2)

        func = 1 if time_f1 < time_f2 else 2
        diff = abs(time_f1-time_f2)
        times = time_f2/time_f2 if time_f1<time_f2 else time_f1/time_f2

        msg = f"f1 is faster ({time_f2/time_f2:.2f})" if time_f1<time_f2 else f"f2 is faster ({time_f1/time_f2:.2f})"
        print(msg)

        return (func, diff, times)

    def time_it(self, f1):
        time_f1 = timeit.timeit(f1)


        print(time_f1)

        # return time_f1


    """ COMMON METHODS START HERE"""
    def minlmax_xy_ofLandmarks(self, landmarks):
        x_ldmks, y_ldmks = [loc[0] for loc in landmarks], [loc[1] for loc in landmarks]
        range_xy = np.array([np.min(x_ldmks), np.max(x_ldmks), np.min(y_ldmks), np.max(y_ldmks)])
        # print(f"range_xy = {range_xy}")

        return range_xy

    def minlmax_xy_ofLandmarks_v2(self, landmarks):
        sorted_byX = sorted(landmarks, key=lambda ldmk: ldmk[0])
        sorted_byY = sorted(landmarks, key=lambda ldmk: ldmk[1])

        x_ldmks, y_ldmks = [loc[0] for loc in landmarks], [loc[1] for loc in landmarks]
        range_xy = np.array([np.min(x_ldmks), np.max(x_ldmks), np.min(y_ldmks), np.max(y_ldmks)])
        # print(f"range_xy = {range_xy}")

        return range_xy



    def fromTo_range_ofLandmarks(self, landmarks):
        px_pd = 5
        x_min, x_max, y_min, y_max = self.minlmax_xy_ofLandmarks(landmarks)
        y_from, y_to = y_min - px_pd, y_max + px_pd
        x_from, x_to = x_min - px_pd, x_max + px_pd

        list_fromTo = [y_from, y_to, x_from, x_to]
        arr_fromTo = np.array(list_fromTo)

        return arr_fromTo

    def pupil_xy_inOriginal(self, image, topLeft, show):
        xy_crpPupl = (0, 0)
        x_topLft, y_topLft = topLeft
        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x),
                          reverse=True)  # Print the contour with the biggest area first.
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            x_pupl_CRP, y_pupl_CRP = (2 * x + w) / 2, (2 * y + h) / 2
            xy_crpPupl = (round(x_topLft + x_pupl_CRP, 0), round(y_topLft + y_pupl_CRP, 0))  # 크롭 이미지 내 좌표를 원본좌표에 맞춤

            # if show is True:
            #     cv2.drawContours(self.view, [cnt], -1, (0, 255, 0), 1)
            #     cv2.circle(self.view, (int(xy_crpPupl[0]), int(xy_crpPupl[1])), int(h / 2), (255, 0, 0), -1)
            break

        return xy_crpPupl

    def processed_image(self, image, mask):
        img, msk = image, mask
        size_filter = (5, 5)

        msk = 255 - msk
        img = cv2.bitwise_not(img, img, mask=msk)
        # bitwise_not = cv2.pyrUp(img.copy())

        img = cv2.medianBlur(img, 5)
        # medianblur = cv2.pyrUp(img.copy())

        thold = np.min(img)
        img = np.where(img <= thold, 255, 0).astype('uint8')
        img = cv2.dilate(img, size_filter, iterations=2)
        # dilate = cv2.pyrUp(img.copy())

        img = cv2.erode(img, size_filter, iterations=1)
        # erode = cv2.pyrUp(img.copy())

        # cv2.imshow("test", np.hstack([bitwise_not, medianblur, dilate, erode]))

        img_prcd = img

        return img_prcd

    def classify_gaze_direcetion(self, x_min, x_max, x_pupil):
        x_minlcenterlmax = x_min, (x_max + x_min) / 2, x_max

        list_distance = [abs(to-x_pupil) for to in x_minlcenterlmax]
        arr_distance = np.array(list_distance)
        index_direction = np.argmin(arr_distance)
        # print(f"min/max/pupl = {x_min}/{x_max}/{x_pupil}")
        # print(f"list_distance = {arr_distance}")
        # print(f"index_direction = {index_direction}")

        return index_direction


    """ COMMON METHODS END HERE"""

    #  #  #  #  #  #  #

    """ 
    METHODS FOR EEF(Lft Eye, Ryt Eye, Face) START HERE
    """
    def xy_topLeft_ofEEF(self):
        list_topLft = []
        for part in self.ldmk_EEF:
            arr_fromTo = self.fromTo_range_ofLandmarks(part) # returns x_min, x_max, y_min, y_max
            xy_topLeft = [arr_fromTo[2], arr_fromTo[0]] # x_min, y_min
            list_topLft.append(xy_topLeft) # save each toplft loc in the list.

        arr_topLft = np.array(list_topLft)
        return arr_topLft

    def cropped_images_ofEEF(self):
        images = [] #list where images will be saved together.
        for part in self.ldmk_EEF: # [ldk_lftE, ldk_rytE, ldk_face]
            image = self.img_ip # grayscale image
            y_from, y_to, x_from, x_to = self.fromTo_range_ofLandmarks(part) #range from/to which the image is cropped.
            img_crop = image[y_from:y_to, x_from:x_to] # image cropped
            images.append(img_crop) # save each image in list.

        crops_EEF = np.array(images[0]), np.array(images[1]), np.array(images[2])
        # ToDo - Delete
        # cv2.imshow("L", crop_EEF[0])
        # cv2.imshow("R", crop_EEF[1])
        # cv2.imshow("F", crop_EEF[2])

        return crops_EEF

    def cropped_mask_ofEEF(self):
        # load left eye/right eye/ face images
        images = self.cropped_images_ofEEF()
        arr_topLft_EEF = self.topLft_EEF # locations of top/left of leftE, rightE, face
        ldmk_EEF = self.ldmk_EEF
        # print(f"arr_topLft_EEF = {arr_topLft_EEF}")
        # print(f"ldmk_EEF = {ldmk_EEF}")
        # print(arr_topLft)
        list_masks = []
        for idx, image in enumerate(images):
            # mask_blackout = np.zeros(, dtype=np.uint8)
            # print(f"len(landmarks) = {len(landmarks)}")
            # print(f"landmarks(before-np) = {ldmk_EEF[idx]}")
            # print(f"topleft = {arr_topLft_EEF[idx]}")
            ldmk_EEF[idx] = np.subtract(ldmk_EEF[idx], arr_topLft_EEF[idx])

            # print(f"landmarks(after-np) = {ldmk_EEF[idx]}")
            mask_blackout = np.zeros_like(image, dtype=np.uint8)
            mask_shape = cv2.fillConvexPoly(mask_blackout, ldmk_EEF[idx], 255)
            mask = cv2.dilate(mask_shape, None, 1)
            list_masks.append(mask)

        masks_EEF = np.array(list_masks[0]), np.array(list_masks[1]), np.array(list_masks[2])
        # cv2.imshow("L", masks_EEF[0])
        # cv2.imshow("R", masks_EEF[1])
        # cv2.imshow("F", masks_EEF[2])

        return masks_EEF

    def ratios_E2Fx2_H2W(self):
        ### Area Ratios (L/R)
        multiply_E2F, multiply_H2W = 1000, 100
        list_area, list_ratios = [], []

        for idx, mask in enumerate(self.msk_EEF):
            area = (mask == 255).sum()
            list_area.append(area)

        ratios_E2F = [round(list_area[idx]/list_area[2]*multiply_E2F, 4) for idx in range(2)]
        """
        ratios_E2F
        area_eye / area_face * adj(multiply) >>> round, 0 >> repeat for 2 
        """
        list_ratios.append(ratios_E2F[0])
        list_ratios.append(ratios_E2F[1])

        ldmk_eyes = [self.ldmk_lftE, self.ldmk_rytE]
        for eye in ldmk_eyes:
            x_min, x_max, y_min, y_max = self.minlmax_xy_ofLandmarks(eye)
            hgt, wth = y_max-y_min, x_max-x_min
            ratio_H2W = round(hgt/wth*multiply_H2W, 4)
            list_ratios.append(ratio_H2W)

        arr_ratios = np.array(list_ratios)
        # print(f"list_ratios = {list_ratios}")

        return arr_ratios

    def new_minlmax_and_normalized_ratios(self, ratios_minlmax):
        arr_ratios = self.ratios_E2Fx2_H2W()
        arr_minlmax = ratios_minlmax
        # print(f"[loaded] arr_ratios = {arr_ratios}/type: {type(arr_ratios)}")
        # print(f"[loaded] arr_minlmax = {arr_minlmax}/type: {type(arr_minlmax)}")
        list_NMratios = []

        # START OF FOR LOOP
        for idx in range(len(arr_ratios)):
            # print(f"len(arr_ratios) = {len(arr_ratios)}")
            # print(f"arr_minlmax[{idx}](before) = {arr_minlmax[idx]}")
            """ min/max update """ # min[idx], max[idx] = arr_minlmax[idx][0], arr_minlmax[idx][1]
            arr_minlmax[idx][0] = arr_ratios[idx] if (arr_ratios[idx]<arr_minlmax[idx][0]) else arr_minlmax[idx][0] # min update
            arr_minlmax[idx][1] = arr_ratios[idx] if (arr_ratios[idx]>arr_minlmax[idx][1]) else arr_minlmax[idx][1] # max update
            # print(f"minlmax[{idx}](after) = {arr_minlmax[idx]}")


            """ calculate normalized ratios / exception """
            range_ratio = arr_minlmax[idx][1] - arr_minlmax[idx][0] # range calculated.
            # print(f"range_ratio = {range_ratio}")
            # print(f"arr_ratios[idx] = {arr_ratios[idx]}")
            nmRatio = (arr_ratios[idx]-arr_minlmax[idx][0])/range_ratio if range_ratio!=0 else np.nan # normalized ratio or nan
            # print(f"nmRatio = {nmRatio}")
            list_NMratios.append(nmRatio) # save the ratio in list
        # END OF FOR LOOP

        arr_ratios_NM = np.array(list_NMratios)
        # print(f"[final] arr_ratios_NM = {arr_ratios_NM}")
        # print(f"[final] arr_minlmax = {arr_minlmax}")
        # print("ONE CYCLE")
        return arr_minlmax, arr_ratios_NM

    def eye_status_open(self, arr_nmRatios):
        thold = 0.4
        results = [1 if ratio > thold else 0 for ratio in arr_nmRatios]
        status = 1 if results.count(1) > results.count(0) else 0

        return results, status

    def display_eye_status(self,results_close, status):

        msg = "OPEN" if status == 1 else "CLOSED"  # 이미지에 표기할 메시지

        cv2.putText(self.view, f"{msg}", (500, 300), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=cm.getGreen(),
                    thickness=2)
        # cv2.putText(self.view, f"Dicrection?{direction}", (620, 330), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=cm.getBlue())
        cv2.putText(self.view, f"results_close: {results_close}", (400, 330), cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                    color=cm.getRed(), thickness=1)

    """ 
    METHODS FOR EEF(Lft Eye, Ryt Eye, Face) START HERE
    """

    # # # # #

    """ 
    METHODS FOR [GAZE ESTIMATION] START HERE
    """

    def eye_gaze_estimation(self, show=True):
        xy_orgPupl = []
        indice_direction = []
        """ pupil xy concersion to original image"""

        ### Contour 계산 및
        for i, eye in enumerate(self.ldmk_Eyes):
            xy_topLft = self.topLft_EEF[i]
            crop_eye, mask_eye = self.crp_EEF[i], self.msk_EEF[i]
            img_eye = self.processed_image(crop_eye, mask_eye)

            xy_crpPupl = self.pupil_xy_inOriginal(img_eye, xy_topLft, show=show)
            xy_orgPupl.append(xy_crpPupl)

            # 좌표 비교 기준점 (좌, 중, 우) 도출 후 각 눈의 예측값을 0, 1, 2 형태로 도출
            x_min, x_max, *_ = self.minlmax_xy_ofLandmarks(eye)
            idx_direction = self.classify_gaze_direcetion(x_min, x_max, xy_crpPupl[0])
            indice_direction.append(idx_direction)
            # print("ONESIDE DONE")
        result = False
        if sum(indice_direction) <= 1:
            msg = "LEFT"
            result = True
        elif sum(indice_direction) == 2:
            msg = "CENTRE"
            result = False
        elif sum(indice_direction) >= 3:
            msg = "RIGHT"
            result = True
        else:
            msg = "ERROR"
            result = False

        return indice_direction, msg, result

    def display_gaze_estimation(self,results_direction, direction):

        cv2.putText(self.view, f"{direction}", (700, 300), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=cm.getGreen(),
                    thickness=2)
        # cv2.putText(self.view, f"Dicrection?{direction}", (620, 330), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=cm.getBlue())
        cv2.putText(self.view, f"results_direction: {results_direction}", (700, 330), cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                    color=cm.getRed(), thickness=1)
