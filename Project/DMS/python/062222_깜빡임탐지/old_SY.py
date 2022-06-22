import cv2
import matplotlib.pyplot as plt
import numpy as np


class PupilDetection:
    def __init__(self, view, inputImage, landmark):
        # print("calculation ON")
        self.img_org = view
        self.img_input = inputImage
        self.landmark = landmark
        self.loc_xy_int = self.loc_xy_intersection()
        self.hgt_wth_eye = self.eye_height_width()
        self.xy_adj = self.get_adjustment_xy()
        self.tmpl_org = self.template_eye()
        self.tmpl_rszd = self.template_eye_cropped_and_largened()
        self.tmpl_prcd = self.template_eye_preprocessed()
        self.loc_xy_pupl_tmpl_rszd = self.loc_xy_pupils_center_in_template()
        self.loc_xy_pupl_view = self.loc_xy_pupils_center_in_original_image()
        self.loc_x_dvsn = self.loc_x_division_lines()

    def line(self, point1, point2):
        """
        returns x, y and intercept of a linear line
        :param point1: location of a point on the line
        :param point2: location of a point on the line
        :return: value of y, x, and intercept from the line
        """
        x_pt1, y_pt1 = point1
        x_pt2, y_pt2 = point2
        diff_Y = (y_pt1 - y_pt2)
        diff_X = (x_pt2 - x_pt1)
        intcp = (x_pt1 * y_pt2 - x_pt2 * y_pt1) * (-1)

        return diff_Y, diff_X, intcp

    def loc_xy_intersection(self):
        x1, y1, b1 = self.line(self.landmark[1], self.landmark[4])
        x2, y2, b2 = self.line(self.landmark[2], self.landmark[5])

        D = x1 * y2 - y1 * x2
        Dx = b1 * y2 - y1 * b2
        Dy = x1 * b2 - b1 * x2
        # print(f"클래스 d, dx, dy = {D}, {Dx}, {Dy}")

        if D != 0:
            xy_int = (Dx / D, Dy / D)

        else:
            print("Error: D==0")

        return xy_int

    def eye_height_width(self):
        hgt = max(self.landmark[5][1] - self.landmark[1][1], self.landmark[4][1] - self.landmark[2][1])
        wth = self.landmark[3][0] - self.landmark[0][0]
        return hgt, wth

    def template_eye(self):
        img_prcd = self.img_input

        mask = get_shape_in_blackout_mask(img_prcd.shape[:2], self.landmark)
        mask = cv2.dilate(mask, kernel=kernel(7, 7), iterations=3)
        eyes = cv2.bitwise_and(img_prcd, img_prcd, mask=mask)
        mask = (eyes == 0)
        eyes[mask] = 255

        return eyes

    def get_adjustment_xy(self):
        xy_adj = self.hgt_wth_eye[1] * 1.5, self.hgt_wth_eye[0] * 1.5

        return xy_adj

    def template_eye_cropped_and_largened(self):
        x_ctr, y_ctr = self.loc_xy_int
        x_adj, y_adj = self.xy_adj

        tmpl_crp = self.tmpl_org[round(y_ctr - y_adj):round(y_ctr + y_adj), round(x_ctr - x_adj):round(x_ctr + x_adj)]
        tmpl_resized = cv2.pyrUp(tmpl_crp)

        return tmpl_resized

    def gaussianBlur(self, image=False, filterSize=(41, 41), std=50):
        if image is False:
            img_blurred = cv2.GaussianBlur(self.img_input, filterSize, std)
        else:
            img_blurred = cv2.GaussianBlur(image, filterSize, std)
        return img_blurred

    def threshold(self, image=False, quantile=0.01, maxValue=255, type=cv2.THRESH_BINARY_INV):
        # 이미지가 주어지지 않았을 때. 객체의 이미지에 연산
        if image is False:
            img_values = flatten_array_remove_item(self.img_input, 255)  # 이미지의 배열을 1차원으로 만든 뒤, 해당 배열에서 255 값을 제거
            thres = np.quantile(img_values, quantile)  # 배열에서 밝기값 하위 5% (quantile == 0.05)를 나누는 밝기구간
            _, img_thold = cv2.threshold(self.img_input, thres, maxValue, type)
        # 이미지가 주어졌을 때. 주어진 이미지에 연산
        else:
            img_values = flatten_array_remove_item(image, 255)
            thres = np.quantile(img_values, quantile)
            _, img_thold = cv2.threshold(image, thres, maxValue, type)

        return img_thold
    #
    # # # TODO : Test
    def template_eye_preprocessed_test(self):
        tmpl_blurred = self.gaussianBlur(self.tmpl_rszd)
        tmpl_thres = self.threshold(tmpl_blurred)
        tmpl_erd = cv2.erode(tmpl_thres, (5,5), 1)
        tmpl_dlt = cv2.dilate(tmpl_erd, (5,5), 5)

        # tmpl_processed = tmpl_erd
        tmpl_processed = tmpl_dlt
        # tmpl_processed = tmpl_thres

        return tmpl_processed

    ##사용본
    def template_eye_preprocessed(self):
        tmpl_blurred = self.gaussianBlur(self.tmpl_rszd)
        tmpl_thres = self.threshold(tmpl_blurred)
        tmpl_erd = cv2.erode(tmpl_thres, (5,5), 1)
        tmpl_dlt = cv2.dilate(tmpl_erd, (5,5), 5)

        tmpl_processed = tmpl_dlt

        return tmpl_processed

    # def template_eye_preprocessed(self):
    #     tmpl_HE = cv2.equalizeHist(self.tmpl_rszd)
    #     tmpl_blurred = self.gaussianBlur(tmpl_HE)
    #     tmpl_thres = self.threshold(tmpl_blurred)
    #
    #     tmpl_processed = tmpl_thres
    #
    #     return tmpl_processed

    def loc_xy_pupils_center_in_template(self):
        contours, _ = cv2.findContours(self.tmpl_prcd, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x),
                          reverse=True)  # Print the contour with the biggest area first.
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            xy_ctr = (2 * x + w) / 2, (2 * y + h) / 2
            # cv2.drawContours(self.tmpl_rszd, [cnt], -1, (0, 255, 0), 1)
            # cv2.circle(self.tmpl_eye, (round(x_ctr), round(y_ctr)), round(min(w, h) / 4), (255, 0, 0))
            break

        return xy_ctr

    # ToDo : delete
    def contour_ratio_wth_hgt(self):
        contours, _ = cv2.findContours(self.tmpl_prcd, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x),
                          reverse=True)  # Print the contour with the biggest area first.
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            xy_ctr = (2 * x + w) / 2, (2 * y + h) / 2
            # cv2.drawContours(self.tmpl_rszd, [cnt], -1, (0, 255, 0), 1)
            # cv2.circle(self.tmpl_eye, (round(x_ctr), round(y_ctr)), round(min(w, h) / 4), (255, 0, 0))
            break
        print(f"cnt ratio/w/h- {w / h}: {w}/{h}")
        ratio = w / h

        return ratio

    def loc_xy_pupils_center_in_original_image(self):
        x_int, y_int = self.loc_xy_int
        x_adj, y_adj = self.xy_adj

        x_pupl_tmpl, y_pupl_tmpl = self.loc_xy_pupl_tmpl_rszd[0] / 2, self.loc_xy_pupl_tmpl_rszd[1] / 2

        xy_pupl_org = round(x_int - x_adj) + x_pupl_tmpl, round(y_int - y_adj) + y_pupl_tmpl

        return xy_pupl_org

    def loc_x_division_lines(self):
        x_mid = self.loc_xy_int[0]
        x_lft = self.landmark[0][0]
        x_ryt = self.landmark[3][0]

        return x_lft, x_mid, x_ryt

    def distance_to_center_x(self):
        dst_to_ctr = abs(self.loc_x_dvsnline[1] - self.loc_xy_pupl[0])
        # print(f"xyDiv, xyPupl: {self.loc_x_dvsnline[1]}/{self.loc_xy_pupl[0]}")
        return dst_to_ctr

    def classify_look_direction(self):
        x_dvsn_lft, x_dvsn_mid, x_dvsn_ryt = self.loc_x_dvsn
        x_pupl = self.loc_xy_pupl_view[0]

        dst_to_lft = abs(x_dvsn_lft - x_pupl)
        dst_to_mid = abs(x_dvsn_mid - x_pupl)
        dst_to_ryt = abs(x_dvsn_ryt - x_pupl)

        list_dst = [dst_to_lft, dst_to_mid, dst_to_ryt]

        dst_min = min(list_dst)
        prediction = list_dst.index((dst_min))

        return prediction


## 일반 함수

def show_plt_img_scatter(image, x, y, titleName="Window"):
    plt.imshow(image, cmap='gray')
    plt.scatter(x, y)
    plt.title(titleName)
    plt.show()


def show_cv_img(image1, image2=False, title="Window"):
    if image2 is False:
        cv2.imshow(title, image1)
        cv2.waitKey(0)
    else:
        cv2.imshow(title, image1)
        cv2.imshow(title, image2)
        cv2.waitKey(0)


def get_shape_in_blackout_mask(size, landmarks):
    mask_blackout = np.zeros(size, dtype=np.uint8)
    mask_withShape = cv2.fillConvexPoly(mask_blackout, landmarks, 255)
    return mask_withShape


def kernel(row, col):
    kernel = np.ones((row, col), np.uint8)
    # print(f"kernel({row}*{col}) created")
    return kernel


def flatten_array_remove_item(array, itemToPop):
    array_flat = np.ndarray.flatten(array)
    array_toPop = np.array(itemToPop)
    array_refined = np.setdiff1d(array_flat, array_toPop)
    return array_refined


def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (0.5)


def average(*args):
    avg = sum(args) / len(args)
    return avg

def final_prediction(prd_L, prd_R):
    if (prd_L==0 and prd_R==0) or (prd_L==0 and prd_R==1) or (prd_L==1 and prd_R==0):
        return "LEFT"
    elif (prd_L==2 and prd_R==2) or (prd_L==1 and prd_R==2) or (prd_L==2 and prd_R==1):
        return "RIGHT"
    elif (prd_L==1 and prd_R==1):
        return "CENTRE"
    else:
        return "Err"