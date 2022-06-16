import pprint

import cv2
import numpy as np
import os
import glob

class my_make_test_case:
    def __init__(self):
        pass

    def window_path_to_linux_path(self, _path):
        result = _path.replace("\\", "/")
        return result

    def eye_pixel_count(self, left, right):
        side_add_rat = 0.25
        cv2.imshow("left", left)
        cv2.imshow("right", right)
        left_count = [0, 0, 0]
        right_count = [0, 0, 0]
        left = cv2.resize(left, dsize=(left.shape[1] * 2, left.shape[0] * 2))
        right = cv2.resize(right, dsize=(right.shape[1] * 2, right.shape[0] * 2))

        H = left.shape[0]
        W_1 = left.shape[1] // 3
        W_2 = round(left.shape[1] * (2 / 3))
        left_count[0] += round(H * W_1 * side_add_rat)
        left_count[2] += round(H * (left.shape[1]-W_2) * side_add_rat)
        # print(f"left.shape{left.shape}, {W_1}, {W_2}")
        for i in range(H):
            for j in range(W_1):
                if left[i][j] == 0:
                    left_count[0] += 1

            for j in range(W_1, W_2):
                if left[i][j] == 0:
                    left_count[1] += 1

            for j in range(W_2, left.shape[1]):
                if left[i][j] == 0:
                    left_count[2] += 1

        H = right.shape[0]
        W_1 = right.shape[1] // 3
        W_2 = round(right.shape[1] * (2 / 3))
        right_count[0] += round(H * W_1 * side_add_rat)
        right_count[2] += round(H * (right.shape[1] - W_2) * side_add_rat)
        # print(f"right.shape{right.shape}, {W_1}, {W_2}")
        for i in range(H):
            for j in range(W_1):
                if right[i][j] == 0:
                    right_count[0] += 1

            for j in range(W_1, W_2):
                if right[i][j] == 0:
                    right_count[1] += 1

            for j in range(W_2, right.shape[1]):
                if right[i][j] == 0:
                    right_count[2] += 1

        # for l, r in zip(left_count, right_count):
        #     print(f"{round(l)}, {round(r)}")
        return left_count, right_count

    def eye_dir(self, left, right):
        left_eye, right_eye = self.eye_pixel_count(left, right)
        l = left_eye.index(max(left_eye))
        r = right_eye.index(max(right_eye))
        # print(f"l:{l}, r:{r}")
        if (l == 0 and r == 0) or (l == 0 and r == 1) or (l == 1 and r == 0):
            return "left"
        elif (l == 2 and r == 2) or (l == 1 and r == 2) or (l == 2 and r == 1):
            return "right"
        else:
            return "center"

# tool = my_make_test_case()
#
# root_path = "D:\mystudy\pholythec\Project\DMS"
# # root_path = root_path.replace("\\", "/")
# file_name_list = glob.glob(root_path + "/*.mp4")
# pprint.pprint(file_name_list)
#
#


