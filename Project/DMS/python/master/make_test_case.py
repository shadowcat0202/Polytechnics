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


tool = my_make_test_case()

root_path = "D:\mystudy\pholythec\Project\DMS"
# root_path = root_path.replace("\\", "/")
file_name_list = glob.glob(root_path + "/*.mp4")
pprint.pprint(file_name_list)




