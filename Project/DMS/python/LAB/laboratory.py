from argparse import ArgumentParser
from function_stub import *
import cv2


cnt = lab_test_class()

video_capture = cv2.VideoCapture(0)  # 카메라

while True:
    frame_got, frame = video_capture.read()
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

    if key == 65:
        cnt.inner()
    ret, img = video_capture.read()

    cv2.imshow("test", img)
