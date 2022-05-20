import cv2
import numpy
import dlib

ER_cnt = 0
state = 1

def test(a, b):
    if b == 30:
        b = 0
    return b + 1, a


while True:
    ER_cnt, state = test([10, 20], ER_cnt)
    print(f"{ER_cnt},{state}")

