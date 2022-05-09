import cv2
import numpy
import dlib
arr = [i for i in range(10)]
print(len(arr))

del arr[0]
print(len(arr))
arr.append(30)
print(f"{len(arr)}, {arr}")
