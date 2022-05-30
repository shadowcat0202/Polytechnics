import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


data_path = "D:/JEON/dataset/eye/MRL/mrlEyes_2018_01/"

f = open("D:/JEON/dataset/eye/MRL/pupil.txt")
# wavg = 0
# havg = 0
# cnt = 0
re_size = (92, 92)
# X = []
# Y = []
# print("read...")
# while True:
#     line = f.readline().strip()
#     if not line:
#         break
#     i = line.find("/")
#     folder = line[:i]
#
#     j = line.find(" ", i+1)
#     file_name = line[i+1:j]
#
#     pos = [0, 0]
#     pos[0], pos[1] = map(int, line[j+1:].split())
#
#     img = cv2.imread(data_path + folder + "/" + file_name)
#     bfh, bfw = len(img), len(img[0])
#     img = cv2.resize(img, re_size)
#     X.append(img)
#     pos[0] = round(pos[0] * (92/bfh))
#     pos[1] = round(pos[1] * (92/bfw))
#     Y.append(pos)
#
#
# X = np.array(X)
# Y = np.array(Y)

# np.save("D:/JEON/dataset/eye/MRL/" + f"X_data", X)  # .npy
# np.save("D:/JEON/dataset/eye/MRL/" + f"Y_data", Y)  # .npy
# print("end!")


print("X, Y nparray loading...")
X = np.load("D:/JEON/dataset/eye/MRL/X_data.npy")
Y = np.load("D:/JEON/dataset/eye/MRL/Y_data.npy")
print(f"X{X.shape}, Y{Y.shape} nparray load completion")