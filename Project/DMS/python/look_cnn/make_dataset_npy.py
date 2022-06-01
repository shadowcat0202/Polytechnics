import glob
import pprint
from tqdm import tqdm
from tqdm import trange
import cv2
import numpy as np
import os
from tensorflow.keras.utils import to_categorical

# 1. 9개 위치 번호 별로 묶은 npy 만들기
# 2. 상 중 하 로 묶은 npy 만들기
# 3. 좌 중 우 로 묶은 npy 만들기
# 번외 위의 모든 항목에 대해서 txt파일 만들기

working_directory = "D:/JEON/dataset/look_direction/"
raw_directory = working_directory + "cnn_eyes/"
sorted_directory = working_directory + "cnn_eyes_number_sort/"


def img_renaming():
    directory_number_list = [str(i) for i in range(1, 7)]
    num = 0
    img_number = 0
    unexpect_finish = False
    for directory_number in directory_number_list:
        curr_directory = raw_directory + directory_number + "/"
        print(f"{curr_directory} renaming...")
        file_name = None
        file_names = os.listdir(curr_directory)
        for i, file_name in enumerate(file_names):
            if i % 1000 == 0:
                percent = i / len(file_names) * 100
                print(f"progress:{round(percent)}%")
            old_dir = curr_directory + file_name
            img = None
            img = cv2.imread(old_dir)
            img = cv2.resize(img, (86, 36)) # (w, h, 3)
            img = img[:, :, 0]  # (36, 86)
            img = np.expand_dims(img, axis=-1)  # (36, 86, 1)

            # '{0:05}'.format(img_name_counter) 번호 포멧팅
            new_dir = sorted_directory + directory_number + "_" + '{0:05}'.format(img_number) + ".png"
            img_number += 1
            # print(new_dir)
            if img is None:
                unexpect_finish = True
                img_number -= 1
                print(f"{directory_number} folder {i} img finish!")
                num = i
                break
            cv2.imwrite(new_dir, img)

        if not unexpect_finish:
            print(f"{directory_number} folder {num} img finish!")
            unexpect_finish = False



def dataset_to_npy():
    X = []
    Y = []
    left = [1,4]
    mid = [2,5]
    right = [3,6]
    file_names = os.listdir(sorted_directory)
    print(f"total img : {len(file_names)}")
    for i, name in enumerate(file_names):
        if i % 2000 == 0:
            print(f"{round(i / len(file_names) * 100)}")
        classification_number_int = int(name[:name.find("_")]) - 1
        img = cv2.imread(sorted_directory + name)  # ndarray
        X.append(img[:, :, 0])
        if classification_number_int in left:
            Y.append([0])
        elif classification_number_int in mid:
            Y.append([1])
        else:
            Y.append([2])
    X = np.array(X)
    # X = X.reshape(len(X), len(X[0]), len(X[0][0]), 1)   # ? 이거 왜 못씀? ㅋ
    X = np.expand_dims(X, axis=-1)
    Y = np.array(Y)
    Y = to_categorical(Y)
    print(f"X:{X.shape}, Y:{Y.shape}")
    np.save(working_directory + f"X_classification_leftmidright", X)  # .npy
    np.save(working_directory + f"Y_classification_leftmidright", Y)  # .npy



img_renaming()
dataset_to_npy()

X = np.load("D:/JEON/dataset/look_direction/X_classification_leftmidright.npy")

Y = np.load("D:/JEON/dataset/look_direction/Y_classification_leftmidright.npy")

print(f"{X.shape}")
print(f"{Y.shape}\n{Y}")