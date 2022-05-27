import cv2
import time

import tensorflow

from my_mark_detector import MarkDetector, FaceDetector
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from branch_test1.pose_estimator import PoseEstimator
import calculation as calc
from branch_test1.img_draw import *
import Preprocessing as ps
import dlib
import numpy as np
from tensorflow import keras


def get_dlib_rectangle_split_int(dlib_rect):
    mul = [0.9, 0.8, 1.1, 1.1]
    return int(dlib_rect.left() * mul[0]), int(dlib_rect.top() * mul[1]), int(dlib_rect.right()*mul[2]), int(dlib_rect.bottom()*mul[3])


print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
SKYBLUE = (255, 255, 0)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def make_model(x, y):
    demention = 1
    if len(x.shape) == 4:
        demention = 3
    outShape = len(y[0])
    print(x.shape, outShape, demention)
    # model = Sequential()
    # model.add(Conv2D(32, (3, 3), padding='same', activation='tanh', input_shape=(inSahpe[1], inSahpe[2], demention)))
    # model.add(MaxPool2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(32, activation='tanh'))
    # model.add(Dropout(0.5))
    # model.add(Dense(outShape, activation='sigmoid'))
    # model.summary()

    model = Sequential()
    # Layer 1
    model.add(Conv2D(64, (5, 5), activation='tanh',
                     input_shape=(x.shape[1], x.shape[2], demention),
                     padding='same'))  # 32 x 32 x 3
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # 단순 사이즈를 줄이는 것이기 때문에 W가 늘어나지는 않는다 16 x 16
    # model.add(AvgPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Layer 2
    # model.add(Conv2D(32, (5,5), activation='tanh', padding='same'))  # Conv2d(filter, kernel_size 부터 시작한다)
    # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # 단순 사이즈를 줄이는 것이기 때문에 W가 늘어나지는 않는다 8 x 8
    # model.add(AvgPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Layer 3
    model.add(Conv2D(32, (5, 5), activation='tanh', padding='same'))  # Conv2d(filter, kernel_size 부터 시작한다)
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # 단순 사이즈를 줄이는 것이기 때문에 W가 늘어나지는 않는다 4 x 4
    # model.add(AvgPool2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Flatten())  # 4 x 4 로 만들어진 이미지를 1 차원으로 핀다

    model.add(Dense(8, activation='tanh'))
    model.add(Dense(outShape, activation="softmax"))
    model.summary()

    opt = tensorflow.keras.optimizers.Adam(learning_rate=0.5)
    # opt = SGD(lr=1.5, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="mean_squared_error",
                  optimizer=opt,
                  metrics=["accuracy"])
    return model



trackers = []

cap = cv2.VideoCapture("D:/JEON/Polytechnics/Project/DMS/dataset/WIN_20220520_16_13_04_Pro.mp4")
# video_capture = cv2.VideoCapture(0)  # 카메라

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
RES_W = 1280 # 1280 # 640 # 256 # 320 # 480 # pixels
RES_H = 720 # 720 # 480 # 144 # 240 # 360 # pixels
frame_count = 0
pose_estimator = PoseEstimator(img_size=(height, width), model_path="../assets/model.txt")

# 3. Introduce a mark detector to detect landmarks.
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("../assets/shape_predictor_68_face_landmarks.dat")
my_model = keras.models.load_model("../face_landmark_training/model/cnn_test(20).h5")
eye_calc = calc.eye_calculation()


def my_predict(img, box):
    # print(f"origin:{img.shape}")

    # print(img.shape)

    # img = img[start_y:end_y, start_x:end_x]
    # img = np.expand_dims(img, axis=0)
    # print(type(img), img.shape)
    # img = cv2.resize(img,(720, 480))    # 480, 720
    # img = [img, img, img]
    # print(f"2{img.shape}")
    img = cv2.resize(img, (720, 480))
    img = np.expand_dims(img, axis=0)
    result = my_model.predict(img)
    # print(result)
    # print(f"res{result}")
    result = list([result[0][i], result[0][i+1]] for i in range(0, len(result[0]), 2))
    return result

if not cap.isOpened():
    print("Error opening video Stream or file")

while cap.isOpened():
    ret, frame = cap.read()
    print("frame:", frame.shape)
    if ret:
        frame_copy = cv2.resize(frame, (RES_W, RES_H), cv2.INTER_AREA)
        # frame_copy = np.array(ps.img_Preprocessing_v3(frame_copy))
        # frame_copy = np.copy(frame)
        if frame_copy is None:
            break

        if frame_count == 0:
            bboxes = face_detector(frame_copy)
            for i in bboxes:
                (start_x, start_y, end_x, end_y) = (i.left(), i.top(), i.right(), i.bottom())
                traker = dlib.correlation_tracker()
                rect = dlib.rectangle(start_x, start_y, end_x, end_y)
                traker.start_track(frame_copy, rect)

                trackers.append(traker)
                frame_count += 1
        else:
            for i, traker in enumerate(trackers):
                # print(traker[0], traker[1])
                traker.update(frame_copy)   # 트레킹 갱신
                pos = traker.get_position() # 트레킹한 네모 받기
                # print(type(pos))
                start_x, start_y, end_x, end_y = get_dlib_rectangle_split_int(pos)

                cv2.rectangle(frame_copy, (start_x, start_y), (end_x, end_y), WHITE, 2)

                # dlib.full_object_detection()
                landmarks = my_predict(frame_copy, (start_x, start_y, end_x, end_y))
                print(landmarks)
                # shape = shape_predictor(frame_copy, dlib.rectangle(start_x, start_y, end_x, end_y))
                # landmarks = list([p.x, p.y] for p in shape.parts())  # pos의 타입을 뭐로 줘야 하는거지?
                # print(landmarks)
                if landmarks:
                    for mark in landmarks:
                        cv2.circle(frame_copy, (int(mark[0]), (int(mark[1]))), 1, WHITE, -1, cv2.LINE_AA)
                cv2.putText(frame_copy, f"{i+1}", (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

        # print(frame_count)
        cv2.imshow("frame", frame_copy)
    key = cv2.waitKey(1)
    if key == 27 or not ret:
        break
print("finish processing")
cv2.destroyAllWindows()
cap.release()
