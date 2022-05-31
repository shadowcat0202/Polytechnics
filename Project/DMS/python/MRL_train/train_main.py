import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential
import tensorflow.keras.optimizers



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

# X = np.mean(X, axis=2)
X = X[:,:,:,0] / 255
X = np.expand_dims(X, axis=-1)
print(type(X[0][0][0][0]))
print(Y[0][0])
print(type(Y[0][0]))
#
pp = Pupil()

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1, shuffle=True)
print(f"{len(train_X)}, {len(test_X)}")

tf.random.set_seed(0)
MY_EPOCH = 50
MY_BATCHSIZE = 20
model = pp.make_model(lr=0.0003)
with tf.device("/device:GPU:0"):
    hist = model.fit(train_X, train_Y, epochs=MY_EPOCH, batch_size=MY_BATCHSIZE, verbose=1,
                     validation_data=(test_X, test_Y))
model.save(f"D:/JEON/dataset/eye/MRL/eye_pupil_test1.h5")

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()