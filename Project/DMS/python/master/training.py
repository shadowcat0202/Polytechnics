import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from EYE import eye_Net
import matplotlib.pyplot as plt

MY_EPOCH = 10
MY_BATCHSIZE = 5
eye = eye_Net()
# 데이터 npy 형식으로 저장
# eye.make_dataset("D:/JEON/dataset/eye/data/train/", "train_save")
# eye.make_dataset("D:/JEON/dataset/eye/data/test/", "test_save")
# train_X, train_Y = eye.load_dataset("D:/JEON/dataset/eye/", "train_save")

# npy불러오기
train_X, train_Y = eye.load_dataset("D:/JEON/dataset/eye/data/train/", "train_save")
# train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size=0.1, shuffle=True)
test_X, test_Y = eye.load_dataset("D:/JEON/dataset/eye/data/test/", "test_save")

print(train_X[0])
print(test_X[0])

model = eye.model()
with tf.device("/device:GPU:0"):
    hist = model.fit(train_X, train_Y, epochs=MY_EPOCH, batch_size=MY_BATCHSIZE, verbose=1,
                     validation_data=(test_X, test_Y))
model.save(f"D:/JEON/dataset/eye/model/keras_eye_trained_model_wow.h5")

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

# eye.eye_predictor("D:/JEON/dataset/eye/model/keras_eye_trained_model_good.h5")
# test_loss, test_acc = eye.eye_model.evaluate(test_X, test_Y)
# print(f"loss:{test_loss}, acc:{test_acc}")
