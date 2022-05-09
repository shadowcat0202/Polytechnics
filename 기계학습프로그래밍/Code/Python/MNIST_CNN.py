from tensorflow.keras.datasets import mnist
from keras.utils import np_utils

import numpy
import os

import tensorflow as tf

device = tf.device('cuda')

numpy.random.seed(0)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# for x in X_train[0]:
#     for i in x:
#         sys.stdout.write("%d\t" % i)
#     sys.stdout.write("\n")


# ndarray.reshape(data_row_num, data_columns_num) -> 각 데이터의 2차원 배열을 1차원 배열로 만들어 주는 과정
X_train = X_train.reshape(X_train.shape[0], 784).astype("float32") / 255
X_test = X_test.reshape(X_test.shape[0], 784).astype("float32") / 255


Y_train = np_utils.to_categorical(Y_train, 10)  # One hot encoding
Y_test = np_utils.to_categorical(Y_test, 10)    # One hot encoding

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from keras.models import Sequential

# inputs = Input(shape=(X_train.shape[0],))
# hidden = Dense(512, activation="relu")(inputs)
# output = Dense(10, activation="softmax")(hidden)
# model = Model(inputs=inputs, outputs=output)
model = Sequential()
model.add(Dense(512, input_dim=784, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
MODEL_DIR = "./model/"
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = "./model/{epoch:02d}-{val_loss:4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath,
                               monitor="val_loss",
                               verbose=1,
                               save_best_only=True)
early_stopping_callback = EarlyStopping(monitor="val_loss",
                                        patience=10)
with tf.device("/device:GPU:0"):
    history = model.fit(X_train, Y_train,
                        validation_data=(X_test, Y_test),
                        epochs=30, batch_size=200, verbose=0,
                        callbacks=[early_stopping_callback, checkpointer])

print("\ntest acc %4f" % (model.evaluate(X_test, Y_test)[1]))