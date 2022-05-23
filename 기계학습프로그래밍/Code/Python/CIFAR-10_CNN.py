import numpy
import tensorflow as tf

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten

from tensorflow.keras.models import load_model
from keras import initializers
from torch.nn import Dropout

device = tf.device('cuda')

numpy.random.seed(0)

MY_EPOCH = 10
MY_BATCHSIZE = 200
filename = f"./model/cnn_e({MY_EPOCH}).h5"

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()


def make_model():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu',
                     input_shape=(32, 32, 3),
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model


def train(model, X, Y):
    X = X / 255.
    Y = tf.keras.utils.to_categorical(Y, 10)
    history = model.fit(X, Y, epochs=MY_EPOCH, batch_size=MY_BATCHSIZE)
    filename = "./model/cnn_e({0}).h5".format(MY_EPOCH)
    model.save(filename)
    return history


def test_all(x, y):
    model = load_model(filename)
    x = x / 255.
    y = tf.keras.utils.to_categorical(y, 10)
    test_loss, test_acc = model.evaluate(x, y)
    return test_loss, test_acc


cnn = make_model()
train(cnn, train_images, train_labels)
print(test_all(test_images, test_labels))
