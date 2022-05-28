import tensorflow.keras.optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import tensorflow as tf

hidden_node = [32, 8, 4, 128]


def my_test_cnn_model_Sequential(input_shape=None, output_shape=None, lr=0.001):
    try:
        if input_shape is None:
            raise Exception("input_shape is None")
        if output_shape is None:
            raise Exception("output_shape is None")
    except Exception as e:
        print(e)

    model = Sequential()
    model.add(Conv2D(hidden_node[0], (3, 3), activation='tanh',
                     input_shape=(input_shape[1], input_shape[2], input_shape[3]),
                     padding='same'))  # 32 x 32 x 3
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # model.add(Conv2D(hidden_node[1], (3, 3), activation='tanh', padding='same'))
    # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # model.add(Conv2D(hidden_node[2], (3, 3), activation='tanh', padding='same'))
    # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(hidden_node[3], activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape[1], activation="relu"))
    model.summary()

    opt = tensorflow.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss="mean_squared_error",
                  optimizer=opt,
                  metrics=["accuracy"])
    return model


def my_test_cnn_model_functional(input_shape, output_shape, lr=0.001):
    try:
        if input_shape is None:
            raise Exception("input_shape is None")
        if output_shape is None:
            raise Exception("output_shape is None")
    except Exception as e:
        print(e)

    # if len(input_shape) == 3:
    X = tf.keras.layers.Input(shape=(input_shape[1], input_shape[2], input_shape[3]))
    # tanh
    H = tf.keras.layers.Conv2D(hidden_node[0], kernel_size=(5, 5), activation="tanh", padding='same')(X)
    H = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(H)

    H = tf.keras.layers.Conv2D(hidden_node[1], kernel_size=(5, 5), activation="tanh", padding='same')(H)
    H = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(H)
    H = tf.keras.layers.Flatten()(H)

    H = tf.keras.layers.Dense(hidden_node[2], activation="tanh")(H)

    Y = tf.keras.layers.Dense(output_shape[1], activation="relu")(H)  # 왜 sigmaid로 해야함?
    model = tf.keras.models.Model(X, Y)

    model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss="mean_squared_error",
                  optimizer=opt,
                  metrics=["accuracy"])

    return model


class eye_Net:
    def __init__(self):
        pass
    # https://www.kaggle.com/datasets/tauilabdelilah/mrl-eye-dataset?resource=download
    def model(self, lr=0.001):
        X = Sequential()
        X.add(Conv2D(8, (3, 3), activation='relu',  # Conv2D 필터 개수에 따른 차이는 미확인 상태
                     input_shape=(90, 90, 1),
                     padding='same'))  # input_shape = (None, 90, 90, 1)
        X.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))

        # X.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        # X.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))
        X.add(Flatten())

        X.add(Dense(16, activation='relu')) # 8일때 acc=0.5 언저리
        X.add(Dense(1, activation='sigmoid'))  # output = 1
        X.summary()

        opt = tensorflow.keras.optimizers.Adam(learning_rate=lr)
        X.compile(loss="binary_crossentropy", # class가 이진 분류 문제의 손실함수는 binary_crossentropy
                  optimizer=opt,
                  metrics=["accuracy"])
        # https://cheris8.github.io/artificial%20intelligence/DL-Keras-Loss-Function/
        # Binary classification
        # sigmoid
        # binary_crossentropy
        # Dog vs cat, Sentiemnt analysis(pos / neg)
        #
        # Multi-class, single-label classification
        # softmax
        # categorical_crossentropy
        # MNIST has 10 classes single label (one prediction is one digit)
        #
        # Multi-class, multi-label classification
        # sigmoid
        # binary_crossentropy
        # News tags classification, one blog can have multiple tags
        #
        # Regression to arbitrary values
        # None
        # mse
        # Predict house price(an integer / float point)
        #
        # Regression to values between 0 and 1
        # sigmoid
        # mse or binary_crossentropy
        # Engine health assessment where 0 is broken, 1 is new
        return X