import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


def imshow_all_columns(dataset, cols_name):
    fig = plt.figure()
    rows = 3
    cols = 4
    for i, name in enumerate(cols_name):
        ax1 = fig.add_subplot(rows, cols, i + 1)
        ax1.set_title(name)
        ax1.hist(dataset[name])
    fig.tight_layout()
    plt.show()


def dataset(data):
    X = data[:, :-1]
    Y = data[:, -1]


def make_model(in_dim,out_dim):
    m = Sequential()
    # diabetes ==============================================================
    # m.add(Dense(12, input_dim=dim, activation="relu"))
    # m.add(Dense(8, activation="relu"))
    # m.add(Dense(1, activation="sigmoid"))
    # m.compile(loss="binary_crossentropy",
    #               optimizer="adam",
    #               metrics=["accuracy"])

    # IRIS ==============================================================
    m.add(Dense(12, input_dim=in_dim, activation="relu"))
    m.add(Dense(8, activation="relu"))
    m.add(Dense(out_dim, activation="sigmoid"))
    m.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    m.summary()
    return m

def Pima_indians_diabetes_use_NN():
    df = np.loadtxt("./dataset/Pima-indians-diabetes.csv", delimiter=",")
    train_X = df[:700, :-1]
    train_Y = df[:700, -1]
    test_X = df[700:, :-1]
    test_Y = df[700:, -1]
    scaler = StandardScaler()
    # Min-Max 정규화 (Normalization)
    # x' = (x-min) / (max - min)    가장 작은값은 0 가장 큰 값을 1로 변경
    # 테스트 데이터를 정규화 해주었다면 당근빠따 학습데이터도 정규화를 해주어야 한다
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.fit_transform(test_X)

    model = make_model(len(train_X[0]), len(train_Y[0]))
    model.fit(train_X, train_Y, epochs=200, batch_size=50)

    print(model.predict([[6, 148, 72, 35, 0, 33.6, 0.627, 50]]))
    print(model.evaluate(test_X, test_Y))


def IRIS_use_NN():
    df = pd.read_csv("./dataset/Iris.csv")
    df = df.values

    X = df[:, : -1]
    Y = df[:, -1]

    # train_X = df[:120, :-1]
    # train_Y = df[:120, -1]
    # test_X = df[120:, :-1]
    # test_Y = df[120:, -1]
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        shuffle=True)
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.fit_transform(test_X)

    e = LabelEncoder()
    e.fit(train_Y)
    train_Y = e.transform(train_Y)
    print(train_Y)
    train_Y = tf.keras.utils.to_categorical(train_Y)

    b = LabelEncoder()
    b.fit(test_Y)
    test_Y = b.transform(test_Y)
    print(test_Y)
    test_Y = tf.keras.utils.to_categorical(test_Y)

    model = make_model(len(train_X[0]), len(train_Y[0]))
    model.fit(train_X, train_Y, epochs=200, batch_size=50)
    print(model.evaluate(test_X, test_Y))






# df = pd.read_csv("./dataset/Pima-indians-diabetes.csv")
# column이 없을때는 만들어 준다 ㅋ
# df = pd.read_csv("./dataset/pima-indians-diabetes.csv",
#                  names=["Pregnancies", "Glucose", "BloodPressure",
#                         "SkinThickness", "Insulin", "BMI",
#                         "Pedigree", "Age", "Class"])

# df.rename(columns={"Outcome": "Class"}, inplace=True)
# df.columns = ["Pregnancies", "Glucose", "BloodPressure",
#               "SkinThickness", "Insulin", "BMI",
#               "Pedigree", "Age", "Class"]
# df_columns = list(df.columns)
# imshow_all_columns(df, df_columns)
# print(df[["Pregnancies","Class"]])
# df_diabetes = df.loc[df["Class"] == 1]
# df_normal = df.loc[df["Class"] == 0]

# print(df[["Pregnancies", "Class"]].groupby(["Pregnancies"], as_index=False).mean().sort_values(by="Pregnancies",ascending=True))

# hist1 = df_diabetes["Pregnancies"].value_counts().sort_index()
# hist2 = df_normal["Pregnancies"].value_counts().sort_index()
# hist3 = hist1 / (hist1 + hist2)
# print(hist3.fillna(1))

# plt.figure(figsize=(12, 12))
# seaborn.heatmap(df.corr(), annot=True,
#                 cmap=plt.cm.inferno,
#                 linewidths=0.1, linecolor="white")
# plt.show()

# grid = seaborn.FacetGrid(df, col="Class")
# grid.map(plt.hist, "Glucose", bins=10)
# plt.show()

IRIS_use_NN()