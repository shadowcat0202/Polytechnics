import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns
import numpy as np
from sklearn.datasets import load_iris


def salmonbass():
    df = pd.read_csv("dataset/ai_score_data.csv")
    # print(df.shape)
    # print(df.info())
    # print(df.describe())
    #
    # print(df["Math"].mean())
    # print(df["Math"].median())
    # print(df.corr())

    canvas = plt.figure(figsize=(7.0, 7.0))
    plt.xlabel("math")
    plt.ylabel("english")

    for i in range(len(df["Sex"])):
        if df.loc[i, "Sex"] == "M":
            plt.scatter(df.loc[i, "Math"], df.loc[i, "English"], color="blue")  # 남
        else:
            plt.scatter(df.loc[i, "Math"], df.loc[i, "English"], color="red")  # 녀

    plt.show()

    # ==================================1개의 featuer를 가지고 scatter============================
    df = pd.read_csv("dataset/salmon_bass_data.csv")
    plt.hist(df["Length"], alpha=.2)
    # plt.show()  #이건 알아보기 어렵다

    salmon = df.loc[df["Class"] == "Salmon"]  # Class feature가 Salmon인 row만 가져온다
    bass = df.loc[df["Class"] == "Bass"]  # Class feature가 Bass인 row만 가져온다
    # print(salmon)

    plt.hist(salmon["Length"], bins=20, alpha=.5, label="Salmon")  # bins몰?루 alpha=투명도 label=이름 붙여주기
    plt.hist(bass["Length"], bins=20, alpha=.5, label="Bass")
    plt.legend(loc="best")
    plt.show()

    # ===========================2개의 featuer로 scatter==============================
    plt.title("Scatter")
    plt.xlabel("Length")
    plt.ylabel("Lightness")

    plt.scatter(salmon["Length"], salmon["Lightness"], color="blue", label="Salmon")
    plt.scatter(bass["Length"], bass["Lightness"], color="red", label="bass")
    plt.legend(loc="best")
    plt.show()

    # ============================가장 간단한 분류 예제===========================

    X = [[0, 0], [1, 1]]
    Y = [0, 1]
    for i in range(len(X)):
        if Y[i] == 0:
            plt.scatter(X[i][0], X[i][1], color='red')
        elif Y[i] == 1:
            plt.scatter(X[i][0], X[i][1], color='blue')
    plt.xlabel('X[0] Features')
    plt.xticks(np.arange(-1, 3, 1))
    plt.ylabel('X[1] Features')
    plt.yticks(np.arange(-1, 3, 1))
    plt.grid()
    plt.show()

    dtree = tree.DecisionTreeClassifier()
    dtree = dtree.fit(X, Y)

    tree.plot_tree(dtree)
    plt.show()

    test_X = [[2, 2], [1, 1], [0, 0], [0, 1]]
    print(dtree.predict(test_X))

    df = pd.read_csv("dataset/salmon_bass_data.csv")
    X = []
    Y = []

    for i in range(len(df)):
        fish = [df.loc[i, "Length"], df.loc[i, "Lightness"]]
        X.append(fish)  # 물고기 특징 리스트
        if df.loc[i, "Class"] == "Salmon":  # 생선 종류
            Y.append("Salmon")
        else:
            Y.append("Bass")

    dtree = tree.DecisionTreeClassifier()
    dtree = dtree.fit(X, Y)

    plt.figure(figsize=(10, 10))
    tree.plot_tree(dtree, fontsize=8, filled=True,
                   class_names=['Salmon', 'Bass'],
                   feature_names=['Lightness', 'Length'])
    plt.show()

    print(dtree.predict([[26, 1.2], [3, 4.0], [7, 6.0]]))


def iris():
    data = pd.read_csv("dataset/Iris.csv")
    # plt.figure(figsize=(15, 15))
    # sns.heatmap(data=data.corr(), annot=True,
    #             fmt='.2f', linewidths=.5, cmap='Blues')
    #
    # df = data.drop(["Id"], axis=1)
    # sns.pairplot(df, hue="Species")
    # plt.show()

    # print(data["Species"].unique())   #클래스 종류 보여주기
    # print(data.info())
    X = data.drop(["Id", "Species"], axis=1)
    Y = data["Species"]

    #
    dtree = tree.DecisionTreeClassifier()
    dtree = dtree.fit(X, Y)

    # 테스트 데이터
    # pred = [
    #     [5.7, 4.4, 1.5, 0.4],
    #     [4.8, 3.1, 1.6, 0.2],
    #     [5.0, 2.0, 3.5, 1.0],
    #     [5.5, 2.6, 4.6, 1.2],
    #     [6.5, 3.0, 5.8, 2.2],
    #     [4.8, 3.0, 4.4, 0.3],
    #     [6.6, 3.0, 4.4, 1.4],
    #     [6.0, 2.2, 5.0, 1.5],
    #     [6.1, 2.6, 5.6, 1.4],
    #     [5.9, 3.0, 5.1, 1.8]
    # ]


def stub():
    n = 6
    X = [[1, 3], [3, 5], [5, 7], [3, 1], [5, 3], [7, 5]]
    y = [1,1,1,0,0,0]
    datapoint = [2,4]
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    sns.regplot(x=X, y=y, data)
    model = LogisticRegression()
    model.fit(X, y)



if __name__ == '__main__':
    # iris = load_iris()
    # X, y = iris.data, iris.target
    # clf = tree.DecisionTreeClassifier()
    # clf = clf.fit(X, y)
    #
    # print(X)
    # print(y)
    # plt.figure(figsize=(10, 10))
    # tree.plot_tree(clf, fontsize=10, filled=True)
    # plt.show()
    # salmonbass()
    iris()
