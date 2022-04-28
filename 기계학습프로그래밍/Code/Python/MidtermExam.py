import pandas as pd
import random

import sklearn.tree


def score():
    return random.randint(0, 100)


def useDataFrame():
    df = pd.read_csv("./dataset/salmon_bass_data.csv")
    print(df)

    dic = {"name": "전세환"}
    print(dic)
    print(dic["name"])
    dic["name"] = "변경"
    print(dic)

    dicRowList = ["이름1", "이름2", "이름3", "이름4"]
    dicList = [1, 2, 3, 4]
    dic["num"] = dicList
    print(dic)

    dic["id"] = [0, 1]
    dic["id"].append(2)
    dic["id"] = [5, 6, 7]
    print(dic)

    del dic
    dic = {"name": dicRowList, "id": dicList}

    df = pd.DataFrame(dic)
    print(df)

    df["국어"] = 0
    df["국어"] = [score(), score(), score(), score()]
    print(df)

    del df
    dic = {}
    nameList = ["이름1", "이름2", "이름3", "이름4"]
    dic["name"] = nameList
    dic["날짜"] = ["2022-1", "2022-2", "2022-3", "2022-4"]
    dic["점수"] = [score(), score(), score(), score()]
    df = pd.DataFrame(dic)
    print(df)

    df.loc[0] = ["변경1", "2022-10", score()]
    print(df)
    df.loc[4] = 0
    print(df)

    df.loc[df.shape[0]] = ["끝행", "2022-end", score()]
    print(df)

    df.loc[2, "날짜"] = "2023-1"
    print(df)

    del df

    dic = {"name": [], "date": [], "score": []}
    df = pd.DataFrame(dic)
    print(df)
    for i in range(len(df), 23):
        df.loc[i] = ["name-{0}".format(i), "2022-{0}".format(i), score()]
    print(df)

    df.drop(10, inplace=True)
    print(df)

    df.to_csv("./test.csv")


def ai_score_data():
    df = pd.read_csv("../dataset/ai_score_data.csv")
    print(df.shape)
    print(df.info())  # DataFrame에서는 str은 Object로 표현

    print(df.describe())

    print(f"{df['Math'].mean():.2f}")

    print(df[["Math", "English"]].corr(), "\n")

    import matplotlib.pyplot as plt
    # plt.hist(df["Math"])
    # plt.scatter(df["Math"], df["English"])
    canvas = plt.figure(figsize=(7.0, 7.0))
    plt.xlabel("Math")
    plt.ylabel("English")

    for i in range(len(df.Sex)):
        if df.loc[i, "Sex"] == "M":
            plt.scatter(df.loc[i, "Math"], df.loc[i, "English"], color="blue")
        else:
            plt.scatter(df.loc[i, "Math"], df.loc[i, "English"], color="red")
    plt.show()


def SalmonBass():
    import matplotlib.pyplot as plt
    df = pd.read_csv("../dataset/salmon_bass_data.csv")
    salmon = df.loc[df["Class"] == "Salmon"]
    bass = df.loc[df["Class"] == "Bass"]

    plt.subplot(1, 2, 1)
    plt.hist(salmon["Length"], bins=20, alpha=0.5, label="Salmon")
    plt.hist(bass["Length"], bins=20, alpha=0.5, label="Bass")
    plt.legend(loc="best")

    plt.subplot(1, 2, 2)
    plt.hist(salmon["Lightness"], bins=20, alpha=0.5, label="Salmon")
    plt.hist(bass["Lightness"], bins=20, alpha=0.5, label="Bass")
    plt.legend(loc="best")
    plt.show()

    plt.xlabel("Length")
    plt.ylabel("Lightness")

    plt.scatter(salmon["Length"], salmon["Lightness"], color='blue', label="Salmon")
    plt.scatter(bass["Length"], bass["Lightness"], color='red', label="Bass")

    plt.legend(loc="best")
    plt.show()


def DecisionTree_data_Visualization():
    from sklearn import tree
    import matplotlib.pyplot as plt
    import numpy as np
    x = [[0, 0], [1, 1]]
    y = [0, 1]

    plt.subplot(1, 2, 1)
    for i in range(len(x)):
        if y[i] == 0:
            plt.scatter(x[i][0], x[i][1], color="blue")
        else:
            plt.scatter(x[i][0], x[i][1], color="red")
    plt.xlabel("x[0]")
    plt.ylabel("x[1]")
    plt.xticks(np.arange(-1, 3, 1))
    plt.yticks(np.arange(-1, 3, 1))
    plt.grid()

    plt.subplot(1, 2, 2)
    dtree = tree.DecisionTreeClassifier()
    dtree = dtree.fit(x, y)
    tree.plot_tree(dtree)
    plt.show()


def IRIS_DecisionTreeClassifier():
    from sklearn import tree
    import matplotlib.pyplot as plt
    iris_df = pd.read_csv("./dataset/Iris.csv")
    # print(iris_df.info())
    # print(iris_df["Species"].unique())

    x = iris_df.drop(["Id", "Species"], axis=1)
    y = iris_df["Species"]

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x, y)

    pred_x = [[5.7, 4.4, 1.5, 0.4],
              [4.8, 3.1, 1.6, 0.2],
              [5.0, 2.0, 3.5, 1.0],
              [5.5, 2.6, 4.4, 1.2],
              [6.5, 3.0, 5.8, 2.2],
              [4.8, 3.0, 1.4, 0.3],
              [6.6, 3.0, 4.4, 1.4],
              [6.0, 2.2, 5.0, 1.5],
              [6.1, 2.6, 5.6, 1.4],
              [5.9, 3.0, 5.1, 1.8]]
    pred_y = clf.predict(pred_x)
    print(pred_y)

    # plt.figure(figsize=(15, 15))
    # tree.plot_tree(clf, fontsize=10, filled=True)
    # plt.show()

def IRIS_RandomForestClassifier():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import tree
    iris_df = pd.read_csv("../dataset/Iris.csv")
    x = iris_df.drop(["Id", "Species"], axis=1)
    y = iris_df["Species"]

    clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0)
    clf.fit(x,y)



# DecisionTree_data_Visualization()
IRIS_DecisionTreeClassifier()
# IRIS_RandomForestClassifier()
