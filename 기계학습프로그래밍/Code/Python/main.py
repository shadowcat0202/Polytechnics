import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree, svm
import numpy as np


def ai_score():
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


def salmonbasss():
    # ==================================1개의 featuer를 가지고 scatter============================
    df = pd.read_csv("dataset/salmon_bass_data.csv")
    # plt.hist(df["Length"], alpha=.2)
    plt.show()  # 이건 알아보기 어렵다

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
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    from sklearn import tree
    iris = load_iris()
    X, y = iris.data, iris.target
    # n_estimators = 트리개수 max_depth=트리 깊이 random_state=0으로 하는게 좋다라고 sklearn에 나와있다
    clf = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)
    clf = clf.fit(X, y)
    print(clf.estimators_)
    print(len(clf.estimators_))
    result = clf.predict([[4.6, 3.1, 1.5, 0.2],
                          [7.0, 3.2, 4.7, 1.4],
                          [6.3, 3.3, 6.0, 2.5]])
    print(result)



df_train = pd.read_csv("./dataset/titanic/train.csv")
# print(df_train.corr())
df_train = df_train.drop(["Name", "PassengerId", "Ticket", "Fare", "Cabin"], axis=1)
df_train["Age"].fillna(df_train["Age"].mean(), inplace=True)
df_train["Embarked"] = df_train["Embarked"].fillna("S")  # 가장 많은 S로 통일
df_train["Sex"] = df_train["Sex"].map({"male": 0, "female": 1})
df_train["Embarked"] = df_train["Embarked"].map({"Q": 0, "C": 1, "S": 2})
# print(df_train.isna().sum())
# seaborn.countplot(data=df_train, x="SibSp", hue="Survived")
# plt.show()
# 가족 인원수 (혼자1, 핵가족2~3, 대가족4~)
family = []
for i in range(len(df_train)):
    if df_train.loc[i, "SibSp"] >= 4:
        family.append(2)
    elif df_train.loc[i, "SibSp"] >= 2:
        family.append(1)
    else:
        family.append(0)

df_train["Family"] = family

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth=4, n_estimators=600)
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression(solver='lbfgs')

# model = svm.SVC(kernel='linear')


X = df_train.drop(["Survived"], axis=1)
y = df_train["Survived"]

model.fit(X,y)

df_test = pd.read_csv('./dataset/titanic/test.csv')
pid = df_test["PassengerId"]  # [!!!]submit.csv 를 만들어줄 때 PassengerId가 필요하기 때문에 drop하기 전에 저장해둔다
# predict하기 위한 test data도 columns를 동일하게 맞춰주어야 한다
df_test = df_test.drop(["Name", "PassengerId", "Ticket", "Fare", "Cabin"], axis=1)
df_test["Age"].fillna(df_test["Age"].mean(), inplace=True)
df_test["Embarked"] = df_test["Embarked"].fillna("S")
df_test["Sex"] = df_test["Sex"].map({"male": 0, "female": 1})
df_test["Embarked"] = df_test["Embarked"].map({"Q": 0, "C": 1, "S": 2})
family = []
for i in range(len(df_test)):
    if df_test.loc[i, "SibSp"] >= 4:
        family.append(2)
    elif df_test.loc[i, "SibSp"] >= 2:
        family.append(1)
    else:
        family.append(0)

df_test["Family"] = family

test_result = model.predict(df_test)

predic = pd.DataFrame({"PassengerId": pid, "Survived": test_result})
# index=False를 안하면 csv파일 0번쨰 columns에 index를 붙여준다 즉 제출용에 맞게 설정해주어야 한다
predic.to_csv("./dataset/titanic/my_test.csv", index=False)
wow = pd.read_csv("./dataset/titanic/wow.csv")

hit = 0
miss = 0
for i in range(len(test_result)):
    if predic["Survived"][i] == wow["Survived"][i]:
        hit += 1
    else:
        miss += 1

print(hit, miss, round(hit / (hit + miss), 4))




