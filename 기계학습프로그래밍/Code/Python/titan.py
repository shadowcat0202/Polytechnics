import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import numpy as np

df_train = pd.read_csv('./dataset/titanic/train.csv')
df_train = df_train.drop(["Name", "PassengerId", "Ticket", "Fare", "Cabin"], axis=1)
df_train["Age"].fillna(df_train["Age"].mean(), inplace=True)
df_train["Embarked"] = df_train["Embarked"].fillna("S")  # 가장 많은 S로 통일
df_train["Sex"] = df_train["Sex"].map({"male": 0, "female": 1})
df_train["Embarked"] = df_train["Embarked"].map({"Q": 0, "C": 1, "S": 2})

# print(df_train.isna().sum())
from sklearn.ensemble import RandomForestClassifier

# classifire = DecisionTreeClassifier()
classifire = RandomForestClassifier(max_depth=4, n_estimators=300)
# classifire = LinearSVC()

X = df_train.drop(["Survived"], axis=1)
y = df_train["Survived"]

classifire.fit(X, y)
print("학습 정확도:", round(classifire.score(X, y), 2))

df_test = pd.read_csv('./dataset/titanic/test.csv')
pid = df_test["PassengerId"]  # [!!!]submit.csv 를 만들어줄 때 PassengerId가 필요하기 때문에 drop하기 전에 저장해둔다
# predict하기 위한 test data도 columns를 동일하게 맞춰주어야 한다
df_test = df_test.drop(["Name", "PassengerId", "Ticket", "Fare", "Cabin"], axis=1)
df_test["Age"].fillna(df_test["Age"].mean(), inplace=True)
df_test["Embarked"] = df_test["Embarked"].fillna("S")
df_test["Sex"] = df_test["Sex"].map({"male": 0, "female": 1})
df_test["Embarked"] = df_test["Embarked"].map({"Q": 0, "C": 1, "S": 2})

test_result = classifire.predict(df_test)

submit = pd.DataFrame({"PassengerId": pid, "Survived": test_result})
# index=False를 안하면 csv파일 0번쨰 columns에 index를 붙여준다 즉 제출용에 맞게 설정해주어야 한다
submit.to_csv("./dataset/titanic/submit.csv", index=False)

hit = 0
miss = 0
# test_gt = pd.read_csv("./dataset/titanic/wow.csv")
base = pd.read_csv("./dataset/titanic/submit.csv")
#
# for i in range(len(test_result)):
#     if base["Survived"][i] == test_gt["Survived"][i]:
#         hit += 1
#     else:
#         miss += 1

print(hit, miss, round(hit / (hit + miss), 4))


def myPredict(x):
    if x["Sex"] == "male":
        return 0
    else:
        return 1



# pyplot 그려주기 위해서 replace작업==================================================================
df_train["Pclass"] = df_train["Pclass"] \
    .replace(1, "1st") \
    .replace(2, "2nd") \
    .replace(3, "3th")
df_train["Survived"] = df_train["Survived"] \
    .replace(1, "Alive") \
    .replace(0, "Dead")

for i in range(len(df_train)):
    df_train.loc[i, "Age"] = int(df_train.loc[i, "Age"] / 10)
df_train["Age"] = df_train["Age"].map({0: "0~9", 1: "10~19", 2: "20~29",
                                       3: "30~39", 4: "40~49", 5: "50~59",
                                       6: "60~69", 7: "Old", 8: "Old"})


def showCountPlot():
    seaborn.set_style(style="darkgrid")
    feat = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    i = 1
    for f in feat:
        plt.subplot(2, 4, i)
        i += 1
        seaborn.countplot(data=df_train, x=f, hue="Survived")

    plt.show(block=True)


# showCountPlot()


def show_survive_rat():
    feat = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]
    df_survive = df_train.loc[df_train["Survived"] == "Alive"]
    df_dead = df_train.loc[df_train["Survived"] == "Dead"]
    i = 1
    for f in feat:
        plt.subplot(2, 3, i)
        i += 1
        sur_info = df_survive[f].value_counts(sort=False)

        category = sur_info.index
        plt.title("Survival Rate in Total ({0})".format(f))
        plt.pie(sur_info, labels=category, autopct="%0.1f%%")

    plt.show(block=True)


# show_survive_rat()


def show_group_rate(feature):
    df_survive = df_train.loc[df_train["Survived"] == "Alive"]
    df_dead = df_train.loc[df_train["Survived"] == "Dead"]

    sur_info = df_survive[feature].value_counts(sort=False)
    dead_info = df_dead[feature].value_counts(sort=False)

    fig = plt.figure()
    plt.title("Survival rete of " + feature)
    for i, index in enumerate(sur_info.index):  # 리스트 요소를 꺼내는것과 동시에 인덱스도 같이 돌린다
        # fig.add_subplot(len(sur_info)//4, len(sur_info) // 2, i + 1)  # Age 그래프 화면의 어느 위치
        # 생존자와 사망자의 해당 클래스로 파이 그래프를 그리겠다
        fig.add_subplot(1, len(sur_info.index), i + 1)  # Pclass
        plt.pie([sur_info[index], dead_info[index]],
                labels=["Survived", "Dead"],
                autopct="%0.1f%%")
        plt.title(str(index))

    print("survive:")
    print(sur_info)
    print("dead:")
    print(dead_info)
    plt.show()

# show_group_rate("Pclass")
# show_group_rate("Sex")
# show_group_rate("Age")
