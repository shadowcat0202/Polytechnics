import pandas as pd
import matplotlib.pyplot as plt
import seaborn


def show_survive_rate(df_train, feature):
    df_survive = df_train.loc[df_train["Survived"] == "Alive"]
    # df_dead = df_train.loc[df_train["Survived"] == 0]

    sur_info = df_survive[feature].value_counts(sort=False)
    print(sur_info)

    category = sur_info.index
    plt.title("Survival Rate in Total ({0})".format(feature))
    plt.pie(sur_info, labels=category, autopct="%0.1f%%")
    plt.show()


def showCountPlt(df_train, feature):
    seaborn.countplot(data=df_train, x=feature, hue="Survived")
    plt.show()
    pass


def showSexServive(df_train):
    """
        예 1) 성별에 따른 생존 비율
            ➢남성 생존자 수 / 남성 승객 수
            ➢여성 생존자 수 / 여성 승객 수
        예 2) 좌석등급에 따른 생존 비율
            ➢ 1등급 생존자 수 / 1등급 승객 수
            ➢ 2등급 생존자 수 / 2등급 승객 수
            ➢ 3등급 생존자 수 / 3등급 승객 수
    """
    df_male = df_train.loc[df_train["Sex"] == "male"]  # 남자만 걸러내기
    # print(df_man)
    # df_dead = df_train.loc[df_train["Survived"] == 0]

    sur_info = df_male["Survived"].value_counts(sort=False)
    # print(sur_info)
    #
    category = sur_info.index
    plt.title("Man Survival Rate in Total ({0})".format("Alive"))
    plt.pie(sur_info, labels=category, autopct="%0.1f%%")
    plt.show()
    # =============여성==========================================
    df_female = df_train.loc[df_train["Sex"] == "female"]
    # print(df_man)
    # df_dead = df_train.loc[df_train["Survived"] == 0]

    sur_info = df_female["Survived"].value_counts(sort=False)
    # print(sur_info)
    #
    category = sur_info.index
    plt.title("Man Survival Rate in Total ({0})".format("Alive"))
    plt.pie(sur_info, labels=category, autopct="%0.1f%%")
    plt.show()


def showPclassServived(df_train):
    Pclasss = []
    Pclasss.append(df_train.loc[df_train["Pclass"] == "1st"])
    Pclasss.append(df_train.loc[df_train["Pclass"] == "2nd"])
    Pclasss.append(df_train.loc[df_train["Pclass"] == "3th"])

    pclass_info = []
    for i in range(len(Pclasss)):
        pclass_info.append(Pclasss[i]["Survived"].value_counts(sort=False))

    categorys = []
    for i in range(len(Pclasss)):
        categorys.append(pclass_info[i].index);

    for i in range(len(Pclasss)):
        plt.subplot(3, 1, i+1)
        plt.title("1st Survival Rate in Total ({0})".format("Alive"))
        plt.pie(pclass_info[i], labels=categorys[i], autopct="%0.1f%%")

    plt.show()


def start():
    df_train = pd.read_csv('./dataset/titanic/train.csv')
    df_test = pd.read_csv('./dataset/titanic/test.csv')
    # print(df_train.head())
    # print(df_test.head())
    #
    # print(df_train.info())

    # print(df_test.info())
    #
    # print(df_train.corr())

    # print(df_train["Survived"].value_counts())

    # 생존 그룹과 그렇지 않은 그룹으로 나눈다.
    # survive = df_train.loc[df_train["Survived"] == 1].copy()
    # dead = df_train.loc[df_train["Survived"] == 0].copy()
    # # 히스토 그램
    # plt.hist(survive["Pclass"], alpha=0.5, label='Survived')
    # plt.hist(dead["Pclass"], alpha=0.5, label='Dead')
    # plt.legend(loc='best')

    # seaborn.set_style(style="darkgrid")
    # showCountPlt(df_train, "Pclass")
    # showCountPlt(df_train, "Sex")
    # showCountPlt(df_train, "Age")
    # showCountPlt(df_train, "SibSp")
    # showCountPlt(df_train, "Parch")
    # showCountPlt(df_train, "Fare")
    # showCountPlt(df_train, "Embarked")

    df_train["Pclass"] = df_train["Pclass"] \
        .replace(1, "1st") \
        .replace(2, "2nd") \
        .replace(3, "3th")
    df_train["Survived"] = df_train["Survived"] \
        .replace(1, "Alive") \
        .replace(0, "Dead")

    # show_survive_rate(df_train, "Pclass")
    # show_survive_rate(df_train, "Sex")
    # show_survive_rate(df_train, "SibSp")
    # show_survive_rate(df_train, "Parch")
    # show_survive_rate(df_train, "Embarked")

    showSexServive(df_train)
    # showPclassServived(df_train)
