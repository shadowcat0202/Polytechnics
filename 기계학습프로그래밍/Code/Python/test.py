import pandas as pd
import matplotlib.pyplot as plt
import seaborn

from sklearn import tree
import seaborn as sns
import numpy as np

train_df = pd.read_csv("./dataset/titanic/train.csv")
# print(train_df.isna().sum())    #Age 177    Cabin 684    Embarked 2
train_df["Age"].fillna(train_df["Age"].mean(), inplace=True)
train_df["Cabin"].fillna("N", inplace=True)
# print(train_df.isna().sum())    #Embarked 2

# print(train_df.groupby(["Sex", "Survived"])["Survived"].count())
"""
Sex     Survived
female  0            81
        1           233
male    0           468
        1           109
"""

# print(train_df.groupby(["Pclass", "Survived"])["Survived"].count())


def age_category(age):
    out = 0
    if age <= -1:
        out = 0
    elif age <= 5:
        out = 1
    elif age <= 12:
        out = 2
    elif age <= 18:
        out = 3
    elif age <= 25:
        out = 4
    elif age <= 35:
        out = 5
    elif age <= 60:
        out = 6
    else:
        out = 7
    return out


train_df["AgeGroup"] = train_df["Age"].apply(lambda x: age_category(x))  # 해당 DataFrame에 맞춰 함수 적용
# print(train_df["AgeGroup"].value_counts())
"""
Young Adult    373
Adult          195
Student        162
Teenager        70
Baby            44
Child           25
Elderly         22
"""

# seaborn.barplot(x="AgeGroup", y="Survived", hue="Sex", data=train_df)
# plt.show()

# One-Hot Encoding
from sklearn.preprocessing import LabelEncoder
def encoding(df):
    features = ["Cabin", "Sex", "Embarked"]
    for feature in features:
        encoder = LabelEncoder()
        df[feature] = encoder.fit_transform(df[feature])
    return df
train_df = encoding(train_df)
# print(train_df)

train_df.drop(["Name", "Ticket","Fare"], axis=1, inplace=True)

family = []
for i in range(len(train_df)):
    if train_df.loc[i, "SibSp"] >= 4:
        family.append(2)
    elif train_df.loc[i, "SibSp"] >= 2:
        family.append(1)
    else:
        family.append(0)

train_df["Family"] = family

X = train_df.drop(["Survived"], axis=1)
y = train_df["Survived"]


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(max_depth=5, min_samples_split=2)
rf_model.fit(X,y)

# from sklearn.tree import DecisionTreeClassifier
# dt_model = DecisionTreeClassifier()
# #
# param = {
#     "max_depth": [3, 4, 5, 6],
#     "min_samples_split": [2, 3, 4, 5]
# }
#
# from sklearn.model_selection import GridSearchCV
# rf_gs = GridSearchCV(rf_model, param_grid=param, cv=5, refit=True)
# dt_gs = GridSearchCV(dt_model, param_grid=param, cv=5, refit=True)
# rf_gs.fit(X, y)
# dt_gs.fit(X, y)
#
# print("Random Forest")
# print(rf_gs.best_score_)
# print(rf_gs.best_params_)
# print("\n\ndecision tree")
# print(dt_gs.best_score_)
# print(dt_gs.best_params_)

test_df = pd.read_csv("./dataset/titanic/test.csv")
pid = test_df["PassengerId"]
test_df["Age"].fillna(test_df["Age"].mean(), inplace=True)
test_df["Cabin"].fillna("N", inplace=True)
test_df["AgeGroup"] = test_df["Age"].apply(lambda x: age_category(x))  # 해당 DataFrame에 맞춰 함수 적용
test_df = encoding(test_df)
test_df.drop(["Name", "Ticket", "Fare"], axis=1, inplace=True)
family = []
for i in range(len(test_df)):
    if test_df.loc[i, "SibSp"] >= 4:
        family.append(2)
    elif test_df.loc[i, "SibSp"] >= 2:
        family.append(1)
    else:
        family.append(0)

test_df["Family"] = family
# print(test_df.isna().sum())
test_result = rf_model.predict(test_df)

submit = pd.DataFrame({"PassengerId": pid, "Survived": test_result})
submit.to_csv("./dataset/titanic/submit.csv", index=False)

hit = 0
miss = 0
my = pd.read_csv("./dataset/titanic/submit.csv")
test_gt = pd.read_csv("./dataset/titanic/wow.csv")

for i in range(len(test_result)):
    if test_gt["Survived"][i] == my["Survived"][i]:
        hit += 1
    else:
        miss += 1

print(hit, miss, round(hit / (hit + miss), 4))