import pandas as pd
import matplotlib.pyplot as plt
import seaborn

from sklearn import tree
import seaborn as sns
import numpy as np

# train_df = pd.read_csv("./dataset/titanic/train.csv")
# # print(train_df.isna().sum())    #Age 177    Cabin 684    Embarked 2
# train_df["Age"].fillna(train_df["Age"].mean(), inplace=True)
# train_df["Cabin"].fillna("N", inplace=True)
# # print(train_df.isna().sum())    #Embarked 2
#
# # print(train_df.groupby(["Sex", "Survived"])["Survived"].count())
# """
# Sex     Survived
# female  0            81
#         1           233
# male    0           468
#         1           109
# """
#
# # print(train_df.groupby(["Pclass", "Survived"])["Survived"].count())
#
#
# def age_category(age):
#     out = 0
#     if age <= -1:
#         out = 0
#     elif age <= 5:
#         out = 1
#     elif age <= 12:
#         out = 2
#     elif age <= 18:
#         out = 3
#     elif age <= 25:
#         out = 4
#     elif age <= 35:
#         out = 5
#     elif age <= 60:
#         out = 6
#     else:
#         out = 7
#     return out
#
#
# train_df["AgeGroup"] = train_df["Age"].apply(lambda x: age_category(x))  # 해당 DataFrame에 맞춰 함수 적용
# # print(train_df["AgeGroup"].value_counts())
# """
# Young Adult    373
# Adult          195
# Student        162
# Teenager        70
# Baby            44
# Child           25
# Elderly         22
# """
#
# # seaborn.barplot(x="AgeGroup", y="Survived", hue="Sex", data=train_df)
# # plt.show()
#
# # One-Hot Encoding
# from sklearn.preprocessing import LabelEncoder
# def encoding(df):
#     features = ["Cabin", "Sex", "Embarked"]
#     for feature in features:
#         encoder = LabelEncoder()
#         df[feature] = encoder.fit_transform(df[feature])
#     return df
# train_df = encoding(train_df)
# # print(train_df)
#
# train_df.drop(["Name", "Ticket","Fare"], axis=1, inplace=True)
#
# family = []
# for i in range(len(train_df)):
#     if train_df.loc[i, "SibSp"] >= 4:
#         family.append(2)
#     elif train_df.loc[i, "SibSp"] >= 2:
#         family.append(1)
#     else:
#         family.append(0)
#
# train_df["Family"] = family
#
# X = train_df.drop(["Survived"], axis=1)
# y = train_df["Survived"]
#
#
# from sklearn.ensemble import RandomForestClassifier
# rf_model = RandomForestClassifier(max_depth=5, min_samples_split=2)
# rf_model.fit(X,y)

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

# test_df = pd.read_csv("./dataset/titanic/test.csv")
# pid = test_df["PassengerId"]
# test_df["Age"].fillna(test_df["Age"].mean(), inplace=True)
# test_df["Cabin"].fillna("N", inplace=True)
# test_df["AgeGroup"] = test_df["Age"].apply(lambda x: age_category(x))  # 해당 DataFrame에 맞춰 함수 적용
# test_df = encoding(test_df)
# test_df.drop(["Name", "Ticket", "Fare"], axis=1, inplace=True)
# family = []
# for i in range(len(test_df)):
#     if test_df.loc[i, "SibSp"] >= 4:
#         family.append(2)
#     elif test_df.loc[i, "SibSp"] >= 2:
#         family.append(1)
#     else:
#         family.append(0)
#
# test_df["Family"] = family
# # print(test_df.isna().sum())
# test_result = rf_model.predict(test_df)
#
# submit = pd.DataFrame({"PassengerId": pid, "Survived": test_result})
# submit.to_csv("./dataset/titanic/submit.csv", index=False)

# 전처리 불필요한 column 제거
# ID, Name, tiket은 생존 여부에 영향을 주지 않아서 제거
train_df = pd.read_csv("./dataset/titanic/train.csv")
test_df = pd.read_csv("./dataset/titanic/test.csv")
pid = test_df["PassengerId"]    # 나중에 submit 만들기 위해서 미리 저장




# "Sex" 인코딩
from sklearn.preprocessing import LabelEncoder
train_df["Sex"] = LabelEncoder().fit_transform(train_df["Sex"])
test_df["Sex"] = LabelEncoder().fit_transform(test_df["Sex"])
# train_df["Sex"] = train_df["Sex"].map({"male":1, "female":0})
# test_df["Sex"] = test_df["Sex"].map({"male":1, "female":0})

# "Name" drop
# train_df["Name"] = train_df["Name"].map(lambda x: x.split(",")[1].split(".")[0].strip())
# title_replace = {
#     'Don':0,
#     'Rev':0,
#     'Capt':0,
#     'Jonkheer':0,
#     'Mr':1,
#     'Dr':2,
#     'Major':3,
#     'Col':3,
#     'Master':4,
#     'Miss':5,
#     'Mrs':6,
#     'Mme':7,
#     'Ms':7,
#     'Lady':7,
#     'Sir':7,
#     'Mlle':7,
#     'the Countess':7
# }
# test_df[test_df['Name'] == 'Dona']
# train_df["Name"] = train_df["Name"].apply(lambda x : title_replace(x))


# "Age".fillna().mean()
train_df.fillna(train_df["Age"].mean(),inplace=True)
test_df.fillna(test_df["Age"].mean(),inplace=True)
# print(train_df.isna().sum())


def age_category(age):
    out = 0
    if age <= 10:
        out = 0
    elif age <= 16:
        out = 1
    elif age <= 20:
        out = 2
    elif age <= 26:
        out = 3
    elif age <= 30:
        out = 4
    elif age <= 36:
        out = 5
    elif age <= 40:
        out = 6
    elif age <= 46:
        out = 7
    elif age <= 50:
        out = 8
    elif age <= 60:
        out = 9
    else:
        out = 10
    return out


train_df["AgeGroup"] = train_df["Age"].apply(lambda x: age_category(x))  # 해당 DataFrame에 맞춰 함수 적용
test_df["AgeGroup"] = test_df["Age"].apply(lambda x: age_category(x))  # 해당 DataFrame에 맞춰 함수 적용
age_point_replace = {
    0: 8,
    1: 6,
    2: 2,
    3: 4,
    4: 1,
    5: 7,
    6: 3,
    7: 2,
    8: 5,
    9: 4,
    10: 0
}

train_df['age_point'] = train_df['AgeGroup'].apply(lambda x: age_point_replace.get(x))
test_df['age_point'] = test_df['AgeGroup'].apply(lambda x: age_point_replace.get(x))

train_df["Embarked"] = train_df["Embarked"].map({"Q": 0, "C": 1, "S": 2})
test_df["Embarked"] = test_df["Embarked"].map({"Q": 0, "C": 1, "S": 2})
train_df['Embarked'] = train_df['Embarked'].fillna(2)
test_df['Embarked'] = test_df['Embarked'].fillna(2)


family = []
for i in range(len(train_df)):
    if train_df.loc[i, "SibSp"] >= 4:
        family.append(2)
    elif train_df.loc[i, "SibSp"] >= 2:
        family.append(1)
    else:
        family.append(0)
train_df["Family"] = family

family = []
for i in range(len(test_df)):
    if test_df.loc[i, "SibSp"] >= 4:
        family.append(2)
    elif test_df.loc[i, "SibSp"] >= 2:
        family.append(1)
    else:
        family.append(0)
test_df["Family"] = family

# Plass에 따라 점수 주기
class_point={
    1:2,
    2:1,
    3:0,
}
train_df["Pclass_point"] = train_df["Pclass"].apply(lambda x : class_point.get(x))
test_df["Pclass_point"] = test_df["Pclass"].apply(lambda x : class_point.get(x))


# train_df["Cabin"].fillna("U", inplace=True)
# test_df["Cabin"].fillna("U", inplace=True)
# cabin_point ={
#     "G" : 0,
#     "C" : 3,
#     "E" : 5,
#     "T" : 1,
#     "D" : 7,
#     "A" : 2,
#     "B" : 6,
#     "F" : 4
# }
# train_df["Cabin_point"] = train_df["Cabin"].apply(lambda x : cabin_point.get(x))
# test_df["Cabin_point"] = test_df["Cabin"].apply(lambda x : cabin_point.get(x))


# for dataset in train_df:
#     dataset.loc[ dataset['Fare']<=30, 'Fare'] = 0,
#     dataset.loc[(dataset['Fare']>30)&(dataset['Fare']<=80), 'Fare'] = 1,
#     dataset.loc[(dataset['Fare']>80)&(dataset['Fare']<=100), 'Fare'] = 2,
#     dataset.loc[(dataset['Fare']>100), 'Fare'] = 3

train_df.drop(['PassengerId','SibSp','Pclass','Parch','Ticket','Fare','Cabin','Age','Name'], axis=1, inplace=True)
test_df.drop(['PassengerId','SibSp','Pclass','Parch','Ticket','Fare','Cabin','Age','Name'], axis=1, inplace=True)
print(train_df.info())
# 모든 값을 정규화
from sklearn.preprocessing import StandardScaler
for dataset in train_df:
    dataset['Sex'] = StandardScaler().fit_transform(dataset['Sex'].values.reshape(-1, 1))
    dataset['Embarked'] = StandardScaler().fit_transform(dataset['Embarked'].values.reshape(-1, 1))
    dataset['AgeGroup'] = StandardScaler().fit_transform(dataset['AgeGroup'].values.reshape(-1, 1))
    dataset['age_point'] = StandardScaler().fit_transform(dataset['age_point'].values.reshape(-1, 1))
    dataset['Family'] = StandardScaler().fit_transform(dataset['Family'].values.reshape(-1, 1))
    dataset['Pclass_point'] = StandardScaler().fit_transform(dataset['Pclass_point'].values.reshape(-1, 1))

for dataset in test_df:
    dataset['Sex'] = StandardScaler().fit_transform(dataset['Sex'].values.reshape(-1, 1))
    dataset['Embarked'] = StandardScaler().fit_transform(dataset['Embarked'].values.reshape(-1, 1))
    dataset['AgeGroup'] = StandardScaler().fit_transform(dataset['AgeGroup'].values.reshape(-1, 1))
    dataset['age_point'] = StandardScaler().fit_transform(dataset['age_point'].values.reshape(-1, 1))
    dataset['Family'] = StandardScaler().fit_transform(dataset['Family'].values.reshape(-1, 1))
    dataset['Pclass_point'] = StandardScaler().fit_transform(dataset['Pclass_point'].values.reshape(-1, 1))




X = train_df.drop("Survived",axis=1)
y = train_df["Survived"]

def score(comp_df):
    perfect_df = pd.read_csv("./dataset/titanic/perfect.csv")
    hit = 0
    miss = 0
    for i in range(len(comp_df)):
        if comp_df["Survived"][i] == perfect_df["Survived"][i]:
            hit += 1
        else:
            miss += 1
    # print(hit, miss)
    return round(hit / (hit + miss), 2)

# Importing Classifier Modules# Import
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

model_svc = SVC()
model_svc.fit(X,y)
prediction = model_svc.predict(test_df)
submission = pd.DataFrame({
    "PassengerId": pid,
    "Survived": prediction
})
print("SVC", score(submission))


model_KN = KNeighborsClassifier(n_neighbors=3)
model_KN.fit(X,y)
prediction = model_KN.predict(test_df)
submission = pd.DataFrame({
    "PassengerId": pid,
    "Survived": prediction
})
print("KNeighborsClassifier" , score(submission))

model_DT = DecisionTreeClassifier()
model_DT.fit(X,y)
prediction = model_DT.predict(test_df)
submission = pd.DataFrame({
    "PassengerId": pid,
    "Survived": prediction
})
print("DecisionTreeClassifier", score(submission))


model_RF = RandomForestClassifier(max_depth=5, n_estimators=300)
model_RF.fit(X,y)
prediction = model_RF.predict(test_df)
submission = pd.DataFrame({
    "PassengerId": pid,
    "Survived": prediction
})
print("RandomForestClassifier", score(submission))

model_GN = GaussianNB()
model_GN.fit(X,y)
prediction = model_GN.predict(test_df)
submission = pd.DataFrame({
    "PassengerId": pid,
    "Survived": prediction
})
print("GaussianNB", score(submission))





