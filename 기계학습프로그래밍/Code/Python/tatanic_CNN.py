import numpy
import pandas as pd
import tensorflow as tf
import numpy as np
# import seaborn as sns

# https://computer-science-student.tistory.com/113
from matplotlib import pyplot as plt


train_df = pd.read_csv("./dataset/titanic/train.csv")
test_df = pd.read_csv("./dataset/titanic/test.csv")
# print(train_df.info())  # 데이터 보고 싶다


# 전처리 과정
# Cabin 데이터 확인===============================================================================================
# 귀찮으니까 G로 채운다
# 방법 1. for 문
# for i in range(train_df.shape[1]):
#     train_df.loc[i, "Cabin"] = str(train_df.loc[i, "Cabin"]).split("")[1]
# for i in range(train_df.shape[1]):
#     test_df.loc[i, "Cabin"] = str(test_df.loc[i, "Cabin"]).split("")[1]

# 방법 2. column를 하나 만들어서 기존 column을 drop시켜도 가능
train_df["Cabin_c"] = train_df["Cabin"].str.split("").str[1]
test_df["Cabin_c"] = test_df["Cabin"].str.split("").str[1]
train_df["Cabin_c"].fillna("D", inplace=True)
test_df["Cabin_c"].fillna("D", inplace=True)
cabin_c_mapping_v2 = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3,
    'E': 4, 'F': 5, 'G': 6, 'T': 7
}
combine = [train_df, test_df]
for dataset in combine:
    dataset['Cabin_c'] = dataset['Cabin_c'].map(cabin_c_mapping_v2)
    dataset['Cabin_c'] = dataset['Cabin_c'].fillna(0)
# print(train_df[['Cabin_c', 'Survived']].groupby(['Cabin_c'], as_index=False).mean().sort_values(by="Survived", ascending=False))
# print(test_df["Cabin_c"].unique())
# Ticket, Cabin 제거===============================================================================================
train_df = train_df.drop(["Ticket", "Cabin"], axis=1)
test_df = test_df.drop(["Ticket", "Cabin"], axis=1)

# Name -> Title===============================================================================================
combine = [train_df, test_df]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
# print(pd.crosstab(train_df['Title'], train_df['Sex']))

# 상대적으로 female에서는 Miss, Mrs, male에서는 Mater, Mr가 두드러지게 나옴
combine = [train_df, test_df]
for dataset in combine:
    dataset["Title"] = dataset["Title"].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
                                                'Other')  # 나머지는 Rare로 표기
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')  # 불어식 표현이라서 어색할 수 있다
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
combine = [train_df, test_df]
title_mapping = {"Mr": 1, "Rare": 2, "Mater": 3, "Miss": 4, "Mrs": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df = train_df.astype({"Title": "int"})  # 형변환 노가다 필요(일반적인 방법으로는 이상하게 불가능)
test_df = test_df.astype({"Title": "int"})

# train_df['Name'] = train_df["Name"].map(lambda x: x.split(',')[1].split('.')[0].strip())
# titles = train_df['Name'].unique()
# test_df['Name'] = test_df['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
# test_titles = test_df['Name'].unique()


# Sex===============================================================================================
combine = [train_df, test_df]
# 원 핫 인코딩? ㅋㅋ 이상황에 쓰는 말이 맞는지는 모르겠지만 문자열을 숫자로 바꿔주는게 인코딩이니ㅋ
# 방법 1.
# from sklearn.preprocessing import LabelEncoder
# train_df["Sex"] = LabelEncoder().fit_transform(train_df["Sex"])
# test_df["Sex"] = LabelEncoder().fit_transform(test_df["Sex"])
# 방법 2.
for dataset in combine:
    dataset["Sex"] = dataset["Sex"].map({"female": 1, "male": 0}).astype(int)
# print(df_train.iloc[0]["Sex"])    # male = 1, female =0


# print(train_df.isna().sum())

# Age===============================================================================================
# 시각화
# grid = sns.FacetGrid(train_df, row="Pclass", col="Sex", height=2.2, aspect=1.6)
# grid.map(plt.hist, "Age", alpha=5, bins=20)
# grid.add_legend()
# v1===================================================================
# combine = [train_df, test_df]
# guess_ages = np.zeros((2, 3))
# for dataset in combine:
#     # train_df["Age"].fillna(train_df["Age"].mean(), inplace=True)
#     for s in range(0, 2):  # Sex
#         for p in range(0, 3):  # Pclass
#             guess_df = dataset[(dataset["Sex"] == s) & (dataset["Pclass"] == p + 1)]["Age"].dropna()
#             age_guess = guess_df.median()
#             # age의 random값의 소수점을 .5에 가깝도록 변형
#             guess_ages[s, p] = int(age_guess / 0.5 + 0.5) * 0.5
#     for s in range(0, 2):  # Sex
#         for p in range(0, 3):  # Pclass
#             dataset.loc[(pd.isnull(dataset["Age"])) & (dataset.Sex == s) & (dataset.Pclass == p + 1), 'Age'] \
#                 = guess_ages[s, p]
#     dataset['Age'] = dataset['Age'].astype(int)

# v2===================================================================
combine = [train_df, test_df]
for dataset in combine:
    dataset["Age"].fillna(dataset.groupby("Title")["Age"].transform("median"), inplace=True)

# print(test_df["Age"].isna().sum())
# print(train_df["Age"].isna().sum())


# 임의로 5개 그룹을 지정
# pd.cut() = 동일한 구간으로 나누기
# pd.qcut()= 동일한 개수로 나누기

# 수치 확인
# print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))

#v1 ==================================================================================
combine = [train_df, test_df]
for dataset in combine:
    dataset["Age"].fillna(dataset.groupby("Title")["Age"].transform("median"), inplace=True)
    dataset["Age_clean"] = 0
    dataset.loc[dataset['Age'] <= 10, 'Age_clean'] = 0
    dataset.loc[(dataset['Age'] > 10) & (dataset['Age'] <= 16), 'Age_clean'] = 1
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 20), 'Age_clean'] = 2
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 26), 'Age_clean'] = 3
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 30), 'Age_clean'] = 4
    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 36), 'Age_clean'] = 5
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 40), 'Age_clean'] = 6
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 46), 'Age_clean'] = 7
    dataset.loc[(dataset['Age'] > 46) & (dataset['Age'] <= 50), 'Age_clean'] = 8
    dataset.loc[(dataset['Age'] > 50) & (dataset['Age'] <= 60), 'Age_clean'] = 9
    dataset.loc[dataset['Age'] > 60, 'Age_clean'] = 10

train_df = train_df.drop(['Age'], axis=1)
test_df = test_df.drop(['Age'], axis=1)
combine = [train_df, test_df]
#v2 ==================================================================================
# train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
# combine = [train_df, test_df]
# for dataset in combine:
#     dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
#     dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
#     dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
#     dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
#     dataset.loc[dataset['Age'] > 64, 'Age'] = 4
# train_df = train_df.drop(['AgeBand'], axis=1)
# combine = [train_df, test_df]

# SibSp + Parch를 통해서 정확한 가족 인원수 파악===============================================================
combine = [train_df, test_df]
for dataset in combine:
    dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1
# print(train_df[["FamilySize", "Survived"]].groupby(["FamilySize"], as_index=False).mean().sort_values(by="Survived", ascending=False))

combine = [train_df, test_df]
for dataset in combine:
    dataset["Solo"] = 0
    dataset.loc[dataset["FamilySize"] == 1, "Solo"] = 1
# 혼자 왔다면 생존 확률이 낮다는 것을 알 수 있음
# print(train_df[["IsAlone","Survived"]].groupby(["IsAlone"], as_index=False).mean())

# 단독인 사람과 아닌사람의 생존 확률을 확인 했으니 사요했던 columns는 드랍
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]
for dataset in combine:
    dataset["Embarked"] = dataset["Embarked"].fillna("S")
    dataset["Embarked"] = dataset["Embarked"].map({"S": 2, "C": 1, "Q": 0})

# Fare====================================================================================
# Pclass별 평균값으로 변경
# print(dataset[["Pclass", "Fare"]].groupby(["Pclass"], as_index=False).mean().sort_values(by="Fare", ascending=False))
combine = [train_df, test_df]
for dataset in combine:
    for p in range(0, 3):
        pclass_mean = dataset.loc[dataset["Pclass"] == p + 1, "Fare"].mean()
        dataset.loc[dataset.Fare.isna() & dataset.Pclass == p + 1, "Fare"] = pclass_mean

# train_df['FareBand'] = pd.qcut(train_df['Fare'], 4) # 수치 확인용
# train_df = train_df.drop(['FareBand'], axis=1)

for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

combine = [train_df, test_df]



# 데이터 준비 ====================================================================
# 나중에 submit처리하기 위해서 PassengerId columne을 미리 저장해둔다
X_train = train_df.drop(["PassengerId","Name","Survived"], axis=1)
Y_train = train_df["Survived"]

pid = test_df["PassengerId"]
X_test = test_df.drop(["PassengerId","Name"],axis=1).copy()
print(X_train.shape, Y_train.shape, X_test.shape)
# print(X_train.head())
# print(X_test.head())


def score(comp_df):
    perfect_df = pd.read_csv("./dataset/titanic/groundtruth.csv")
    hit = 0
    miss = 0
    for i in range(len(comp_df)):
        if comp_df["Survived"][i] == perfect_df["Survived"][i]:
            hit += 1
        else:
            miss += 1
    # print(hit, miss)
    return round(hit / (hit + miss), 5)

# Importing Classifier Modules# Import
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score




model_logistic = LogisticRegression()
model_logistic.fit(X_train, Y_train)
prediction = model_logistic.predict(X_test)
submission = pd.DataFrame({
    "PassengerId": pid,
    "Survived": prediction
})
print("logistic", score(submission))

model_svc = SVC()
model_svc.fit(X_train, Y_train)
prediction = model_svc.predict(X_test)
submission = pd.DataFrame({
    "PassengerId": pid,
    "Survived": prediction
})
print("SVC", score(submission))


model_KN = KNeighborsClassifier(n_neighbors=3)
model_KN.fit(X_train, Y_train)
prediction = model_KN.predict(X_test)
submission = pd.DataFrame({
    "PassengerId": pid,
    "Survived": prediction
})
print("KNeighborsClassifier" , score(submission))

model_DT = DecisionTreeClassifier()
model_DT.fit(X_train, Y_train)
prediction = model_DT.predict(X_test)
submission = pd.DataFrame({
    "PassengerId": pid,
    "Survived": prediction
})
print("DecisionTreeClassifier", score(submission))

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
model_RF = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=0)
model_RF.fit(X_train, Y_train)
prediction = model_RF.predict(X_test)
submission = pd.DataFrame({
    "PassengerId": pid,
    "Survived": prediction
})
submission.to_csv("./submit.csv", index=False)
print("RandomForestClassifier", score(submission))

model_GN = GaussianNB()
model_GN.fit(X_train, Y_train)
prediction = model_GN.predict(X_test)
submission = pd.DataFrame({
    "PassengerId": pid,
    "Survived": prediction
})
print("GaussianNB", score(submission))

seed = 3
numpy.random.seed(0)
tf.random.set_seed(seed)

## Functional
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
# # 각 층 셋팅
# # 입력층 (shape=(열,행)) 행:데이터는 계속 추가할수 있으니까 필요한건 column vector가 몇개 있는지가 중요하다

inputs = Input(shape=(X_train.shape[1],))
# """
# from tensorflow import TensorShape
# inputs = Input()
# inputs.shape = TensorShape([None, 60])
# """
hidden = Dense(24, activation='relu')(inputs)
hidden = Dense(18, activation='relu')(hidden)
hidden = Dense(10, activation='relu')(hidden)
output = Dense(1, activation='sigmoid')(hidden)
# output = Dense(1, activation='relu')(hidden)    # 무조건 0보다 크면 생존이라고 하는걸로 그 이상은 동일하다
f_logistic_model = Model(inputs=inputs, outputs=output)
#
# # 모델 컴파일
f_logistic_model.compile(loss='binary_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])
# f_logistic_model.optimizer.lr = 0.001

# 학습jf
with tf.device("/device:GPU:0"):
    f_logistic_model.fit(x=X_train, y=Y_train, epochs=50, batch_size=10, verbose=False)
    # verbose=False 하면 막대바 안보임
# 모델 저장
from keras.models import load_model
f_logistic_model.save('my_model.h5')
del f_logistic_model    # 테스트를 위해 메모리 내의 모델을 삭제
f_logistic_model = load_model('my_model.h5')

standard = 0.5
cnn_prediction = f_logistic_model.predict(X_test)
prediction = pd.DataFrame(cnn_prediction, columns=["Survived"])
prediction.loc[prediction['Survived'] >= standard, 'Survived'] = int(1)
prediction.loc[prediction['Survived'] < standard, 'Survived'] = int(0)
prediction = prediction.values.tolist()
submission = pd.DataFrame({
    "PassengerId": pid,
    "Survived": prediction
})
# submission["Survived"] = submission["Survived"].astype(int)
# print(submission.info())
print("CNN",round(standard,1), score(submission))

# for standard in np.arange(0,1,0.1):
#     cnn_prediction = f_logistic_model.predict(X_test)
#     prediction = pd.DataFrame(cnn_prediction, columns=["Survived"])
#     prediction.loc[prediction['Survived'] >= standard, 'Survived'] = int(1)
#     prediction.loc[prediction['Survived'] < standard, 'Survived'] = int(0)
#     prediction = prediction.values.tolist()
#     submission = pd.DataFrame({
#         "PassengerId": pid,
#         "Survived": prediction
#     })
#     # submission["Survived"] = submission["Survived"].astype(int)
#     # print(submission.info())
#     print("CNN",round(standard,1), score(submission))