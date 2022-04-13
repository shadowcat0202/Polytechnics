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


# 나중에 submit처리하기 위해서 PassengerId columne을 미리 저장해둔다
pid = test_df["PassengerId"]
train_df = train_df.drop("PassengerId", axis=1)
test_df = test_df.drop("PassengerId", axis=1)

seed = 3
numpy.random.seed(0)
tf.random.set_seed(seed)

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
# print(train_df[['Cabin_c', 'Survived']].groupby(['Cabin_c'], as_index=False).mean())

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
                                                'Rare')  # 나머지는 Rare로 표기
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
combine = [train_df, test_df]
guess_ages = np.zeros((2, 3))
for dataset in combine:
    # train_df["Age"].fillna(train_df["Age"].mean(), inplace=True)
    for s in range(0, 2):  # Sex
        for p in range(0, 3):  # Pclass
            guess_df = dataset[(dataset["Sex"] == s) & (dataset["Pclass"] == p + 1)]["Age"].dropna()
            age_guess = guess_df.median()
            # age의 random값의 소수점을 .5에 가깝도록 변형
            guess_ages[s, p] = int(age_guess / 0.5 + 0.5) * 0.5
    for s in range(0, 2):  # Sex
        for p in range(0, 3):  # Pclass
            dataset.loc[(pd.isnull(dataset["Age"])) & (dataset.Sex == s) & (dataset.Pclass == p + 1), 'Age'] \
                = guess_ages[s, p]
    dataset['Age'] = dataset['Age'].astype(int)

# print(test_df["Age"].isna().sum())
# print(train_df["Age"].isna().sum())

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
# 임의로 5개 그룹을 지정
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
combine = [train_df, test_df]
for dataset in combine:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]

# SibSp + Parch를 통해서 정확한 가족 인원수 파악===============================================================


# from sklearn.preprocessing import LabelEncoder
# e = LabelEncoder()
# e.fit(y)
# y = e.transform(y)
#
# ## Functional
# from tensorflow.keras.layers import Input, Dense
# from tensorflow.keras.models import Model
# # 각 층 셋팅
# # 입력층 (shape=(열,행)) 행:데이터는 계속 추가할수 있으니까 필요한건 column vector가 몇개 있는지가 중요하다
# inputs = Input(shape=(len(X[0]),))
# """
# from tensorflow import TensorShape
# inputs = Input()
# inputs.shape = TensorShape([None, 60])
# """
# hidden = Dense(24, activation='relu')(inputs)
# hidden = Dense(10, activation='relu')(inputs)
# output = Dense(1, activation='sigmoid')(hidden)
# f_logistic_model = Model(inputs=inputs, outputs=output)
#
# # 모델 컴파일
# f_logistic_model.compile(loss='binary_crossentropy',
#                          optimizer='adam',
#                          metrics=['accuracy'])
# # f_logistic_model.optimizer.lr = 0.001
#
# # 학습
# with tf.device("/device:GPU:0"):
#     f_logistic_model.fit(x=X, y=y, epochs=130, batch_size=5)
#     # verbose=False 하면 막대바 안보임
# # 모델 저장
# from keras.models import load_model
# f_logistic_model.save('my_model.h5')
# del f_logistic_model    # 테스트를 위해 메모리 내의 모델을 삭제
# f_logistic_model = load_model('my_model.h5')
#
# # model.evaluate() = [오차율?, 예측율]
# score = f_logistic_model.evaluate(X_test, y_test, verbose=1)
# print(f"정답률={score[1]*100} loss={score[0]}")
