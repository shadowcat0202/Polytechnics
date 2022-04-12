import numpy
import pandas as pd
import tensorflow as tf

# https://computer-science-student.tistory.com/113

train_df = pd.read_csv("./dataset/titanic/train.csv")
test_df = pd.read_csv("./dataset/titanic/test.csv")
# print(train_df.info())  # 데이터 보고 싶다
# print(test_df.head())

seed = 3
numpy.random.seed(0)
tf.random.set_seed(seed)

# Ticket, Cabin 제거
train_df = train_df.drop(["Ticket", "Cabin"], axis=1)
test_df = test_df.drop(["Ticket", "Cabin"], axis=1)

# Name
comb = [train_df, test_df]
for dataset in comb:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(train_df['Title'], train_df['Sex'])

for dataset in comb:
    dataset["Title"] = dataset["Title"].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
title_mapping = {"Mr": 1, "Rare": 2, "Mater": 3, "Miss": 4, "Mrs": 5}

for dataset in comb:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df = train_df.astype({"Title":"int"}) # 형변환 노가다 필요
test_df = test_df.astype({"Title":"int"})


print(train_df.head())

# Sex
from sklearn.preprocessing import LabelEncoder

train_df["Sex"] = LabelEncoder().fit_transform(train_df["Sex"])
test_df["Sex"] = LabelEncoder().fit_transform(test_df["Sex"])
# print(df_train.iloc[0]["Sex"])    # male = 1, female =0

train_df['Name'] = train_df["Name"].map(lambda x: x.split(',')[1].split('.')[0].strip())
titles = train_df['Name'].unique()
test_df['Name'] = test_df['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
test_titles = test_df['Name'].unique()









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
