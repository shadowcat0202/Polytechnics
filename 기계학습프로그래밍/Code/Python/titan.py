import pandas as pd
import numpy as np
import re  # 비정규식
import seaborn as sns

train_df = pd.read_csv("./dataset/titanic/train.csv")
test_df = pd.read_csv("./dataset/titanic/test.csv")

PassengerId = test_df["PassengerId"]
# print(train_df.head())

# print([(cab, type(cab)) for cab in train_df.Cabin.unique()])

original_train = train_df.copy()

full_data = [train_df, test_df]


def get_title(name):
    title_search = re.search(" ([A-Za-z]+)\.", name)
    if title_search:
        return title_search.group(1)
    return ""


Sex_mapping = {"female": 0, "male": 1}
title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
Embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

for dataset in full_data:
    # Cabin nan데이터 변경
    dataset["Has_Cabin"] = dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

    # 가족 인원수 정확하게 파악
    dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1  # 총 가족 인원수

    # 혼자 or 단체에 따른 생존율이 다르므로 단독 여부 체크
    dataset["IsAlone"] = 0  # 일단 빈거 만들고
    dataset.loc[dataset["FamilySize"] == 1, "IsAlone"] = 1

    # Embarked에서 nan을 가장 많이 존재하는 data로 채우기
    dataset["Embarked"] = dataset["Embarked"].fillna(
        str(dataset["Embarked"].value_counts().to_frame().iloc[0].name)
    )

    age_avg = dataset["Age"].mean()  # 평균
    age_std = dataset["Age"].std()  # 표준편차
    age_null_count = dataset["Age"].isna().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

    # Passenger Name으로 새로운 칼럼 생성
    dataset["Title"] = dataset["Name"].apply(get_title)
    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # Mapping Sex
    dataset["Sex"] = dataset["Sex"].map(Sex_mapping).astype(int)

    # Mapping Title
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    # Mapping Embarked
    dataset["Embarked"] = dataset["Embarked"].map(Embarked_mapping).astype(int)
    # Fare은 중앙값으로 대체
    dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
    # Mapping Fare
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    # Mapping Age
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

train_df = train_df.drop(drop_elements, axis=1)
test_df = test_df.drop(drop_elements, axis=1)

from sklearn.model_selection import KFold

cv = KFold(n_splits=10)
accuracies = list()
max_attrivutes = len(list(test_df))
depth_range = range(1, max_attrivutes + 1)  # 1~속성개수 + 1 만큼 왜?

from sklearn.tree import DecisionTreeClassifier

for depth in depth_range:
    # print("depth:", depth, "==========================================")
    fold_accuracy = []
    tree_clf = DecisionTreeClassifier(max_depth=depth)
    # print("Current max depth: ", depth, "\n")
    for train_fold, valid_fold in cv.split(train_df):  # row index가 반환된다
        f_train = train_df.loc[train_fold]  # Extract train data with cv indices
        f_valid = train_df.loc[valid_fold]  # Extract valid data with cv indices

        model = tree_clf.fit(X=f_train.drop(['Survived'], axis=1),
                             y=f_train["Survived"])  # We fit the model with the fold train data
        valid_acc = model.score(X=f_valid.drop(['Survived'], axis=1),
                                y=f_valid["Survived"])  # We calculate accuracy with the fold validation data
        fold_accuracy.append(valid_acc)
        # print(valid_acc)

    avg = sum(fold_accuracy) / len(fold_accuracy)
    accuracies.append(avg)
    # print("Accuracy per fold: ", fold_accuracy, "\n")
    # print("Average accuracy: ", avg)
    # print("\n")

# Just to show results conveniently
df = pd.DataFrame({"Depth": depth_range, "Average_Accuracy": accuracies})
df = df[["Depth", "Average_Accuracy"]]
print(df.to_string(index=False))

y_train = train_df["Survived"]
x_train = train_df.drop(["Survived"], axis=1).values
x_test = test_df.values

# 위에서 결정트리 depth = 3일때 정확도가 가장 높게 나왔으므로 3으로 설정하고 학습 시킨다
tree_clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
tree_clf.fit(x_train, y_train)

y_pred = tree_clf.predict(x_test)

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
    return round(hit / (hit + miss), 4)

submission = pd.DataFrame({
    "PassengerId": PassengerId,
    "Survived": y_pred
})
submission.to_csv("out.csv",index=False)
print("logistic", score(submission))

