import pandas as pd
import numpy as np
import seaborn as sns
NEW_LINE = "======================================================================\n"
titanic = sns.load_dataset("titanic")

# 모든 행과 5개의 열을 가지고 와서 dataframe 생성

df = titanic.loc[:, ["age", "sex", "class", "fare", "survived"]]

print(f"승객수:{len(df)}")
print(df.head())
print(NEW_LINE)
# Groupby 실습
grouped = df.groupby(["class"])
print(grouped.head())
print(grouped.sum())
print(NEW_LINE)
# 그룹 객체를 iteration을 돌면서 출력

for key, group in grouped:
    print(f"* key : {key}")
    print(f"* number : {len(group)}")
    print(grouped.head())
    print("--------------------------")
print(NEW_LINE)

# 연산 메서드 사용
# 연산 메서드 사용시에는 연산이 가능한 열에 대해서만 선택적으로 연산 수행
# 문자열을 포함한 sex, class 열을 제외하고
# 숫자형 데이터에 대해서만 평균을 구해보자
average = grouped.mean()
print(average)
print(NEW_LINE)
# 결과 1등석의 경우, 평균 나이가 가장 많고, 구조 확률도 62%로 가장 높음

# 개별 그룹 선택 (first or second class 선택 등
group3 = grouped.get_group("Third")
print(group3)
print(NEW_LINE)

grouped_tow = df.groupby(["class", "sex"])
for key, group in grouped_tow:
    print(f"* key : {key}")
    print(f"* number : {len(group)}")
    print(group.head())
    print("--------------------------")
print(NEW_LINE)

average = grouped_tow.mean()
print(average)
print(NEW_LINE)

# 멀티인덱스 형태로 되어있는 그룹에서 개별 그룹을 가지고 오고 싶을때
# 튜플형태로 칼럼을 지정하면 됨 (First group, Second group, Third group ...)
# (3등석에 여성의 데이터를 가지고 오고 싶을때): ("Third", "female")

grouped3f = grouped_tow.get_group(("Third", "female"))
print(grouped3f.head())
print(NEW_LINE)

# Filtering
# 예: 나이 평균이 30보다 작은 그룹만을 필터링해서 DF로 반환
average = grouped.mean()
print(average)
# 평균 나이가 30미만인 그룹(클래스)는 second, third 그룹
age_filter = grouped.filter(lambda x: x.age.mean() < 30)
print(age_filter)
