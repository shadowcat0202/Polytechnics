import pprint

import pandas as pd
import numpy as np


def auto_mpg_standardization():
    df = pd.read_csv("./dataset/auto-mpg.csv", header=None)
    df.columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin",
                  "name"]

    # df.set_option("display.max_column", None)

    # 데이터 표준화, 익숙한 단위로 사용하자
    # mpg ==> kpl(km per liter)
    # 1 mile: 1.690934km 1갤론: 3.78541 리터
    # mpg to kpl = 1.690934 / 3.78541 ==> about 0.425
    mpg_to_kpl = 1.690934 / 3.78541
    df["kpl"] = df["mpg"] * mpg_to_kpl
    print(df.head())

    # 자료형 변환 ==> 범주형 데이터 처리
    # 칼럼 속성 분석 차료형 변환(변경)
    df.info()  # horsepower의 "?" 변환
    df["horsepower"].replace("?", np.nan, inplace=True)
    df.dropna(subset=["horsepower"], axis=0, inplace=True)
    df["horsepower"] = df["horsepower"].astype("float")
    print(df["horsepower"].dtype)

    # 범주형 데이터 처리
    # 1) numpy histogram을 이용해서 binning을 하고,
    count, bin_divider = np.histogram(df["horsepower"], bins=3)
    print(count, bin_divider)
    # 2) pandas cut 함수를 이용해서 범주형 데이터로 변환
    bin_name = ["low", "norm", "high"]  # bin이름은 리스트로
    # horsepower를 (low, norm, high)로 변경
    df["horsepower_bin"] = pd.cut(x=df["horsepower"], bins=bin_divider, labels=bin_name, include_lowest=True)

    print(df[["horsepower", "horsepower_bin"]].head(15))

    # one-hot encoding
    hp_dummies = pd.get_dummies(df["horsepower_bin"])
    print(hp_dummies.head(15))

    # 정규화
    # 칼럼별 데이터의 번위가 차이가 많이 나면 학습에 영향을 미칠 수 있음
    # 각 칼럼의 데이터들을 해당 칼럼의 최댓값으로 나누느 방법
    # 정규화하기 전에는 필요에 따라 이상치 제거를 해주면 좋음
    # (col / col.max())
    test = [-6, -2, 2, 1, 3, 4]
    abs_list = list(map(abs, test))
    col_max = max(abs_list)
    test = [round(i / col_max, 4) for i in test]
    print(test)


def titanic_standardization():
    def add_10(n):
        return n + 10

    def add_two_obj(a, b):
        return a + b

    import seaborn as sns
    titanic = sns.load_dataset("titanic")
    df = titanic.loc[:, ["age", "fare"]]
    df["ten"] = 10  # ten 칼럼 추가 모든 행 원소에 10 대입
    print(df.head())

    # 시리즈의 원소에 함수를 매핑: 시리즈객체.apply(사용자함수명)
    sr1 = df["age"].apply(add_10)  # age칼럼의 모든 열에 add_10() 사용
    print(sr1)

    sr_add10 = df["age"] + 10
    print(sr_add10.head())

    # 시리즈 객체와 숫자를 이용해서 add_two_obj 사용자 함수 적용
    sr2 = df["age"].apply(add_two_obj, b=10)
    print(sr2)

    # 람다도 활용 가능
    sr3 = df["age"].apply(lambda x: add_10(x))  # x = df["age"]

    # Dataframe에서 함수를 매핑하기 위해서는: df.applymap()
    df_map = df.applymap(add_10)
    print(df_map.head())

    def missing_Value(series):
        return series.isnull()

    result = df.apply(missing_Value, axis=0)
    print(result.head(20))

    def min_max(x):
        return x.max() - x.min()

    result = df.apply(min_max)
    print(type(result))

    df1 = pd.DataFrame({"A": ['a0', 'a1', 'a2'],
                        "B": ['b0', 'b1', 'b2'],
                        "C": ['c0', 'c1', 'c2']},
                       index=[0, 1, 2])
    print(df1)
    df2 = pd.DataFrame({"A": ['0a', 'a1'],
                        "D": ['0d', 'd1'],
                        "E": ['0e', 'e1']},
                       index=[0, 1])
    print(df2)
    result_axis_0 = pd.concat([df1, df2], ignore_index=True, axis=0)
    result_axis_1 = pd.concat([df1, df2], ignore_index=True, axis=1)
    print(result_axis_0)
    print(result_axis_1)

    df1 = pd.read_excel("./dataset/stock price.xlsx")
    df2 = pd.read_excel("./dataset/stock price.xlsx")

    print(df1)
    print(df2)

    merge_inner = pd.merge(df1, df2)
    merge_outer = pd.merge(df1, df2, how='outer', on='id')
    pprint.pprint(merge_inner)
    pprint.pprint(merge_outer)

    merge_left = pd.merge(df1, df2, how='left', left_on='stock_name', right_on='name')
    print(merge_left)

titanic_standardization()
