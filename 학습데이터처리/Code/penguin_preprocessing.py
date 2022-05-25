import pandas as pd
import seaborn as sns

def titanic():
    pd.set_option("display.max_columns", None)
    df_titanic = sns.load_dataset("titanic")
    print(df_titanic.head())
    print(df_titanic.info())
    # deck 칼럼에 NaN이 많이 있다는 것을 알 수 있다
    print(df_titanic["deck"].value_counts(dropna=False))

    print(df_titanic["deck"].isnull().all)
    print(df_titanic["deck"].isnull().count())

    print("=======================================================")
    # sum() 대신 count도 가능
    # sum(axis=), isnull ==> column 기준으로 계산하는 메서드
    # axis=0: column, axis=1: row를 의미
    # 해당모듈, 특정한 모듈이 칼럼단위로 계산하는 모듈이면 axis=0: col
    print(df_titanic.isnull().sum())
    print(df_titanic.isnull().sum(axis=1))
    # count: 열의 개수를 카운트 옵션으로 dropna와 같은 내용 있음
    # count(): 자체는 NaN를 배제하고 카운팅
    # isnull.sum() <== 모른다고 하더라도 시리즈 연산을 하면 각 열의 NaN개수 확인 가능
    print(len(df_titanic) - df_titanic.count())
    df_drop = df_titanic.drop(columns=["deck"], axis=1)
    # NaN이 특정 개수 이상인 열을 날리고 싶을때
    df_thresh = df_titanic.dropna(axis=1, thresh=500)
    print(df_thresh)

def penguins():
    df_pg = sns.load_dataset("penguins")
    # print(df_pg)
    # print(df_pg.isnull().sum())
    print(df_pg[df_pg.isnull().any(axis=1)])
    # 3, 339번 인덱스와 성별 칼럼은 drop
    df_drop_condition = df_pg[df_pg["bill_length_mm"].isnull()].index
    print(df_drop_condition)
    df_drop = df_pg.drop(df_drop_condition)
    print(df_drop)


penguins()


