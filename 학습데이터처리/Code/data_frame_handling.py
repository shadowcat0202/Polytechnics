import pandas as pd


def test1():
    data = {
        'c0': [1, 2, 3],
        'c1': [4, 5, 6],
        'c2': [7, 8, 9],
        'c3': [10, 11, 12],
        'c4': [13, 14, 15]
    }
    data_df = pd.DataFrame(data)
    # pandas.DataFrame(2-dim-array, index=행 인덱스 배열, columns=열 이름 배열
    df = pd.DataFrame([{23, '남', '대구'}, [24, '여', '부산']],
                      index=["row이름1", "row이름2"],
                      columns=['나이', '성별', '지역'])
    print(df)

    # DataFrame객체.columns = 새로운 열 이름 배열
    df.index = ['학생1', '학생2']
    df.columns = ['연령', '남녀', '주소지']
    print(df)

    # DataFrame.rename(index={기존인덱스:새인덱스, ...},columns={기존이름:새이름, ...})
    df = df.rename(
        index={'학생1': 'newrowname1', '학생2': 'newrowname2'},
        columns={'연령': 'col1', '남녀': 'col2', '주소지': 'col3'}
    )
    print(df)


def test2():
    df = pd.read_csv("./Data_set/read_my_sample.csv")
    df_copy = df.set_index("이름")
    print("====================원본====================")
    print(df)
    print()

    print("====================기학프 열 drop()================")
    df_copy = df_copy.drop('기학프', axis=1)
    print(df_copy)
    print()

    print("====================1번 인덱스 행 drop()====================")
    df_copy = df_copy.drop('홍길동', axis=0)
    print(df_copy)
    print()

    print("====================1개 row선택====================")
    df_copy = df.set_index("이름")
    print(df_copy.loc['이순신'])
    print()

    print("====================1개 col선택 = Series====================")
    df_copy = df.set_index("이름")
    print(type(df_copy['AI기초']))
    print(df_copy['AI기초'])  # 단일 columns
    print(df_copy[['자바', '자료']])  # n개 columns

    print("====================특정 원소 선택 = 타입은 그때그때 달라요====================")
    # DataFrame.loc[row, col]
    print(df_copy.loc['이순신', "자바"])
    print(df_copy.loc['정약용', "자료":"자바"])

    print("====================set_index()====================")
    # 특정 열을 행 인덱스로 설정
    # DataFrame.set_index(['열 이름'] | '열 이름')
    print()

    print("====================데이터 추가====================")
    # 새로운 행 추가
    # DataFrame[새로운 열 이름] = [데이터,...]
    df_copy = df
    df_copy["newcol"] = [50, 84, 63]
    print(df_copy)
    # 새로운 열 추가
    df_copy.loc["newrow"] = ["세종", 22, 54, 6, 87, 52]
    print(df_copy)

    print("====================리인덱스====================")
    df_copy = df.set_index("이름")
    new_idx = ['홍길동', '이순신', '정약용', 'new1', 'new2']
    df_copy1 = df_copy.reindex(new_idx, fill_value=0)
    print(df_copy1)
    df_copy1 = df_copy1.sort_index(ascending=False)
    print(df_copy1)
    df_copy1 = df_copy1.sort_values(by='newcol', ascending=False)
    print(df_copy1)

    print("====================리셋인덱스====================")
    df_copy = df.set_index("이름")
    print(df_copy)
    df_copy = df_copy.reset_index()
    print(df_copy)

    print("===================원소 변경=======================")
    df_copy = df.set_index("이름")
    print(df_copy.iloc[0][0])
    df_copy.iloc[0, [1, 2, 3]] = [10, 10, 10]
    print(df_copy)
    df_copy.iloc[[0, 1], [1, 2, 3]] = [[100, 100, 100], [90, 90, 90]]
    print(df_copy)

    df_copy.loc[["세종", '정약용'], ["자료", "자바", "기학프"]] = [[11, 22, 33], [44, 55, 66]]
    print(df_copy)


def test3():
    df = pd.read_csv("./Data_set/read_my_sample.csv")
    df = df.set_index(["이름"])
    print(df)

    # df에서 2개 이상의 행과 열로부터 데이터를 가지고 와보자
    # 홍길동, 이순신 자료, 자바 점수
    print(df.loc[['홍길동', '이순신'], ['자바', '자료']])
    # 홍길동, 정약용의 자료, 기학프 점수
    print(df.loc[['홍길동', '정약용'], ['자료', '기학프']])


def test4():
    import pandas as pd
    student1 = pd.Series({'국어': 100, '영어': 80, '수학': 90})
    print(student1)

    percentage = student1 / 100
    print(percentage)
    print(type(percentage))
    student2 = pd.Series({'영어': 50, '수학': 90, '국어': 100})

    #산술 연산 가능
    result = pd.DataFrame([student1 + student2,
                           student1 - student2,
                           student1 * student2,
                           student1 / student2],
                          index=['add','sub','mul','div'])
    print(result, "\n")

    #NaN을 만들어보고, fill_value라는 옵션 사용
    student1 = pd.Series({'국어': 100, '영어': 80, '수학': 90})
    student2 = pd.Series({'수학': 90, '국어': 100})
    print(student1.add(student2, fill_value=0))
    result = pd.DataFrame([student1.add(student2,fill_value=0),
                           student1.sub(student2,fill_value=0),
                           student1.mul(student2,fill_value=0),
                           student1.div(student2,fill_value=0)])
    print(result)


def test5():
    # df = pd.read_json("./Data_set/read_json_sample.json")
    # print(df)
    # file_path = "./Data_set/read_csv_sample.csv"
    #
    # df1 = pd.read_csv(file_path)
    # print(df1)
    # print()
    #
    # df2 = pd.read_csv(file_path, header=None)
    # print(df2)
    # print()
    #
    # df3 = pd.read_csv(file_path, index_col=None)
    # print(df3)
    # print()

    df_html = pd.read_html("./Data_set/sample.html")
    for df in df_html:
        df.to_csv("./html_to.csv")
        print(df)