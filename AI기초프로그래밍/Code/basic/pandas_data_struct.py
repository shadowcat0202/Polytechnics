<<<<<<< HEAD
def Series_test():
    import pandas as pd
    # [1]=========================pandas.Series(딕셔너리)====================================
    print("[1-1]k:v 구조의 딕셔너리를 만들 경우 k = 인덱스 v = value로 매칭되어 반환")
    dict_data = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}
    sr = pd.Series(dict_data)  # k:v 구조의 딕셔너리를 만들 경우 k = 인덱스 v = value로 매칭되어 반환
    print(type(sr), "\n")
    print(sr, "\n")

    print("[1-2]리스트를 시리즈(k:v 구조 x)로 변환해서 저장할 경우 인덱스는 자동으로 정수형 위치 인덱스로 반환")
    list_data = ['2020-01-02', 3.14, 'ABC', 100, True]
    ld = pd.Series(list_data)  # 리스트를 시리즈(k:v 구조 x)로 변환해서 저장할 경우 인덱스는 자동으로 정수형 위치 인덱스로 반환
    print(ld, "\n")
    print(ld[1], "\n")  # 정수형 인덱스로 참조하기

    idx = sr.index
    val = sr.values
    print(idx, "\n")
    print(val, "\n")

    # 튜플을 시리즈로 반환(index옵션에 인덱스 이름을 지정)
    print("[2-1]튜플을 시리즈로 반환(index옵션에 인덱스 이름을 지정)")
    tup_data = ['전세환', '19xx-xx-xx', '남', True]
    sr = pd.Series(tup_data, index=['name', 'jumin', 'sex', 'student'])
    print(sr[1], "\n")
    print(sr['jumin'], "\n")



    # 원소 참조 방법
    # 1.인덱스 번호 or 인덱스 이름으로 참조 가능
    print("[2-2]인덱스 번호 or 인덱스 이름으로 참조 가능")
    print(sr[0])
    # print(sr['name'])  #둘중 하나

    # 2.여러개의 원소 선택(인덱스 리스트 활용)
    print("[2-3]여러개의 원소 선택(인덱스 리스트 활용)")
    print(sr[[1, 2]], "\n")
    # print(sr[["jumin", "sex"]])    #둘중 하나

    # 3.여러개의 원소 선택(슬라이싱)
    print("[2-3]여러개의 원소 선택(슬라이싱)")
    print(sr[1:2], "\n")
    # print(sr["jumin", 'sex'])  #둘중 하나
    del pd


def DataFrame_test():
    import pandas as pd
    # 딕셔너리 -> DataFrame 변환
    # pandas.DataFrame(딕셔너리 객체)
    dict_data = {'c0': [1, 2, 3],
                 'c1': [4, 5, 6],
                 'c2': [7, 8, 9],
                 'c3': [10, 11, 12],
                 'c4': [13, 14, 15]}
    df = pd.DataFrame(dict_data)

    print(type(df))
=======
def Series_test():
    import pandas as pd
    # [1]=========================pandas.Series(딕셔너리)====================================
    print("[1-1]k:v 구조의 딕셔너리를 만들 경우 k = 인덱스 v = value로 매칭되어 반환")
    dict_data = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}
    sr = pd.Series(dict_data)  # k:v 구조의 딕셔너리를 만들 경우 k = 인덱스 v = value로 매칭되어 반환
    print(type(sr), "\n")
    print(sr, "\n")

    print("[1-2]리스트를 시리즈(k:v 구조 x)로 변환해서 저장할 경우 인덱스는 자동으로 정수형 위치 인덱스로 반환")
    list_data = ['2020-01-02', 3.14, 'ABC', 100, True]
    ld = pd.Series(list_data)  # 리스트를 시리즈(k:v 구조 x)로 변환해서 저장할 경우 인덱스는 자동으로 정수형 위치 인덱스로 반환
    print(ld, "\n")
    print(ld[1], "\n")  # 정수형 인덱스로 참조하기

    idx = sr.index
    val = sr.values
    print(idx, "\n")
    print(val, "\n")

    # 튜플을 시리즈로 반환(index옵션에 인덱스 이름을 지정)
    print("[2-1]튜플을 시리즈로 반환(index옵션에 인덱스 이름을 지정)")
    tup_data = ['전세환', '19xx-xx-xx', '남', True]
    sr = pd.Series(tup_data, index=['name', 'jumin', 'sex', 'student'])
    print(sr[1], "\n")
    print(sr['jumin'], "\n")



    # 원소 참조 방법
    # 1.인덱스 번호 or 인덱스 이름으로 참조 가능
    print("[2-2]인덱스 번호 or 인덱스 이름으로 참조 가능")
    print(sr[0])
    # print(sr['name'])  #둘중 하나

    # 2.여러개의 원소 선택(인덱스 리스트 활용)
    print("[2-3]여러개의 원소 선택(인덱스 리스트 활용)")
    print(sr[[1, 2]], "\n")
    # print(sr[["jumin", "sex"]])    #둘중 하나

    # 3.여러개의 원소 선택(슬라이싱)
    print("[2-3]여러개의 원소 선택(슬라이싱)")
    print(sr[1:2], "\n")
    # print(sr["jumin", 'sex'])  #둘중 하나
    del pd


def DataFrame_test():
    import pandas as pd
    # 딕셔너리 -> DataFrame 변환
    # pandas.DataFrame(딕셔너리 객체)
    dict_data = {'c0': [1, 2, 3],
                 'c1': [4, 5, 6],
                 'c2': [7, 8, 9],
                 'c3': [10, 11, 12],
                 'c4': [13, 14, 15]}
    df = pd.DataFrame(dict_data)

    print(type(df))
>>>>>>> 367930bd7abb1210c0f4aa3318a60d33fb395f8b
    print(df, "\n")