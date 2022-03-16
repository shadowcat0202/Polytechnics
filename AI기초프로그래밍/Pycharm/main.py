import math

import homework


def pandas_test():
    import pandas as pd

    data = {
        'name': ['안녕', '나는', '이름'],
        'ID': ['2022-01', '2022-02', '2022-03']
    }

    df = pd.DataFrame(data)
    print(df)


def stub():
    import basic.valuable as ba
    ba.float_type()


def why_use_mod():
    num = 0.0
    arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(100):
        print(arr[i % 10], end=" ")
        if i % 10 == 9:
            print()


def multiplication_table():
    for i in range(2, 10):
        print(f"{i}단----------------------------")
        for j in range(1, 10):
            print(i, "*", j, "=", i * j)


import basic.String as basic
basic.practic6_1_1()

# basic.practic6_1_1()
# print("\n")
# basic.practic6_1_2()

homework.학습데이터처리숙제_3월말까지()