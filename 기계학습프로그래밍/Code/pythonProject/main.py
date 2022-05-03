import pandas as pd
import random as rand

import SalmonBass as sb


def score():
    return rand.randint(0, 100)

def stub():
    dic = {"Name": "전세환"}
    print(dic)

    dic["Name"] = "새로운 이름"
    print(dic)

    # 하나의 key값에 여러개의 list를 할당 가능
    nameList = ["강현우", "김근형", "김동혁"]
    dic["Name"] = nameList
    print(dic)

    dic["id"] = [20220321, 20220322]
    print(dic)

    dic["id"].append(20220323)
    print(dic)

    df = pd.DataFrame(dic)
    print(df)

    df["국어"] = 0
    print(df)

    df["국어"] = [score(), score(), score()]
    print(df)

    # 행을 접근 할 경우에는 loc함수를 사용하여 loc[index]로 접근
    # 기존의 행을 바꾸기
    df.loc[0] = ["신주석", "2022-0", score()]
    print(df)

    # 새로 행 추가하기
    df.loc[3] = ["남현진", "2021-3", score()]
    df.loc[4] = 0
    print(df)

    # 하드코딩 하지 않고, 제일 마지막 행에 추가하려면?
    # shape 함수를 이용한다. 자료의 크기를 알려준다.
    # shape = (행, 열)
    print(df.shape)

    # shape[0], shape[1]을 출력해보자.
    print(df.shape[0])
    print(df.shape[1])
    df.loc[df.shape[0]] = ["박수연", "2022-5", score()]
    print(df)

    df.loc[1, "id"] = "2022-6"
    df.loc[4] = ["노영하", "2021-10", score()]
    print(df)
    df.loc[6] = ["새로운", "2021-new", score()]
    print(df)


def training():
    dic = {
        "ID": [],
        "이름": [],
        "점수1": [],
        "점수2": []
    }

    df = pd.DataFrame(dic)

    name = ["김근형", "김동혁", "남현진", "노영하", "박수연",
            "심우석", "안원영", "오수은", "이근혁", "이시영",
            "이은정", "이은주", "장민규", "전세환", "정경임",
            "차민욱", "최민석", "최비결", "최윤정", "최지호",
            "표주혁", "허진행"]

    for i in range(len(name)):
        df.loc[i] = ["2022-{0}".format(i+1), name[i], score(), score()]



if __name__ == '__main__':
    training()
    # sb.start()
    # stub()