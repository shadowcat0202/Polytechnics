import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # 매트플랏라이브러리 도움으로 폰트매니저 사용
import matplotlib

font_location = 'C:/Windows/Fonts/HMKMMAG.TTF'

font_name = fm.FontProperties(fname=font_location).get_name()

matplotlib.rc('font', family=font_name)

import seaborn as sns


# print(plt.style.available)
# df = pd.read_excel("./dataset/남북한발전전력량.xlsx")
# # print(df.info())
# print(df.head(10))
# df1 = df.iloc[[0,5], 2:]
#
# df1.index = ["South", "North"]
# print(df1)
#
# df2 = df1.T
# print(df2.head(4))
# df2.plot(kind="bar")
# df2.plot(kind="line")
# df2.plot.bar()    # 위와 같다


# 구분선==========================================================================================================
# df = pd.read_excel("./dataset/시도별 전출입 인구수.xlsx")
# print(df.head(10))
# total = df.iloc[2, 2:].T
# print(total)
#
#
# city = df.iloc[3:20, 1:].T
# city = city.rename(columns=city.iloc[0])
# print(city.head(5))
# # for i in range(city.shape[1]):
# #     plt.subplot(city.shape[1], int(i / 5 + 1), int(i / 3 + 1))
# #     var = city.iloc[i + 1, :]
# #     print(var)
# #     var.plot(kind="bar")
#
# # plt.show()


# 구분선==========================================================================================================
# df = pd.read_csv("./dataset/auto-mpg.csv")
# print(df.shape)
# print(df.head())
# df.columns = [f"f{i}" for i in range(df.shape[1])]
# df.plot(kind="box")
# plt.show()
# df_scatter = df[["f4", "f0"]]
# sns.regplot(x="f0", y="f4", data=df_scatter, scatter_kws={"color":"green"})
# plt.show()

# 구분선==========================================================================================================
def deg_to_seou():
    plt.style.use('ggplot')
    df = pd.read_excel("./dataset/시도별 전출입 인구수.xlsx")
    df = df.fillna(method="ffill")

    Seoul_to = df.iloc[19:37, :]
    print(Seoul_to)
    mask = (df["전출지별"] == "서울특별시")  # & (df["전입지별"] == "경기도")
    Seoul_to = df[mask]
    Seoul_to.drop(["전출지별"], axis=1, inplace=True)
    Seoul_to.rename({"전입지별": "전입지"}, axis=1, inplace=True)
    Seoul_to.set_index("전입지", inplace=True)
    print(Seoul_to)

    s_to_g = Seoul_to.loc["경기도"]

    plt.subplot(1, 2, 1)
    plt.plot(s_to_g)
    plt.title("Seoul_to_Gyeonggi")
    plt.xlabel("year", )
    plt.xticks(rotation=90)
    plt.ylabel("number")

    s_to_d = Seoul_to.loc["대구광역시"]

    s_to_d.fillna(0)
    s_to_d.replace("-", 0, inplace=True)
    s_to_d.replace(0, s_to_d.mean(), inplace=True)
    # s_to_d.replace(0, s_to_d.mean(), inplace=True)
    print(s_to_d)

    plt.subplot(1, 2, 2)
    plt.plot(s_to_d)
    plt.title("Seoul_to_Daegu")
    plt.xlabel("year", )
    plt.xticks(rotation=90)
    plt.ylabel("number")
    plt.show()


# ======================================마스크 컨트롤 그래프 그리기============================================
def mask_control():
    plt.style.use('ggplot')
    df = pd.read_excel("./dataset/시도별 전출입 인구수.xlsx")
    df = df.fillna(method="ffill")

    tmp = df.iloc[19:37, 1:]
    print(tmp.head())
    # =============그래프 그리기=====================
    mask = (df["전출지별"] == "서울특별시") & (df["전입지별"] == "경기도")
    data = df[mask]
    data = data.drop(["전출지별", "전입지별"], axis=1)
    data = data.replace("-", 0)
    data = data.iloc[0, :]
    print(data)
    plt.xticks(rotation=90)
    plt.plot(data, markersize=6, color="red", marker="o")
    plt.ylim(5000, 800000)
    plt.annotate("",  # 표시할 문자
                 xy=(20, 620000),  # 화살표 머리부분
                 xytext=(2, 250000),  # 화살표의 끝부분
                 arrowprops=dict(arrowstyle="->", color="red", lw=5)
                 )
    plt.annotate("",  # 표시할 문자
                 xy=(45, 450000),  # 화살표 머리부분
                 xytext=(30, 620000),  # 화살표의 끝부분
                 arrowprops=dict(arrowstyle="->", color="blue", lw=5)
                 )
    plt.annotate("인구 이동 증가 1970~1995",  # 표시할 문자
                 xy=(10, 400000),  # 화살표 머리부분
                 rotation=42,
                 va="baseline",
                 ha="center",
                 fontsize=10
                 )
    plt.annotate("인구 이동 감소 1995~2017",  # 표시할 문자
                 xy=(40, 500000),  # 화살표 머리부분
                 rotation=-25,
                 va="baseline",
                 ha="center",
                 fontsize=10
                 )

    # ============================여러개=================================================
    to_list = [["충청남도", "red"], ["경상북도", "blue"], ["강원도", "olive"]]
    # to_list = [["경기도", "olive"]]
    fig = plt.figure(figsize=(16, 8))  # 캔버스 생성
    ax = fig.add_subplot()  # 그림 프레임 생성
    for to in to_list:
        mask = (df["전출지별"] == "서울특별시") & (df["전입지별"] == to[0])
        data = df[mask]
        data = data.drop(["전출지별", "전입지별"], axis=1)
        data = data.replace("-", 0)
        data = data.iloc[0, :]
        print(data)
        # sr_deagu = seoul_to_deagu.lic[22,:] #해당 상황에서는 행 번호을 따로 변경한게 아니기 때문에 22라는 숫자가 행 이름이다
        # ax.xticks(rotation=90)경상북도
        # ax.xticks(rotation="vertical")
        plt.xticks(rotation=90)
        ax.plot(data, markersize=6, color=to[1], marker="o")

    plt.legend([to[0] for to in to_list])
    plt.show()


# ===========================map(함수, 리스트)==================================
def use_map():
    df = pd.read_excel("./dataset/시도별 전출입 인구수.xlsx")
    df = df.fillna(method="ffill")
    # map: 리스트 요소를 지정된 함수로 처리해주는 함수
    # map(function, iterable)
    var_list = list(map(str, range(1970, 1980)))
    print(var_list)
    mask = (df["전출지별"] == "서울특별시") & (df["전입지별"] != "서울특별시")
    df_seoul = df[mask]
    df_seoul = df_seoul.drop(["전출지별"], axis=1)
    df_seoul = df_seoul.set_index("전입지별")
    print(df_seoul.head())

    col_year = list(map(str, range(1970, 2018)))

    locals = [["강원도", "red"], ["충청북도", "blue"], ["충청남도", "olive"]]
    local_name = [name[0] for name in locals]
    df_3 = df_seoul.loc[local_name, col_year]
    print(df_3.head())

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)

    # 각각의 위치에 그리기
    # ax_list = []
    # for i in range(1, 5):
    #     ax_list.append(fig.add_subplot(2, 2, i))

    for name in locals:
        ax.plot(col_year, df_3.loc[name[0], :], markersize=6, color=name[1], marker="o")

    # ax.set_facecolor("w")
    ax.grid(True)
    plt.legend(local_name)
    plt.title("서울->강원도,충청북도,충청남도")
    plt.xlabel("년도")
    plt.ylabel("인구수")

    plt.xticks(rotation=90)
    plt.show()


# ===========================선 그래프 + 히스토그램 겹치기?==================================
def north():
    plt.rcParams["axes.unicode_minus"] = False
    original = pd.read_excel("./dataset/남북한발전전력량.xlsx")
    df = original.fillna(method="ffill")

    df = df[df["전력량 (억㎾h)"] == "북한"].drop(["전력량 (억㎾h)"], axis=1)
    df.set_index("발전 전력별", inplace=True)
    df = df.T
    # 방법 1. shift
    # df["증감 1년"] = df["합계"].shift(1)
    # df["증감율"] = ((df["합계"] / df["증감 1년"]) - 1) * 100

    # 방법 2. 그냥
    df["증감율"] = np.nan
    for i in range(1, int(df.shape[0])):
        # ((현재년도 - 전년도) / 전년도) * 100
        df.iloc[i, -1] = ((df.iloc[i, 0] - df.iloc[i - 1, 0]) / df.iloc[i - 1, 1]) * 100
    print(df.head())

    ax1 = df[["수력", "화력"]].plot(kind="bar", figsize=(20, 10), stacked=True)
    ax2 = ax1.twinx()  # 히스토그램 2개 동시에 그리기
    ax2.plot(df.index, df["증감율"], ls="-", marker="o", color="green")
    ax1.set_ylim(0, 500)
    ax2.set_ylim(-50, 50)

    ax1.set_xlabel("연도")
    ax1.set_ylabel("발전량")
    ax2.set_ylabel("전년 대비 증감율(%)")
    plt.title("북한 발전량", size=20)
    ax1.legend(loc="upper left")

    plt.show()


# ===========================파이 그래프=================================================
def pichart():
    original = pd.read_csv("./dataset/auto-mpg.csv")
    df = original

    df.columns = ["mpg", "cylinders", "displacement", "horseposer", "weight", "acceleration", "model_year", "origin",
                  "name"]

    # # 방법 1. Groupby 사용을 위해서 count된 값을 넣기 위한 count 칼럼 생성
    df["count"] = 1
    print(df.head(10))
    # 제조국가 열을 기준으로 그룹화 및 합계 연산
    df = df.groupby("origin").sum()

    # 제조 국가 값(1,2,3)을 usa, eu, japan으로 변경
    df.index = ["USA", "EU", "JAPAN"]
    df["count"].plot(kind="pie", figsize=(7, 5), autopct="%.2f%%", colors=["chocolate", "bisque", "cadetblue"])
    plt.title("Model origin")
    plt.legend(labels=df.index, loc="upper left")

    # 방법 2. df.value_counts() 사용
    # cnt = df["origin"].value_counts()
    # cnt.index = ["USA", "EU", "JAPAN"]
    # cnt.plot(kind="pie", figsize=(7, 5), autopct="%.2f%%", colors=["chocolate", "bisque", "cadetblue"])
    # plt.title("Model origin")
    # plt.legend(labels=cnt.index, loc="upper left")
    # plt.show()


def Seaborn():
    import seaborn as sns

    original = pd.read_csv("./dataset/auto-mpg.csv", header=None)
    df = original

    df.columns = ["mpg", "cylinders", "displacement", "horseposer", "weight", "acceleration", "model_year", "origin",
                  "name"]

    df_scatter = df[["weight", "mpg"]]

    # 산점도에서 선형회귀 선을 같이 그리는 함수(라이브러리)
    sns.regplot(x="weight", y="mpg", data=df_scatter,
                scatter_kws={"color": "green"}, line_kws={"color": "red"})
    plt.show()
    del sns


def heatmap():
    import seaborn as sns
    flights_data = sns.load_dataset("flights")
    df = flights_data.pivot("month", "year", "passengers")
    print(df.head())
    ax = sns.heatmap(df)
    plt.title("Heatmap of Flight by Seborn")
    plt.show()

    del sns

north()
# pichart()
# Seaborn()
# heatmap()
