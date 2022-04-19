import pandas as pd
import matplotlib.pyplot as plt
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

plt.style.use('ggplot')
df = pd.read_excel("./dataset/시도별 전출입 인구수.xlsx")
df = df.fillna(method="ffill")

Seoul_to = df.iloc[19:37,:]
print(Seoul_to)
mask = (df["전출지별"] == "서울특별시") #& (df["전입지별"] == "경기도")
Seoul_to = df[mask]
Seoul_to.drop(["전출지별"], axis=1, inplace=True)
Seoul_to.rename({"전입지별":"전입지"}, axis=1, inplace=True)
Seoul_to.set_index("전입지", inplace=True)
print(Seoul_to)

s_to_g = Seoul_to.loc["경기도"]

plt.subplot(1,2,1)
plt.plot(s_to_g)
plt.title("Seoul_to_Gyeonggi")
plt.xlabel("year",)
plt.xticks(rotation=90)
plt.ylabel("number")



s_to_d = Seoul_to.loc["대구광역시"]

s_to_d.fillna(0)
s_to_d.replace("-", 0, inplace=True)
s_to_d.replace(0, s_to_d.mean(), inplace=True)
# s_to_d.replace(0, s_to_d.mean(), inplace=True)
print(s_to_d)

plt.subplot(1,2,2)
plt.plot(s_to_d)
plt.title("Seoul_to_Daegu")
plt.xlabel("year",)
plt.xticks(rotation=90)
plt.ylabel("number")
plt.show()




