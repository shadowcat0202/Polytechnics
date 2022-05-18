import matplotlib.pyplot as plt
import pandas as pd
import seaborn
df = pd.read_csv("../references/한국산업안전보건공단_규모별 사고재해자수_12_31_2020.csv", encoding='cp949')

col_name = [df.columns[0]]
for name in df.columns[1:]:
    col_name.append(name[:4])

df.columns = col_name

df = df.T
df = df.rename(columns=df.iloc[0])
df = df.drop(df.index[0])
print(df)
fig, ax_base = plt.subplots(figsize=(40,5))

lis = ax_base.plot(df[df.columns[0]], label=df.columns[0])
lis_buf = [df.columns[0]]
for name in df.columns[1:]:
    lis = ax_base.plot(df[name], label=df.columns[0])
    lis_buf.append(name)
labs = [l.get_label() for l in lis_buf]
ax_base.legend(lis_buf, labs, loc=1)

ax_base.grid()
plt.show()
# line1 = ax1.plot(df[""])



