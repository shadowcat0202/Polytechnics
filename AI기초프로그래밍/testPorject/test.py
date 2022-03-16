import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/' 'machine-learning-databases/iris/iris.data', header=None)
print(df.head())

sepal_l = df.iloc[0:100, 0]
sepal_w = df.iloc[0:100, 1]
petal_l = df.iloc[0:100, 2]
petal_w = df.iloc[0:100, 3]
