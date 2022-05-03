import pandas as pd

df = pd.read_csv("./dataset/auto-mpg.csv", header=None)

df.columns = ["mpg", "cylinders", "displacement", "horseposer", "weight", "acceleration", "model_year", "origin", "name"]
print(df.head())


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print(df.shape)
