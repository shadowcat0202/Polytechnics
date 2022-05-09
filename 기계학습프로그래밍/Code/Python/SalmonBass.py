import pandas as pd

from matplotlib.animation import adjusted_figsize

df = pd.read_csv("dataset/salmon_bass_data.csv")

X = []
y = []
for i in range(len(df)):
    fish = [df.loc[i, "Length"], df.loc[i, "Lightness"]]
    X.append(fish)
    y.append(df.loc[i, "Class"])

from sklearn import tree
dtree = tree.DecisionTreeClassifier(max_depth=3)
dtree = dtree.fit(X, y)

from matplotlib import pyplot as plt
plt.figure(figsize=(10,10))
tree.plot_tree(dtree, fontsize=8, filled=True)
plt.show()


