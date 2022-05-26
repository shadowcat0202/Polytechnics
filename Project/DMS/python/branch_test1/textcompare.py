import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df_1 = pd.read_csv("./시영눈코입비율.csv")
df_2 = pd.read_csv("./은정눈코입비율.csv")
df_3 = pd.read_csv("./세환눈코입비율.csv")

df_1["name"] = 0
df_2["name"] = 1
df_3["name"] = 2

df = pd.concat([df_1, df_2, df_3])
df.reset_index(drop=True, inplace=True)
df = np.array(df)
df_data = df[:,:-1]
df_label = df[:, -1]
Xtrain, Xtest, Ytrain, Ytest = train_test_split(df_data, df_label, test_size=0.2, random_state=20)
print(type(Xtrain))
clf = DecisionTreeClassifier()
clf.fit(Xtrain, Ytrain)

pickle.dump(clf, open("./decisionTreeClassifier.sav", "wb"))
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)
result = clf.score(Xtest, Ytest)
print(result)






