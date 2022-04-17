import re

import pandas as pd

train_df = pd.read_csv("./dataset/titanic/train.csv")
test_df = pd.read_csv("./dataset/titanic/test.csv")

combin = [train_df, test_df]


def get_title(name):
    title_search = re.search(" ([A-Za-z]+)\.", name)
    if title_search:
        return title_search.group(1)
    return ""


string = "Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)"
title_search = re.search(" ([A-Za-z]+)\.", string)
# for i in range(title_search.span()[]):
#     # for j in title_search.span()[1]:
#     print(title_search.group(i))