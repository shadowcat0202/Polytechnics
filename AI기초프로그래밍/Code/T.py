import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
def draw_bar(Data_Frame, column_name):
    header_name_list = Data_Frame.columns.tolist()
    unique_count = Data_Frame[column_name].value_counts()
    index_list = unique_count.index.tolist()
    # print(unique_count, index_list)
    plt.bar(index_list, unique_count)
    plt.xlabel(f"{column_name}", fontsize=18)
    plt.ylabel(f"count", fontsize=18)
    plt.show()

def pokemon():
    original_data = pd.read_csv("dataset/pokemon/Pokemon.csv", sep=',', index_col=0)
    print(original_data)

    original_data.info()

    draw_bar(original_data, "Generation")


x_data = [[1, 2], [2, 3], [3, 1],
          [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]


