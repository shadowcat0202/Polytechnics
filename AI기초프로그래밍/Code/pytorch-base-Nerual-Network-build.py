import pprint
import time
import torch
from torch import cuda

print(torch.__version__)

a = [15, 17, 19.2, 22.3, 20, 19, 16]
temp = torch.FloatTensor(a)

print(temp.size(), temp.dim())

print(f"월, 화 평균온도는 : {temp[0]}, {temp[1]}")
print(f"화 ~ 목 평균온도는 : {temp[1:4]}")

t = torch.FloatTensor(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ]
)

print(t.size(), t.dim())

print(t[1:3, 1:3])

from sklearn.datasets import load_boston

boston = load_boston()
# data = torch.from_numpy(boston['data'])
# feature_names = torch.from_numpy(boston['feature_names'])
# target = torch.from_numpy(boston['target'])
# pprint.pprint(boston)
# pprint.pprint(data)
# pprint.pprint(feature_names)
# pprint.pprint(target)

data_tensor = torch.from_numpy(boston.data)

print(data_tensor.size(), data_tensor.dim())