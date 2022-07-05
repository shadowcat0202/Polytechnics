import numpy as np
import torch

t = np.array([0.,1.,2.,3,4.,5.])
print(t.shape)
t = torch.FloatTensor(t)
print(t)