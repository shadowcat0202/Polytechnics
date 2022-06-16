import matplotlib.pyplot as plt

W = 10
pos = []
for d1 in range(1,10):
    d2 = W - d1
    pos.append(round((d1 / d2 / W), 3) )

print(pos)
