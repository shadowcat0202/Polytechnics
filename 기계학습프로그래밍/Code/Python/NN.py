from tensorflow.keras.models import Sequential

model = Sequential()

"""
v1  v2  out
0   0   0
0   1   0
1   0   0
1   1   1
"""

def And_Gate(v1, v2):
    w1 = 0.4
    w2 = 0.2
    w3 = 1
    b = -0.5

    w_sum = w1 * v1 + w2 * v2 + w3*b
    return 1 if w_sum > 0 else 0





# for a in range(2):
#     for b in range(2):
#         print(f"({a}, {b}) = {And_Gate(a, b)}")





