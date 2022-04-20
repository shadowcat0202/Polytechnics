import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
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

    w_sum = w1 * v1 + w2 * v2 + w3 * b
    return 1 if w_sum > 0 else 0


# for a in range(2):
#     for b in range(2):
#         print(f"({a}, {b}) = {And_Gate(a, b)}")

x = [[0,0],[0,1],[1,0],[1,1]]
y = [[1,0],[1,0],[1,0],[0,1]]

# 순차형?==========================================================================================================
model = Sequential()
model.add(Dense(2, input_dim=2, kernel_initializer="normal", activation="softmax"))
model.summary()

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
start = time.time()
model.fit(x,y,epochs=1000, verbose=False)
print(time.time() - start)

result = model.predict(x)
print(result)


# 함수형==========================================================================================================
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

inputs = Input(shape=(len(x[0]),))
output = Dense(2, activation="softmax")(inputs)
model = Model(inputs=inputs, outputs=output)
model.summary()

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=['accuracy'])

start = time.time()
with tf.device("/device:GPU:0"):
    model.fit(x=x, y=y, epochs=1000, batch_size=10,verbose=False) # verbose=False
print(time.time() - start)
result = model.predict(x)
print(result)