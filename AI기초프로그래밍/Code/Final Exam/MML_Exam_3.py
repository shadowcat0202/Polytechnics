"""
입력층에 있는 Weight들이 출력 층의 Error에 미치는 영향을 계산하기 위해서 Chain-Rule을 이용한 Bp을 수행해야한다.
다음과 같은 신경망이 주어졌을때, Bp에 의해 업데이트 되는 Weight들의 구하는 코드를 작성 MML_Exam_3.py로 작성
(실제 코드 작성시에는 W=[0.2, 0.23, 0.38, 0.35, 0.42, 0.32, 0.64, 0.6] 로 셋팅 해서 도출
조건 1. Weight 5,6,7,8이 Total Error에 미치는 영향을 구하는 코드만 작성하시오 Bp 1단계
조건 2. Forward phase에서 계산되어지는 값을 미리 계산해놓고 z3, z4를 z5,z6,z7,z8로 편미분한 값도 미리 계산해 놓는다.
조건 3. Sigmoid 함수는 별도로 만들어서 사용하며, Learning Rate=0.5로 셋팅한다.

업데이트된 Weight(w5~8)작성
"""
import numpy as np


def sigmoid(val):
    return 1 / (1 + np.exp(-val))


def backpropagation_1(weight, d_weight, target, output):
    """
    :param weight: weigth
    :param d_weight: dz / dw
    :param target: 실값
    :param output: 예측값
    :return: new weight
    """
    dEo = -(target - output)
    doz = output * (1 - output)
    dEw = dEo * doz * d_weight
    newW = weight - (lr * dEw)
    return round(newW, 8)

x =[0, 0.1, 0.2]
lr = 0.5

w = [0, 0.2, 0.23, 0.38, 0.35, 0.42, 0.32, 0.64, 0.5]
h = [0 for _ in range(3)]
z = [0 for _ in range(5)]

target_o1 = 0.4
target_o2 = 0.6
"""
forward
"""
z[1] = w[1] * x[1] + w[2] * x[2]
z[2] = w[3] * x[1] + w[4] * x[2]
h[1] = sigmoid(z[1])
h[2] = sigmoid(z[2])

z[3] = w[5] * h[1] + w[6] * h[2]
z[4] = w[7] * h[1] + w[8] * h[2]

dz3_w5 = h[1]
dz3_w6 = h[2]
dz4_w7 = h[1]
dz4_w8 = h[2]

o1 = sigmoid(z[3])
o2 = sigmoid(z[4])

eo1 = 0.5 * (target_o1 - o1) ** 2
eo2 = 0.5 * (target_o2 - o2) ** 2
print(round(eo1,8), round(eo2, 8))

Etotal = eo1 + eo2


print(z[1], z[2], h[1], h[2], z[3], z[4], o1, o2, Etotal)
print(backpropagation_1(w[5], dz3_w5, target_o1, o1))
print(backpropagation_1(w[6], dz3_w6, target_o1, o1))
print(backpropagation_1(w[7], dz4_w7, target_o2, o2))
print(backpropagation_1(w[8], dz4_w8, target_o2, o2))