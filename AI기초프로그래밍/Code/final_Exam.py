"""
Matrix Multiplication 코드 작성 MML_Exam_1.py로 작성 제출
조건1. 라이브러리를 사용하지 마록 직접 구현할 것(라이브러리 작성시 30% 인정)
조건2. n * m 메트릭스 생성하고, n x m * m x n = n x n  결과가 나오도록 입력 데이터 생성 할 것(입력 데이터 생성하는 부분 함수로 작성), n m: >= 3 and <= 20, *:matrix multiplication (Not Element-wise multiplication)
np.random.seed(8)로 셋팅하여 동일한 랜덤 값이 출력되도록 셋팅 최종 6 x 6 결과 작성
※ 코드 제출 전 Octave Online을 통하여 3x3 matraix 곱하기 먼저 확인
"""

import numpy as np

np.random.seed(8)


def gen_data(mat, n, m):
    for i in range(n):
        for j in range(m):
            mat[i][j] = np.random.randint(0, 11)


def matmul(mat1, mat2, result):
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for u in range(len(mat1[0])):
                result[i][j] += (mat1[i][u] * mat2[u][j])



def MML_Exam_1():
    n = np.random.randint(3, 20)
    m = np.random.randint(3, 20)
    # n = 4
    # m = 3
    print(n, m)
    mat1 = np.zeros((n, m))
    mat2 = np.zeros((m, n))
    result = np.zeros((n, n))
    gen_data(mat1, n, m)
    gen_data(mat2, m, n)
    # print(mat1)
    # print(mat2)
    matmul(mat1, mat2, result)
    print(result, result.shape)
MML_Exam_1()

"""
DB에는 총 101개의 문서 존재 , DB에 저장된 문서는 모두 Vectorization 수행했음, 첫번째 문서의 vectorization 결과는 다음과 같다
vec_doc_1 = [0.2, 0.4, 0.02, 0.1, 0.9, 0.8]
100개의 문서중 첫 번째 문서화 가장 유사한 문서를 검색하는 코드를 작성하고, MML_Exam_2.py에 작성하여 제출
조건1. 100 x 6 array를 만들고 np.random.seed(1234)로 랜덤 값이 고정, np.random.random() 함수 사용 각 칼럼 값을 랜덤하게 생성
조건2. Inner Production을 이용하여 가장 유사도가 높은 문서(row)를 선택할 것 (Inner Production을 이용하는 부분은 함수로 만들 것, 출력:(인덱스 값))※값을 소수점 3자리에서 반올림
"""


def inner_product(vectors, in_vec):
    max = [0, 0]
    for i in range(len(vectors)):
        mse = 0
        for v in range(len(vectors[i])):
            # mse += np.sqrt((vectors[i][v] - in_vec[v]) ** 2)
            mse += vectors[i][v] * in_vec[v]
        if max[0] < mse:
            max[0] = mse
            max[1] = i
    max = (round(max[0], 2), max[1])
    return max


def MML_Exam_2():
    np.random.seed(1234)
    doc_vec = np.zeros((100, 6))
    for i in range(100):
        for j in range(6):
            doc_vec[i][j] = round(np.random.random(), 2)
    # print(doc_vec)
    vec_doc_1 = [0.2, 0.4, 0.02, 0.1, 0.9, 0.8]

    print(inner_product(doc_vec, vec_doc_1))
print(MML_Exam_2())


"""
입력층에 있는 Weight들이 출력 층의 Error에 미치는 영향을 계산하기 위해서 Chain-Rule을 이용한 Bp을 수행해야한다.
다음과 같은 신경망이 주어졌을때, Bp에 의해 업데이트 되는 Weight들의 구하는 코드를 작성 MML_Exam_3.py로 작성
(실제 코드 작성시에는 W=[0.2, 0.23, 0.38, 0.35, 0.42, 0.32, 0.64, 0.6] 로 셋팅 해서 도출
조건 1. Weight 5,6,7,8이 Total Error에 미치는 영향을 구하는 코드만 작성하시오 Bp 1단계
조건 2. Forward phase에서 계산되어지는 값을 미리 계산해놓고 z3, z4를 z5,6,7,8로 편미분한 값도 미리 계산해 놓는다.
조건 3. Sigmoid 함수는 별도로 만들어서 사용하며, Learning Rate=0.5로 셋팅한다.

업데이트된 Weight(w5~8)작성
"""


def sigmoid(val):
    return 1 / (1 + np.exp(-val))


def backpropagation_1(weight, d_weight, target, output):
    """
    code
    """


def MML_Exam_3():
    x =[0, 0.1, 0.2]
    lr = 0.5
    """
    code
    """
    target_o1 = 0.1
    target_o2 = 0.2
    o1 = 0.4
    o2 = 0.6

    w = [0, 0.2, 0.23, 0.38, 0.35, 0.42, 0.32, 0.64, 0.6]
    h = [0 for _ in range(3)]
    z = [0 for _ in range(5)]

    """
    forward
    """
    z[1] = w[1] * x[1] + w[2] * x[2]
    z[2] = w[3] * x[1] + w[4] * x[2]
    h[1] = sigmoid(z[1])
    h[2] = sigmoid(z[2])

    z[3] = w[5] * h[1] + w[6] * h[2]
    z[4] = w[7] * h[1] + w[8] * h[2]

    target_o1 = sigmoid(z[3])
    target_o2 = sigmoid(z[4])




    z[1] = x[1]

    z[3] = w[5] * h[1] + w[6] * h[2]
    dz3_w5 = h[1]
    dz3_w6 = h[2]
    z[4] = w[7] * h[1] + w[8] * h[2]
    print(z[3], z[4])
    dz4_w7 = h[1]
    dz4_w8 = h[2]

    # Eo1 = 0.5 * (target_o1 - o1) ** 2
    # Eo2 = 0.5 * (target_o2 - o2) ** 2
    # Etotal = Eo1 + Eo2
    #
    # print(z[1], z[2], h[1], h[2], z[3], z[4], o1, o2, Etotal)
    # print(backpropagation_1(w[5], dz3_w5, target_o1, o1))
    # print(backpropagation_1(w[6], dz3_w6, target_o1, o1))
MML_Exam_3()
