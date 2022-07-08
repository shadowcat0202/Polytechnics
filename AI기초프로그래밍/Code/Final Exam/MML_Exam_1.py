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