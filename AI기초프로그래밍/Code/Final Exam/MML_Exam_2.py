"""
DB에는 총 101개의 문서 존재 , DB에 저장된 문서는 모두 Vectorization 수행했음, 첫번째 문서의 vectorization 결과는 다음과 같다
vec_doc_1 = [0.2, 0.4, 0.02, 0.1, 0.9, 0.8]
100개의 문서중 첫 번째 문서화 가장 유사한 문서를 검색하는 코드를 작성하고, MML_Exam_2.py에 작성하여 제출
조건1. 100 x 6 array를 만들고 np.random.seed(1234)로 랜덤 값이 고정, np.random.random() 함수 사용 각 칼럼 값을 랜덤하게 생성
조건2. Inner Production을 이용하여 가장 유사도가 높은 문서(row)를 선택할 것 (Inner Production을 이용하는 부분은 함수로 만들 것, 출력:(인덱스 값))※값을 소수점 3자리에서 반올림
"""
import numpy as np


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


np.random.seed(1234)
doc_vec = np.zeros((100, 6))
for i in range(100):
    for j in range(6):
        doc_vec[i][j] = round(np.random.random(), 2)
# print(doc_vec)
vec_doc_1 = [0.2, 0.4, 0.02, 0.1, 0.9, 0.8]

print(inner_product(doc_vec, vec_doc_1))
