import numpy as np


class Perceptron(object):
    """
    퍼셉트론 분류기

    매개변수
    ------------
    eta : float
      학습률 (0.0과 1.0 사이)
    n_iter : int
      훈련 데이터셋 반복 횟수(에포크)
    random_state : int
      가중치 무작위 초기화를 위한 난수 생성기 시드

    속성
    -----------
    w_ : 1d-array
      학습된 가중치
    errors_ : list
      에포크마다 누적된 분류 오류

    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    """
    
    훈련 데이터 학습
    

    매개변수
    ----------
    X : {array-like}, shape = [n_samples, n_features]
      n_samples개의 샘플과 n_features개의 특성으로 이루어진 훈련 데이터
    y : array-like, shape = [n_samples]
      타깃값

    반환값
    -------
    self : object

    """

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])  # array.shape는 array의 차수를 알려준다 ex)[3,4]
        self.errors_ = []

        for _ in range(self.n_iter):    #학습 횟수만큼 돌리겠다
            errors = 0
            for xi, target in zip(X, y):    #zip내장 함수는 쌍에 맞게 묶어서 object형식으로 리턴해준다
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

        return self

    #w^T와 X의 점곱 연산
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    # 추론
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
