import cv2

class myMouth:
    def __init__(self):
        self.count_mouth = 0
        self.max_count = 50
        self.min_count = 0

    # (두 점 사이의 유클리드 거리 계산)
    def distance(self, p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1 / 2)

    def openMouthCheck(self, mark):
        """
        입을 열고 있는지 확인하는 함수
        :param mark: 68개의 랜드마크(np.array)를 넣어줌
        :return: 입을 열고있으면 1, 입을 닫고있으면 0을 반환
        """
        # 입을 다물었을 때 (평상시)
        a = 1 / 2
        # 하품할 때
        b = self.distance(mark[62], mark[66]) / self.distance(mark[60], mark[64])

        if (b > a * 1.5):
            return 1
        else:
            return 0

    def openMouthText(self, landmarks, frame):
        """
        하품(입을 열고 있는 시간이 일정 시간 지나면)하면 putText해주는 함수
        :param landmarks: openMouthCheck 함수 부를때 필요한 랜드마크(np.array)를 넣어줌
        :param frame: 글씨 적을 frame 넣어줌
        :return: 없음
        """
        if (self.openMouthCheck(landmarks) == 1):
            if self.count_mouth < self.max_count:
                self.count_mouth += 1
        else:
            if self.count_mouth > self.min_count:
                self.count_mouth -= 1
        if self.count_mouth > 35:
            cv2.putText(frame, "Yawn!!", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)