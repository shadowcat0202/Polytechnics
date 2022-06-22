from collections import deque
import cv2

class myHead:
    def __init__(self):
        self.count_head = deque(maxlen=1)
        self.count_sleep = deque(maxlen=30)

    # (두 점 사이의 유클리드 거리 계산)
    def distance(self, p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1 / 2)

    def lowerHeadCheck(self, mark):
        """
        머리가 숙였는지 계산하는 함수
        :param mark: 68개의 랜드마크(np.array)를 넣어줌
        :return: 머리를 숙였을때 1, 안숙였을때 0 반환
        """
        # 고개 안숙였을 때 (현재)
        a = 1.5 / 5
        # 고개 숙였을 때 (1초 뒤)
        b = self.distance(mark[27], mark[30]) / self.distance(mark[30], mark[8])

        if (b > a * 1.8):
            return 1
        else:
            return 0

    def lowerHeadText(self, landmarks, frame):
        """
        머리를 숙이고 있으면 영상에 putText해주는 함수
        :param landmarks: lowerHeadCheck 함수 부를때 필요한 랜드마크(np.array)를 넣어줌
        :param frame: 글씨 적을 frame 넣어줌
        :return: 없음
        """
        self.count_head.append(self.lowerHeadCheck(landmarks))
        if (sum(self.count_head)) > 0:
            cv2.putText(frame, "lower_head!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
            self.count_sleep.append(1)
            if (sum(self.count_sleep)) > 20:
                cv2.putText(frame, "sleep_head!!", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        else:
            if sum(self.count_sleep) > 0:
                self.count_sleep.pop()

    def directionCheck(self, axis):
        """
        얼굴이 어느 방향을 보고있는지 알려주는 함수
        :param axis: 축을 넣어준다
        :return: 오른쪽을 보면 0, 중앙을 보면 2, 왼쪽을 보면 1을 반환
        """
        num = axis[3][0][0] - axis[2][0][0]
        if num >= 6:
            return 0
        elif -6 < num < 6:
            return 2
        else:
            return 1

    def directionText(self, axis, frame):
        """
        얼굴 방향을 영상에 putText해주는 함수
        :param axis: directionCheck 함수 부를때 필요한 축을 넣어줌
        :param frame: 글씨 적을 frame 넣어줌
        :return: 없음
        """
        axis = axis.tolist()  # numpy array를 list로 변환
        if self.directionCheck(axis) == 0:
            cv2.putText(frame, "RIGHT", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        elif self.directionCheck(axis) == 2:
            cv2.putText(frame, "FACADE", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        elif self.directionCheck(axis) == 1:
            cv2.putText(frame, "LEFT", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)