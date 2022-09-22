from collections import deque
import cv2
import numpy as np
class myHead:
    def __init__(self, frame_control):
        self.count_head = deque(maxlen=1)
        self.count_sleep = deque(maxlen=30)
        self.frame_control = frame_control

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
        # print(f"b{b}, a{a}")
        return b, b > a * 1.8

    def lowerHeadText(self, landmarks, frame):
        """
        머리를 숙이고 있으면 영상에 putText해주는 함수
        :param landmarks: lowerHeadCheck 함수 부를때 필요한 랜드마크(np.array)를 넣어줌
        :param frame: 글씨 적을 frame 넣어줌
        :return: 없음
        """
        sleepHead = False
        _, lower = self.lowerHeadCheck(landmarks)
        self.count_head.append(lower)
        if (self.count_head.count(True)) > 0:
            cv2.putText(frame, "lower_head!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
            self.count_sleep.append(True)
            if (self.count_sleep.count(True)) > 20:
                sleepHead = True
                cv2.putText(frame, "sleep_head!!", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        else:
            if self.count_sleep.count(True) > 0:
                self.count_sleep.pop()
        return sleepHead

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
            # return "LEFT"
            return 1
            # cv2.putText(frame, "LEFT", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        elif self.directionCheck(axis) == 2:
            # return "FACADE"
            return 0
            # cv2.putText(frame, "FACADE", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        elif self.directionCheck(axis) == 1:
            # return "RIGHT"
            return 1
            # cv2.putText(frame, "RIGHT", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

    """ SY """


    def updatedLog_modeRatio_tholdRatio(self, headLog, headRatio):
        ratio = headRatio
        keys = np.array(list(headLog.keys()))

        headLog[ratio] = headLog[ratio]+1 if ratio in keys else 1

        max_inDic = max(headLog, key=headLog.get)
        headLog_sorted = sorted(headLog.items(), key=lambda x: x[0])
        arr_sorted = np.array(headLog_sorted)

        i = 0
        num_row = len(arr_sorted)
        # print(num_row)
        while (i<num_row):
            if arr_sorted[i][0] == max_inDic:
                break
            else:
                i += 1
        # print(f"index of mode = {i}")

        while (i+1<num_row):
            if arr_sorted[i+1][1] <= arr_sorted[i][1]:
                i += 1
            else:
                break
        # print(f"index of optimal_value = {i}")
        thold = arr_sorted[i][0]

        return headLog, max_inDic, thold

    def head_down(self, ratio, thold):
        status = 1 if ratio > thold else 0

        return status


    def headMinlMax_and_nmRatio(self, arr_minlmax, ratio):
        arr_minlmax[0] = ratio if ratio<arr_minlmax[0] else arr_minlmax[0]
        arr_minlmax[1] = ratio if ratio>arr_minlmax[1] else arr_minlmax[1]

        range = arr_minlmax[1]-arr_minlmax[0]
        nmRatio = (ratio - arr_minlmax[0])/range if range !=0 else np.nan

        return arr_minlmax, nmRatio




    def updated_statusLog(self, arr_log, thold, ratio):
        arr_status = arr_log
        # print(f"log/thold/ratio = {arr_log}/{thold}/{ratio}")
        if ratio>thold:
            output = 1
        else:
            output = 0
        arr_status = np.append(arr_status, output)

        return arr_status