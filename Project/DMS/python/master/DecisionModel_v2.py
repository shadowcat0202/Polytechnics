import cv2
from collections import deque


class DecisionModel:
    def __init__(self):
        self.frame = 300

        self.headDropSum = 0
        self.headDropFrame = deque(maxlen=self.frame)
        self.headDropMinSum = 0
        self.headDropMinFrame = deque(maxlen=self.frame)
        self.headDropMaxSum = 0
        self.headDropMaxFrame = deque(maxlen=self.frame)
        self.headDropMinMax = [1000, -1000]

        self.eyeCloseShortLen = 30
        self.eyeCloseShortSum = 0
        self.eyeCloseShortTerm = deque(maxlen=self.eyeCloseShortLen)
        self.eyeCloseLongLen = 90
        self.eyeCloseLongSum = 0
        self.eyeCloseLongTerm = deque(maxlen=self.eyeCloseLongLen)

        # self.eyeDirectionFrame = deque(maxlen=self.frame)

        # self.eyeCloseTotal = 0

        # self.eyeDirectionTotal = 0

    def analyze(self):
        pass

    def Update(self, img, eyeClose, headDrop, eyeDirection):
        """
        :param img: 원본 이미지
        :param eyeClose: 눈 감김 boolean
        :param headDrop: ratio
        :param eyeDirection:
        :return:
        """

        """
        headDrop Update
        """
        self.headDropMinMax[0] = headDrop if self.headDropMinMax[0] > headDrop else self.headDropMinMax[0]
        self.headDropMinMax[1] = headDrop if self.headDropMinMax[1] < headDrop else self.headDropMinMax[1]
        if len(self.headDropFrame) == self.frame:
            self.headDropSum += (headDrop - self.headDropFrame.popleft())
            self.headDropFrame.append(headDrop)
            cv2.putText(img, f"headDropAvg: {round(self.headDropSum / self.frame, 4)}",
                        (20, 20), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
            cv2.putText(img, f"headDropMin{round(self.headDropMinMax[0], 4)}",
                        (20, 40), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
            cv2.putText(img, f"headDropMax{round(self.headDropMinMax[1], 4)}",
                        (20, 60), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
        else:
            self.headDropSum += headDrop
            self.headDropFrame.append(headDrop)

        """
        eyeClose frame buffer
        """
        eyeClose = not eyeClose
        if len(self.eyeCloseShortTerm) == self.eyeCloseShortLen:
            self.eyeCloseShortSum += (eyeClose - self.eyeCloseShortTerm.popleft())
            self.eyeCloseShortTerm.append(eyeClose)
        else:
            self.eyeCloseShortSum += eyeClose
            self.eyeCloseShortTerm.append(eyeClose)

        if len(self.eyeCloseLongTerm) == self.eyeCloseLongLen:
            self.eyeCloseLongSum += (eyeClose - self.eyeCloseLongTerm.popleft())
            self.eyeCloseLongTerm.append(eyeClose)
        else:
            self.eyeCloseLongSum += eyeClose
            self.eyeCloseLongTerm.append(eyeClose)

        """
        Awake, Drows, Sleep check!
        """
        s_SleepStats = ""
        if self.eyeCloseShortSum < 15 and self.eyeCloseLongSum < 50:
            s_SleepStats = "Awake"
            # cv2.putText(img, f"Awake", (20, 300), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
        elif self.eyeCloseShortSum >= 15 and self.eyeCloseLongSum >= 50:
            s_SleepStats = "Sleep"
            # cv2.putText(img, f"Sleep", (20, 300), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
        else:
            s_SleepStats = "Drows"
            # cv2.putText(img, f"Drows", (20, 300), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)


        # if self.eyeDirectionFrame.__sizeof__() == self.frame:
        #     self.eyeDirectionTotal += (eyeDirection - self.eyeDirectionFrame.popleft())
        #     self.eyeDirectionFrame.append(eyeDirection)
        # else:
        #     self.eyeDirectionTotal += eyeDirection
        #     self.eyeDirectionFrame.append(eyeDirection)

        return s_SleepStats
