import cv2
from collections import deque


class DecisionModel:
    def __init__(self):
        self.frame = 300
        self.shortLen = 30
        self.longLen = 90

        self.headDropShortSum = 0
        self.headDropShortTerm = deque(maxlen=self.shortLen)
        self.headDropLongSum = 0
        self.headDroLongTerm = deque(maxlen=self.longLen)

        self.eyeCloseShortSum = 0
        self.eyeCloseShortTerm = deque(maxlen=self.shortLen)
        self.eyeCloseLongSum = 0
        self.eyeCloseLongTerm = deque(maxlen=self.longLen)

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
        :param eyeDirection: 눈 방향 정면 0 보는거만 정상으로
        :return:
        """
        eye_close = None
        head_drop = None
        """
        headDrop Update
        """
        if len(self.headDropShortTerm) == self.shortLen:
            self.headDropShortSum += (headDrop - self.headDropShortTerm.popleft())
            self.headDropShortTerm.append(headDrop)
        else:
            self.headDropShortSum += headDrop
            self.headDropShortTerm.append(headDrop)

        if len(self.eyeCloseLongTerm) == self.shortLen:
            self.headDropLongSum += (headDrop - self.eyeCloseLongTerm.popleft())
            self.eyeCloseLongTerm.append(headDrop)
        else:
            self.headDropLongSum += headDrop
            self.eyeCloseLongTerm.append(headDrop)

        if self.headDropShortSum < 15 and self.headDropLongSum < 50:
            # SleepStats_result = "Awake"
            head_drop = 0
            # cv2.putText(img, f"Awake", (20, 300), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
        elif self.headDropShortSum >= 15 and self.headDropLongSum >= 50:
            # SleepStats_result = "Sleep"
            head_drop = 1
            # cv2.putText(img, f"Sleep", (20, 300), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
        else:
            # SleepStats_result = "Drows"
            head_drop = 1
            # cv2.putText(img, f"Drows", (20, 300), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)

        """
        eyeClose frame buffer
        """
        eyeClose = not eyeClose
        if len(self.eyeCloseShortTerm) == self.shortLen:
            self.eyeCloseShortSum += (eyeClose - self.eyeCloseShortTerm.popleft())
            self.eyeCloseShortTerm.append(eyeClose)
        else:
            self.eyeCloseShortSum += eyeClose
            self.eyeCloseShortTerm.append(eyeClose)

        if len(self.eyeCloseLongTerm) == self.longLen:
            self.eyeCloseLongSum += (eyeClose - self.eyeCloseLongTerm.popleft())
            self.eyeCloseLongTerm.append(eyeClose)
        else:
            self.eyeCloseLongSum += eyeClose
            self.eyeCloseLongTerm.append(eyeClose)

        """
        Awake, Drows, Sleep check!
        """

        if self.eyeCloseShortSum < 15 and self.eyeCloseLongSum < 50:
            # eyeclose_r = "Awake"
            eyeclose_r = 0
            # cv2.putText(img, f"Awake", (20, 300), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
        elif self.eyeCloseShortSum >= 15 and self.eyeCloseLongSum >= 50:
            # eyeclose_r = "Sleep"
            eyeclose_r = 1
            # cv2.putText(img, f"Sleep", (20, 300), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
        else:
            # eyeclose_r = "Drows"
            eyeclose_r = 1
            # cv2.putText(img, f"Drows", (20, 300), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)

        # if self.eyeDirectionFrame.__sizeof__() == self.frame:
        #     self.eyeDirectionTotal += (eyeDirection - self.eyeDirectionFrame.popleft())
        #     self.eyeDirectionFrame.append(eyeDirection)
        # else:
        #     self.eyeDirectionTotal += eyeDirection
        #     self.eyeDirectionFrame.append(eyeDirection)

        """
        warning
        """
        if eye_close or head_drop:
            return True
        return False
