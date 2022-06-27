import cv2
from collections import deque

class DecisionModel:
    def __init__(self):
        self.frame = 900
        self.eyeCloseMinMax = [0, 0]
        self.headDropMinMax = [0, 0]
        self.eyeDirectionMinMax = [0, 0]

        self.eyeCloseFrame = deque(maxlen=self.frame)
        self.headDropFrame = deque(maxlen=self.frame)
        self.eyeDirectionFrame = deque(maxlen=self.frame)

        self.eyeCloseTotal = 0
        self.headDropTotal = 0
        self.eyeDirectionTotal = 0




    def analyze(self):
        pass

    def Update(self, eyeClose, headDrop, eyeDirection):
        if self.eyeCloseFrame.__sizeof__() == self.frame:
            self.eyeCloseTotal += (eyeClose - self.eyeCloseFrame.popleft())
            self.eyeCloseFrame.append(eyeClose)
        else:
            self.eyeCloseTotal += eyeClose
            self.eyeCloseFrame.append(eyeClose)

        if self.headDropFrame.__sizeof__() == self.frame:
            self.headDropTotal += (headDrop - self.headDropFrame.popleft())
            self.headDropFrame.append(headDrop)
        else:
            self.headDropTotal += headDrop
            self.headDropFrame.append(headDrop)

        if self.eyeDirectionFrame.__sizeof__() == self.frame:
            self.eyeDirectionTotal += (eyeDirection - self.eyeDirectionFrame.popleft())
            self.eyeDirectionFrame.append(eyeDirection)
        else:
            self.eyeDirectionTotal += eyeDirection
            self.eyeDirectionFrame.append(eyeDirection)

