from collections import deque
import cv2


class OutputSignalModel:
    def __init__(self):
        self.eyeMax = 30
        self.eyeThreshold = self.eyeMax * 0.8
        self.eyeClose = deque(maxlen=self.eyeMax)
        self.eyeCloseCount = 0

        self.headMax = 60
        self.headThreshold = self.headMax * 0.8
        self.headDrop = deque(maxlen=self.headMax)
        self.headDropCount = 0

        self.warningLevel = 0
        print(f"eyeThreshold:{self.eyeThreshold}")
        print(f"headThreshold:{self.headThreshold}")

    def analyze(self, eye, head):
        self.eyeClose.append(True) if eye == 0 else self.eyeClose.append(False)
        self.headDrop.append(True) if head else self.headDrop.append(False)

        if self.eyeClose.count(True) > self.eyeThreshold:
            self.eyeCloseCount += 1
            self.eyeClose = deque(maxlen=self.eyeMax)
        if self.headDrop.count(True) > self.headThreshold:
            self.headDropCount += 1
            self.headDrop = deque(maxlen=self.headMax)

    def drawResult(self, img):
        cv2.putText(img, f"eyeCloseCount: {self.eyeCloseCount}",
                    (20, 20), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
        cv2.putText(img, f"headDropCount: {self.headDropCount}",
                    (20, 40), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
