from collections import deque
import cv2


class DecisionModel:
    def __init__(self):
        self.eyeMax = 30
        self.eyeThreshold = self.eyeMax * 0.8
        self.eyeClose = deque(maxlen=self.eyeMax)
        self.eyeCloseCount = 0

        self.headMax = 60
        self.headThreshold = self.headMax * 0.8
        self.headDrop = deque(maxlen=self.headMax)
        self.headDropCount = 0

        # self.eyeWarning = 0
        # self.headWarning = 0
        # self.warningLevel = 0
        print(f"eyeThreshold:{self.eyeThreshold}")
        print(f"headThreshold:{self.headThreshold}")

    def analyze_v1(self, eye, head):
        self.eyeClose.append(True) if eye == 0 else self.eyeClose.append(False)
        self.headDrop.append(True) if head else self.headDrop.append(False)

        if self.eyeClose.count(True) > self.eyeThreshold:
            # self.eyeWarning = 3
            self.eyeCloseCount += 1
            self.eyeClose = deque(maxlen=self.eyeMax)
        # else:
        #     self.eyeWarning = 0

        if self.headDrop.count(True) > self.headThreshold:
            # self.headWarning = 2
            self.headDropCount += 1
            self.headDrop = deque(maxlen=self.headMax)
        # else:
        #     self.headWarning = 0

        # self.warningLevel = self.eyeWarning + self.headWarning
    def analyze_v2(self, eye, head):
        pass

    def drawResult(self, img):
        cv2.putText(img, f"eyeCloseCount: {self.eyeCloseCount}",
                    (20, 20), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
        cv2.putText(img, f"headDropCount: {self.headDropCount}",
                    (20, 40), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
        # cv2.putText(img, f"Warning Level: {self.warningLevel}",
        #             (20, 60), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
