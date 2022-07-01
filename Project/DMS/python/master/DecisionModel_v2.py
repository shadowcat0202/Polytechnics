import cv2
from collections import deque


class DecisionModel:
    def __init__(self):
        self.frame = 300
        self.shortLen = 30
        self.longLen = 120

        self.shortTh = round(self.shortLen * 0.8)
        self.LongTh = round(self.longLen * 0.8)

        self.headDropShortSum = 0
        self.headDropShortTerm = deque(maxlen=self.shortLen)
        self.headDropLongSum = 0
        self.headDropLongTerm = deque(maxlen=self.longLen)

        self.eyeCloseShortSum = 0
        self.eyeCloseShortTerm = deque(maxlen=self.shortLen)
        self.eyeCloseLongSum = 0
        self.eyeCloseLongTerm = deque(maxlen=self.longLen)

        self.gazeShortSum = 0
        self.gazeShortTerm = deque(maxlen=self.shortLen)
        self.gazeLongSum = 0
        self.gazeLongTerm = deque(maxlen=self.longLen)

        self.headDirShortSum = 0
        self.headDirShortTerm = deque(maxlen=self.shortLen)
        self.headDirLongSum = 0
        self.headDirLongTerm = deque(maxlen=self.longLen)

        # self.eyeDirectionFrame = deque(maxlen=self.frame)

        # self.eyeCloseTotal = 0

        # self.eyeDirectionTotal = 0

    def analyze(self):
        pass

    def Update(self, img, eyeClose, headDrop, gaze, headDir):
        """
        :param img: 원본 이미지
        :param eyeClose: 눈 감김 boolean
        :param headDrop: ratio
        :param eyeDirection: 눈 방향 정면 0 보는거만 정상으로
        :return:
        """
        eye_close = None
        head_drop = None
        non_gaze = None
        head_side = None
        """
        headDrop Update
        """
        if len(self.headDropShortTerm) == self.shortLen:
            self.headDropShortSum += (headDrop - self.headDropShortTerm.popleft())
            self.headDropShortTerm.append(headDrop)
        else:
            self.headDropShortSum += headDrop
            self.headDropShortTerm.append(headDrop)

        if len(self.headDropLongTerm) == self.longLen:
            self.headDropLongSum += (headDrop - self.eyeCloseLongTerm.popleft())
            self.eyeCloseLongTerm.append(headDrop)
        else:
            self.headDropLongSum += headDrop
            self.headDropLongTerm.append(headDrop)

        if self.headDropShortSum < self.shortTh and self.headDropLongSum < self.LongTh:
            # SleepStats_result = "Awake"
            head_drop = False
            # cv2.putText(img, f"Awake", (20, 300), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
        elif self.headDropShortSum >= self.shortTh and self.headDropLongSum >= self.LongTh:
            # SleepStats_result = "Sleep"
            head_drop = True
            # cv2.putText(img, f"Sleep", (20, 300), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
        else:
            # SleepStats_result = "Drows"
            head_drop = True
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

        if self.eyeCloseShortSum < self.shortTh and self.eyeCloseLongSum < self.LongTh:
            # eye_close = "Awake"
            eye_close = False
            # cv2.putText(img, f"Awake", (20, 300), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
        elif self.eyeCloseShortSum >= self.shortTh and self.eyeCloseLongSum >= self.LongTh:
            # eye_close = "Sleep"
            eye_close = True
            # cv2.putText(img, f"Sleep", (20, 300), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
        else:
            # eye_close = "Drows"
            eye_close = True
            # cv2.putText(img, f"Drows", (20, 300), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)

        # if self.eyeDirectionFrame.__sizeof__() == self.frame:
        #     self.eyeDirectionTotal += (eyeDirection - self.eyeDirectionFrame.popleft())
        #     self.eyeDirectionFrame.append(eyeDirection)
        # else:
        #     self.eyeDirectionTotal += eyeDirection
        #     self.eyeDirectionFrame.append(eyeDirection)

        """
        gaze update
        """
        if gaze is not None:
            if len(self.gazeShortTerm) == self.shortLen:
                self.gazeShortSum += (gaze - self.gazeShortTerm.popleft())
                self.gazeShortTerm.append(gaze)
            else:
                self.gazeShortSum += gaze
                self.gazeShortTerm.append(gaze)

            if len(self.gazeLongTerm) == self.longLen:
                self.gazeLongSum += (gaze - self.gazeLongTerm.popleft())
                self.gazeLongTerm.append(gaze)
            else:
                self.gazeLongSum += gaze
                self.gazeLongTerm.append(gaze)

            if self.gazeShortSum < self.shortTh and self.gazeLongSum < self.LongTh:
                # SleepStats_result = "Awake"
                non_gaze = False
                # cv2.putText(img, f"Awake", (20, 300), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
            elif self.gazeShortSum >= self.shortTh and self.gazeLongSum >= self.LongTh:
                # SleepStats_result = "Sleep"
                non_gaze = True
                # cv2.putText(img, f"Sleep", (20, 300), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
            else:
                # SleepStats_result = "Drows"
                non_gaze = True
                # cv2.putText(img, f"Drows", (20, 300), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)

        """
        head direction update
        """
        if headDir is not None:
            if len(self.headDirShortTerm) == self.shortLen:
                self.headDirShortSum += (headDir - self.headDirShortTerm.popleft())
                self.headDirShortTerm.append(headDir)
            else:
                self.headDirShortSum += headDir
                self.headDirShortTerm.append(headDir)

            if len(self.headDirLongTerm) == self.longLen:
                self.headDirLongSum += (headDir - self.headDirLongTerm.popleft())
                self.headDirLongTerm.append(headDir)
            else:
                self.headDirLongSum += headDir
                self.headDirLongTerm.append(headDir)

            if self.headDirShortSum < self.shortTh and self.headDirLongSum < self.LongTh:
                # SleepStats_result = "Awake"
                head_side = False
                # cv2.putText(img, f"Awake", (20, 300), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
            elif self.headDirShortSum >= self.shortTh and self.headDirLongSum >= self.LongTh:
                # SleepStats_result = "Sleep"
                head_side = True
                # cv2.putText(img, f"Sleep", (20, 300), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
            else:
                # SleepStats_result = "Drows"
                head_side = True
                # cv2.putText(img, f"Drows", (20, 300), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)

        """
        warning
        """
        # cv2.putText(img, f"eye_close:{eyeClose}", (20, 300), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
        # cv2.putText(img, f"head_drop:{head_drop}", (20, 320), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
        # cv2.putText(img, f"non_gaze:{non_gaze}", (20, 340), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
        # cv2.putText(img, f"head_side:{head_side}", (20, 360), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
        if eye_close or head_drop or non_gaze or head_side:
            return True
        return False
