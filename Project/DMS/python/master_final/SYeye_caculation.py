class EyeCalculation():
    def __init__(self, newRatio, extRatio_minMax):
        self.ratio_input = newRatio
        self.ratio_extMin, self.ratio_extMax = extRatio_minMax
        self.ratio_newMin, self.ratio_newMax = self.get_min_max_eyeRatio()
        self.pct_ratio = self.evaluate_ratio_fromRange()


    def get_min_max_eyeRatio(self):
        ratio_min = self.ratio_extMin
        ratio_max = self.ratio_extMax

        if self.ratio_input > ratio_max:
            ratio_max = self.ratio_input

        if self.ratio_input < ratio_min:
            ratio_min = self.ratio_input

        return [ratio_min, ratio_max]

    def evaluate_ratio_fromRange(self):
        ratio_range = self.ratio_newMax - self.ratio_newMin
        pct = self.ratio_input / ratio_range

        return pct






# def distance(p1, p2):
#     return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1 / 2)
#
#
# def ER_ratio(self, landmarks):
#     # 얼굴 특징점 번호 사진 참조
#     A = distance(landmarks[1], landmarks[5])
#     B = distance(landmarks[2], landmarks[4])
#     C = distance(landmarks[0], landmarks[3])
#
#     return (A + B) / (2.0 * C)
