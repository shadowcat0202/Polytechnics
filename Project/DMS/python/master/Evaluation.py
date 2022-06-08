class evaluation:
    def __init__(self):
        self.total_frame = 0
        self.left = 0
        self.center = 0
        self.right = 0
        self.close = 0

    def measurement(self, pred):
        self.total_frame += 1
        if pred == "left":
            self.left += 1
        elif pred == "right":
            self.right += 1
        elif pred == "center":
            self.center += 1
        else:
            self.close += 1

    def measurement_result(self):
        print(f"total frame : {self.total_frame}")
        print(f"left:{self.left}\ncenter:{self.center}\nright:{self.right}")
        print(f"acc\tleft\tcenter\tright\tclose?")
        print(f"\t{round(self.left/self.total_frame), 2}\t{round(self.center/self.total_frame), 2}\t{round(self.right/self.total_frame), 2}\t{round(self.close/self.total_frame), 2}")