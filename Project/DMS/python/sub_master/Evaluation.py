class evaluation:
    def __init__(self):
        self.total_frame = 0
        self.left = 0
        self.center = 0
        self.right = 0
        self.close = 0

    def measurement(self, pred):
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
        print(f"=================== acc ============================")
        print(f"left:{round(self.left/self.total_frame,2) * 100}%\n"
              f"center:{round(self.center/self.total_frame, 2) * 100}%\n"
              f"right:{round(self.right/self.total_frame, 2) * 100}%\n"
              f"none:{round(self.close/self.total_frame, 2) * 100}%")