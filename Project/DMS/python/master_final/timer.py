import time

class Timer():
    def __init__(self, isDanger):
        self.on = isDanger
        self.sp = time.time()

    def