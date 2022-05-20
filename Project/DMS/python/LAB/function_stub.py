from functools import wraps

import cv2
import numpy
import dlib
import time

lastsave = 0


class lab_test_class:
    def __init__(self):
        self.count = 0

    def inner_counter(self, func):
        def tmp(self, *args, **kwargs):
            self.count += 1
            global lastsave
            # print(f"during close: {time.time() - lastsave}")
            if time.time() - lastsave > 5:
                lastsave = time.time()
                tmp.count = 0
            return func(*args, **kwargs)
        tmp.count = 0
        return tmp

        # @wraps(func)
        # def tmp(*args, **kwargs):
        #     tmp.count += 1
        #     global lastsave
        #     # print(f"during close: {time.time() - lastsave}")
        #     if time.time() - lastsave > 5:
        #         lastsave = time.time()
        #         tmp.count = 0
        #     return func(*args, **kwargs)
        # tmp.count = 0
        # return tmp

    @inner_counter
    def inner(self, a):
        print(a, "inner")
