import cv2
import numpy as np
import imutils


class Camera:
    # Color
    __RED = (0, 0, 255)
    __GREEN = (0, 255, 0)
    __BLUE = (255, 0, 0)
    __YELLOW = (0, 255, 255)
    __SKYBLUE = (255, 255, 0)
    __PURPLE = (255, 0, 255)
    __WHITE = (255, 255, 255)
    __BLACK = (0, 0, 0)

    # Resolution
    __PIXELS = [(1920, 1080), (1440, 980),(1280, 720), (1040, 680), (900, 600), (640, 480)] # 해상도 픽셀수가 커질수록 좋다(프레임 차이는 거의 없는거 같다)

    def __init__(self, path=0):
        self.PIXEL_NUMBER = 2
        self.RES_W = self.__PIXELS[self.PIXEL_NUMBER][0]
        self.RES_H = self.__PIXELS[self.PIXEL_NUMBER][1]
        print(f"pixel set ({self.__PIXELS[self.PIXEL_NUMBER][1]},{self.__PIXELS[self.PIXEL_NUMBER][0]})")
        try:
            self.cap = cv2.VideoCapture(path)
        except:
            print("Error opening video stream or file")

    def getFrameResize2ndarray(self, frame):
        return np.array(imutils.resize(frame, width=self.RES_W, height=self.RES_H))

    def getFrame2array(self, frame):
        pass

    def setPixels(self, resolution):
        print(f"H:{self.__PIXELS[resolution][1]} W{self.__PIXELS[resolution][0]}")
        return self.__PIXELS[resolution]

    def getRed(self):
        return self.__RED

    def getGreen(self):
        return self.__GREEN

    def getBlue(self):
        return self.__BLUE

    def getYellow(self):
        return self.__YELLOW

    def getSkyblue(self):
        return self.__SKYBLUE

    def getPurple(self):
        return self.__PURPLE

    def getWhite(self):
        return self.__WHITE

    def getBlack(self):
        return self.__BLACK
