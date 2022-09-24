import cv2
import numpy as np
import imutils
import platform


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
    __PIXELS = [(1920, 1080), (1440, 980), (1280, 720), (1040, 680), (900, 600),
                (640, 480)]  # 해상도 픽셀수가 커질수록 좋다(프레임 차이는 거의 없는거 같다)

    def __init__(self, path=0):
        self.PIXEL_NUMBER = 2
        self.RES_W = self.__PIXELS[self.PIXEL_NUMBER][0]
        self.RES_H = self.__PIXELS[self.PIXEL_NUMBER][1]
        print(f"pixel set ({self.__PIXELS[self.PIXEL_NUMBER][1]},{self.__PIXELS[self.PIXEL_NUMBER][0]})")
        self.rec_out = None
        self.recoding = False
        self.rec_number = 0
        """
        self.cap = None
        cap_num = 0
        while True:
            self.cap = cv2.VideoCapture(cap_num)
            if self.cap.isOpened():
                break
            else:
                cap_num += 1
                try:
                    if cap_num > 4:
                        raise Exception('please check Camera!')
                except Exception as e:
                    print('err', e)
        """

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

    def rec(self, _frame, _file_name, _rec_trigger):
        """
        영상 녹화
        :param _frame: 영상에 대한 프레임
        :param _file_name: 저장할 파일 이름 저장 횟수마다 저장 번호 부여
        :param _rec_trigger: 녹화 트리거 (key)
        :return: None
        """
        if self.recoding:
            self.rec_out.write(_frame)
            if _rec_trigger:
                self.rec_out.release()
                self.rec_out = None
                self.recoding = False
        else:
            if _rec_trigger:
                if not ('.mp4' in _file_name):
                    _file_name = f'{_file_name}_{self.rec_number}.avi'
                """
                cv2.VideoWriter(filename, fourcc, fps, frameSize, isColor=None) -> retval
                • filename : 비디오 파일 이름 (e.g. 'video.mp4')
                • fourcc : fourcc (e.g. cv2.VideoWriter_fourcc(*'DIVX'))        
                • fps : 초당 프레임 수 (e.g. 30)        
                • frameSize : 프레임 크기. (width, height) 튜플.        
                • isColor : 컬러 영상이면 True, 그렇지않으면 False. 기본값은 True입니다.        
                • retval : cv2.VideoWriter 객체
                """
                if platform.system() == 'Windows':
                    self.rec_out = cv2.VideoWriter(f'./{_file_name}', cv2.VideoWriter_fourcc(*'DIVX'), 30,
                                                   (self.RES_W, self.RES_H))
                self.recoding = True

    def rec_logo(self, _frame):
        _frame = cv2.circle(_frame, (50, 33), 12, self.__RED, -1)
        _frame = cv2.putText(_frame, "REC", (65, 50),
                             cv2.FONT_HERSHEY_PLAIN,
                             3, self.__RED, 3, cv2.LINE_AA)
        return _frame

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
