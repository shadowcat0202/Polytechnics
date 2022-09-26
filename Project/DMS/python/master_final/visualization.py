import cv2
import numpy as np
import imutils
import platform


def getRed():
    return (0, 0, 255)


def getGreen():
    return (0, 255, 0)


def getBlue(self):
    return (255, 0, 0)


def getYellow(self):
    return (0, 255, 255)


def getSkyblue(self):
    return (255, 255, 0)


def getPurple(self):
    return (255, 0, 255)


def getWhite(self):
    return (255, 255, 255)


def getBlack(self):
    return (0, 0, 0)

class Camera:

    # Resolution
    __PIXELS = [(1920, 1080), (1440, 980), (1280, 960), (1280, 720), (1040, 680), (900, 600),
                (640, 480)]  # 해상도 픽셀수가 커질수록 좋다(프레임 차이는 거의 없는거 같다)

    def __init__(self, path=0):
        self.PIXEL_NUMBER = 2
        self.RES_H = self.__PIXELS[self.PIXEL_NUMBER][1]
        self.RES_W = self.__PIXELS[self.PIXEL_NUMBER][0]
        # print(f"pixel set ({self.__PIXELS[self.PIXEL_NUMBER][1]},{self.__PIXELS[self.PIXEL_NUMBER][0]})")

        self.encoded_avi = None  # cv2.VideoWriter()
        self.recoding = False  # 녹화 중 flag
        self.rec_number = 1  # 영상 번호 할당
        self.rec_trigger = False
        self.cap = None
        try:
            self.cap = self.camara_init()
            if self.cap == -1:
                raise Exception('Error opening video stream or file')
        except Exception as e:
            print(e)
        # try:
        #     self.cap = cv2.VideoCapture(path)
        # except:
        #     print("Error opening video stream or file")
        self.fps = round(self.cap.get(cv2.CAP_PROP_FPS))

    def camara_init(self):
        print('camera ready, please wait ...')
        for camera_number in range(4):
            cap = cv2.VideoCapture(camera_number)
            if cap.isOpened():
                print(f'{camera_number}번 카메라 확인 중...')
                _, frame = cap.read()
                pixel_check = frame[0][0]
                for h in range(self.RES_H):
                    for w in range(self.RES_W):
                        if frame[h][w] is not pixel_check:
                            print(f'{camera_number} 셋팅!')
                            return cap
            print('카메라를 찾지 못했습니다.')

        return -1

    def getFrameResize2ndarray(self, frame):
        return np.array(imutils.resize(frame, width=self.RES_W, height=self.RES_H))

    def rec(self, _frame, _file_name, _rec_trigger=False):
        """
        영상 녹화
        :param _frame: 영상에 대한 프레임
        :param _file_name: 저장할 파일 이름 저장 횟수마다 저장 번호 부여
        :param _rec_trigger: 녹화 트리거 (key)
        :return: None
        """
        if self.recoding:  # 녹화 중인지 판단
            self.encoded_avi.write(_frame)  # 영상 녹화
            if _rec_trigger:  # 녹화중 버튼 ==> 녹화 종료
                self.encoded_avi.release()  # cv2.VideoWriter 해제함 으로서 최종 저장
                self.encoded_avi = None  # 메모리 포인터 제거
                self.recoding = False  # 녹화 종료 flag
                self.rec_number += 1  # 다음 녹화 영상 번호
                print('녹화 종료')
        else:
            if _rec_trigger:
                _file_name = f'{_file_name}_{self.rec_number}.avi'
                codec = 'DIVX'
                fourcc = cv2.VideoWriter_fourcc(*codec)
                self.encoded_avi = cv2.VideoWriter(_file_name, fourcc, self.fps, (self.RES_W, self.RES_H))  # TODO 카메라의 Resolution이 다르다고 저장 못하는 이유 찾기 (solution _frame의 해상도와 일치 하지 않았다)
                # print(f'self.encoded_avi = cv2.VideoWriter({_file_name}, cv2.VideoWriter_fourcc({codec}), {self.fps}, ({self.RES_W}, {self.RES_H}))')
                """
                cv2.VideoWriter(filename, fourcc, fps, frameSize, isColor=None) -> retval
                • filename : 비디오 파일 이름 (e.g. 'video.avi')
                • fourcc : fourcc (e.g. cv2.VideoWriter_fourcc(*'DIVX'))        
                • fps : 초당 프레임 수 (e.g. 30)        
                • frameSize : 프레임 크기. (width, height) 튜플.        
                • isColor : 컬러 영상이면 True, 그렇지않으면 False. 기본값은 True입니다.        
                • retval : cv2.VideoWriter 객체
                """
                self.recoding = True
                print(f'{_file_name} 녹화 시작')

    def rec_logo(self, _frame):
        _frame = cv2.circle(_frame, (50, 33), 12, getRed(), -1)
        _frame = cv2.putText(_frame, "REC", (65, 50),
                             cv2.FONT_HERSHEY_PLAIN,
                             3, getRed(), 3, cv2.LINE_AA)
        return _frame


