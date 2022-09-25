import cv2


def rec_logo(self, _frame):
    _frame = cv2.circle(_frame, (50, 33), 12, self.__RED, -1)
    _frame = cv2.putText(_frame, "REC", (65, 50),
                         cv2.FONT_HERSHEY_PLAIN,
                         3, self.__RED, 3, cv2.LINE_AA)


if __name__ == "__main__":
    encoded_avi = None
    recoding = False
    rec_trigger = False
    file_name = 'test_video'
    fps = None
    RES_W = None
    RES_H = None

    vid = cv2.VideoCapture(0)

    if vid.isOpened():
        fps = vid.get(cv2.CAP_PROP_FPS)
        RES_W = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        RES_H = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while vid.isOpened():
        ret, frame = vid.read()
        if ret:
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                break
            if key == ord('r'):
                rec_trigger = True

            if recoding:
                encoded_avi.write(frame)
                if rec_trigger:
                    encoded_avi.release()
                    encoded_avi = None
                    recoding = False
                    print('녹화 종료')
            else:
                if rec_trigger:
                    if not ('.avi' in file_name):
                        file_name = f'{file_name}.avi'
                    """
                    cv2.VideoWriter(filename, fourcc, fps, frameSize, isColor=None) -> retval
                    • filename : 비디오 파일 이름 (e.g. 'video.mp4')
                    • fourcc : fourcc (e.g. cv2.VideoWriter_fourcc(*'DIVX'))        
                    • fps : 초당 프레임 수 (e.g. 30)        
                    • frameSize : 프레임 크기. (width, height) 튜플.        
                    • isColor : 컬러 영상이면 True, 그렇지않으면 False. 기본값은 True입니다.        
                    • retval : cv2.VideoWriter 객체
                    """
                    codec = "DIVX"
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    encoded_avi = cv2.VideoWriter(file_name, fourcc, fps, (RES_W, RES_H))
                    recoding = True
                    print(f'{file_name} 녹화 시작')
            cv2.imshow("img", frame)
            rec_trigger = False

        else:
            break
    vid.release()
    cv2.destroyAllWindows()
