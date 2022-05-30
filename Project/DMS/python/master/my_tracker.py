import cv2
import dlib


class Tracker:
    def __init__(self):
        self.frame_counter = 0
        self.track_number = 0
        self.tracker = None

    def find_tracker(self, img, fd, re=False):
        box = fd.get_facebox_2dot_form(img)     # FaceDetector
        tk = dlib.correlation_tracker()    # correlation_traker타입 객채 생성
        if box is not None:
            if re:
                self.frame_counter = 0
                tk.start_track(img, box)
                self.tracker = tk
            else:
                # rect = dlib.rectangle(box.left(), box.top(), box.right(), box.bottom())
                tk.start_track(img, box)  # 얼굴 감지한 네모를 트래킹 하기 시작
                self.tracker = tk  # 트래킹 할 정보 추가
                self.track_number += 1
            return self.dlib_corr_tracker_to_rectangle(self.tracker)

    def tracking(self, img):
        self.frame_counter += 1
        self.tracker.update(img)    #트래킹 갱신
        # 여기서 tracker는 find_tracker에서 correlation_tracker 타입으로 append되었기 때문에
        # rectangle 타입으로 바꿔서 넘겨 주어야 한다
        rect = self.dlib_corr_tracker_to_rectangle(self.tracker)
        # self.draw_rectangle(img, rect)
        return rect

    def dlib_corr_tracker_to_rectangle(self, corr):
        pos = corr.get_position()
        rect = dlib.rectangle(int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom()))
        return rect

    def draw_rectangle(self, frame, tracker, color=(0, 255, 0)):
        cv2.rectangle(frame, (tracker.left(), tracker.top()),
                      (tracker.right(), tracker.bottom()), color, 3)
