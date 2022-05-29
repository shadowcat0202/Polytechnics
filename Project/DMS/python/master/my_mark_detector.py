import cv2
import dlib
import numpy as np

NOTHING = list(range(0, 0))

ALL = list(range(0, 68))

FACE_OUTLINE = list(range(0, 17))

LEFT_EYEBROW = list(range(17, 22))
RIGHT_EYEBROW = list(range(22, 27))

NOSE = list(range(27, 36))

LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INLINE = list(range(60, 68))

MARK_INDEX = RIGHT_EYE + LEFT_EYE + MOUTH_INLINE


class MarkDetector:
    def __init__(self, save_model="../assets/"):

        print(f"stub loading facial landmark predictor {save_model}...")
        self.shape_predictor = dlib.shape_predictor(save_model)
        print(f"complete loading facial landmark predictor!")

    def get_marks(self, image, detect):
        # print(type(detect)) # <class '_dlib_pybind11.rectangles'> 인데?
        # detect인자는 <class _dlib_pybind11.full_object_detection> 타입으로 입력해야함 이라고 오류 나는데
        # 다른 파일에서는 <class '_dlib_pybind11.rectangle'> 로 shape_predictor가능한데? 뭐지? 날 화나게 하는건가?
        # 와 코드랑 싸울뻔 했다 (결론 본인이 멍청 했던 걸로)
        shape = self.shape_predictor(image, detect)
        return shape

    def draw_marks(self, image, marks, color=(225, 255, 255)):
        for i in MARK_INDEX:
            cv2.circle(image, (marks.part(i).x, marks.part(i).y), 1, color, -1, cv2.LINE_AA)

    def draw_box(self, image, boxes, box_color=(255, 255, 255)):
        """Draw square boxes on image"""
        for box in boxes:
            cv2.rectangle(image,
                          (box[0], box[1]), (box[2], box[3]),
                          box_color, 3)

    def full_object_detection_to_ndarray(self, full_object):
        result = [[p.x, p.y] for p in full_object.parts()]
        result = np.array(result)
        return result
