import cv2
import dlib
import numpy as np

class MarkDetector:
    __NOTHING = list(range(0, 0))

    __ALL = list(range(0, 68))

    __FACE_OUTLINE = list(range(0, 17))

    __LEFT_EYEBROW = list(range(17, 22))
    __RIGHT_EYEBROW = list(range(22, 27))

    __NOSE = list(range(27, 36))

    __LEFT_EYE = list(range(36, 42))
    __RIGHT_EYE = list(range(42, 48))

    __MOUTH_OUTLINE = list(range(48, 60))
    __MOUTH_INLINE = list(range(60, 68))

    __MARK_INDEX = __RIGHT_EYE + __LEFT_EYE + __MOUTH_INLINE

    def __init__(self, save_model="./assets/shape_predictor_68_face_landmarks.dat"):

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
        for i in self.__MARK_INDEX:
            cv2.circle(image, (marks.part(i).x, marks.part(i).y), 1, color, -1, cv2.LINE_AA)

    def draw_box(self, image, rect, box_color=(255, 255, 255)):
        """Draw square boxes on image"""
        for box in rect:
            cv2.rectangle(image,
                          (box[0], box[1]), (box[2], box[3]),
                          box_color, 3)

    def full_object_detection_to_ndarray(self, full_object):
        result = [[p.x, p.y] for p in full_object.parts()]
        result = np.array(result)
        return result
    
    def changeMarkIndex(self, key):
        # TODO: 랜드마크 보여주는거 변경하는거 뭐지? ㅎ
        if key == 1:
            self.__MARK_INDEX = self.__NOTHING
        elif key == 2:
            self.__MARK_INDEX = self.__LEFT_EYEBROW + self.__RIGHT_EYEBROW
        elif key == 3:
            self.__MARK_INDEX = self.__LEFT_EYE + self.__RIGHT_EYE
        elif key == 4:
            self.__MARK_INDEX = self.__NOSE
        elif key == 5:
            self.__MARK_INDEX = self.__MOUTH_INLINE + self.__MOUTH_OUTLINE
        elif key == 6:
            self.__MARK_INDEX = self.__FACE_OUTLINE

