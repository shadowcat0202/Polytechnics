import cv2
import dlib


class FaceDetector:
    def __init__(self):
        # HOG(Histogram of Oriented Gradients) 특성
        # dlib에 있는 얼굴 검출기 사용
        self.face_detector = dlib.get_frontal_face_detector()

    def front_detection(self, squares):
        most_front_detection_index = 0
        max_size_area = 0
        for i, sq in enumerate(squares):
            curr_area = (sq.right() - sq.left()) * (sq.bottom() - sq.top())
            if curr_area > max_size_area:
                max_size_area = curr_area
                most_front_detection_index = i
        return most_front_detection_index

    def get_faceboxes(self, image, upscale=0):
        face_detect = self.face_detector(image, upscale)
        if face_detect:
            if len(face_detect) > 1:
                return face_detect[self.front_detection(face_detect)]
            else:
                return face_detect[0]
        else:
            return None


class MarkDetector:
    def __init__(self, save_model="assets/pose_model"):
        print(f"stub loading facial landmark predictor {save_model}...")
        self.shape_predictor = dlib.shape_predictor(save_model)
        print(f"complete loading facial landmark predictor!")

    # 랜드마크를 리스트타입으로 반환
    def get_marks(self, image, detect):
        # print(type(detect)) # <class '_dlib_pybind11.rectangles'> 인데?
        # detect인자는 <class _dlib_pybind11.full_object_detection> 타입으로 입력해야함 이라고 오류 나는데
        # 다른 파일에서는 <class '_dlib_pybind11.rectangle'> 로 shape_predictor가능한데? 뭐지? 날 화나게 하는건가?
        # 와 코드랑 싸울뻔 했다 (결론 본인좌 멍청 했던 걸로)
        shape = self.shape_predictor(image, detect)
        return list([p.x, p.y] for p in shape.parts())

    @staticmethod
    def draw_marks(image, marks, color=(225, 255, 255)):
        for mark in marks:
            cv2.circle(image, (int(mark[0]), (int(mark[1]))), 1, color, -1, cv2.LINE_AA)

    @staticmethod
    def draw_box(image, boxes, box_color=(255, 255, 255)):
        """Draw square boxes on image"""
        for box in boxes:
            cv2.rectangle(image,
                          (box[0], box[1]),
                          (box[2], box[3]), box_color, 3)
