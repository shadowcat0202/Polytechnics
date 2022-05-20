import cv2
import dlib


class FaceDetector:
    def __init__(self):
        # HOG(Histogram of Oriented Gradients) 특성
        # dlib에 있는 얼굴 검출기 사용
        self.face_detector = dlib.get_frontal_face_detector()

    def front_detection(self, squares):
        most_front_face_index = 0
        max_size_area = 0
        for i, sq in enumerate(squares):
            curr_area = (sq.right() - sq.left()) * (sq.bottom() - sq.top())
            if curr_area > max_size_area:
                max_size_area = curr_area
                most_front_face_index = i
        return most_front_face_index

    def get_faceboxes(self, image, upscale=0):
        face_detect = self.face_detector(image, upscale)
        if face_detect:
            if len(face_detect) > 1:
                return face_detect[self.front_detection(face_detect)]
            else:
                return face_detect
        else:
            return None

    # def draw_all_faceboxes(self, image, boxes):
    #     for facebox in boxes:
    #         cv2.rectangle(image, ())


class MarkDetector:
    def __init__(self, save_model="./assets/shape_predictor_68_face_landmarks.dat"):
        self.face_detector = FaceDetector()
        self.shape_predictor = dlib.shape_predictor(save_model)
        print("stub loading facial landmark predictor...")

    # 랜드마크를 리스트타입으로 반환
    def get_marks(self, image, detect):
        print(type(detect))
        shape = self.shape_predictor(image, detect)
        return list([p.x, p.y] for p in shape.parts())

    def set_INDEX(self, i):
        ALL = list(range(0, 68))
        RIGHT_EYEBROW = list(range(17, 22))
        LEFT_EYEBROW = list(range(22, 27))
        RIGHT_EYE = list(range(36, 42))
        LEFT_EYE = list(range(42, 48))
        NOSE = list(range(27, 36))
        MOUTH_OUTLINE = list(range(48, 61))
        MOUTH_INLINE = list(range(51, 68))
        FACE_OUTLINE = list(range(0, 17))
        NOTHING = list(range(0, 0))

    @staticmethod
    def draw_marks(image, marks, color=(225, 255, 255)):
        for mark in marks:
            cv2.circle(image, (int(mark[0]) ,(int(mark[1]))), 1, color, -1, cv2.LINE_AA)

    @staticmethod
    def draw_box(image, boxes, box_color=(255, 255, 255)):
        """Draw square boxes on image"""
        for box in boxes:
            cv2.rectangle(image,
                          (box[0], box[1]),
                          (box[2], box[3]), box_color, 3)
    # def draw_all_rectangle(self, image):
