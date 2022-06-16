import dlib


class FaceDetector:
    def __init__(self):
        # HOG(Histogram of Oriented Gradients) 특성
        # dlib에 있는 얼굴 검출기 사용
        self.face_detector = dlib.get_frontal_face_detector()

    def front_detection(self, squares):
        print("front_detection")
        most_front_detection_index = 0
        max_size_area = 0
        for i, sq in enumerate(squares):
            curr_area = (sq.right() - sq.left()) * (sq.bottom() - sq.top())
            if curr_area > max_size_area:
                max_size_area = curr_area
                most_front_detection_index = i
        return most_front_detection_index

    def get_facebox_2dot_form(self, image, upscale=0):
        face_detect = self.face_detector(image, upscale)
        if face_detect:
            if len(face_detect) > 1:
                return face_detect[self.front_detection(face_detect)]
            else:
                return face_detect[0]
        return None
