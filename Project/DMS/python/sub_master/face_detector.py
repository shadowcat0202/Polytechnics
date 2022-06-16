import cv2
import dlib
import numpy as np


class FaceDetector:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.image_preprocessing = None
        self.detection_result = None

    def preprocessing(self, image, size=(400, 400)):
        image = cv2.resize(image, size, cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0,
                                tileGridSize=(8, 8))
        image = clahe.apply(image)
        self.image_preprocessing = self.face_detector(image, 0)

    def get_faceboxes(self, image):
        rows, cols, _ = image.shape
        faceboxes = []

        detection = self.face_detector(image, 0)
        for result in detection:
            x_left_top = result.left()
            y_left_top = result.top()
            x_right_bottom = result.right()
            y_right_bottom = result.bottom()
            faceboxes.append([x_left_top, y_left_top, x_right_bottom, y_right_bottom])
        self.detection_result = faceboxes

    def draw_all_result(self, image, color=(0,255,0)):
        for facebox in self.detection_result:
            cv2.rectangle(image, (facebox[0], facebox[1]), (facebox[2],facebox[3]), color)


class MarkDetector:
    def __init__(self, model="assets/shape_predictor_68_face_landmarks.dat"):
        self.shape_predictor = dlib.shape_predictor(model)
        self.f_detector = FaceDetector()

    @staticmethod
    def draw_box(image, boxes, box_color=(255, 255, 255)):
        for box in boxes:
            cv2.rectangle(image,
                          (box[0], box[1]),
                          (box[2], box[3]), box_color, 3)

    @staticmethod
    def get_square_box(box):
        """Get a square box out of the given box, by expanding it."""
        left_x = box[0]
        top_y = box[1]
        right_x = box[2]
        bottom_y = box[3]

        box_width = right_x - left_x
        box_height = bottom_y - top_y

        # Check if box is already a square. If not, make it a square.
        diff = box_height - box_width
        delta = int(abs(diff) / 2)

        if diff == 0:  # Already a square.
            return box
        elif diff > 0:  # Height > width, a slim box.
            left_x -= delta
            right_x += delta
            if diff % 2 == 1:
                right_x += 1
        else:  # Width > height, a short box.
            top_y -= delta
            bottom_y += delta
            if diff % 2 == 1:
                bottom_y += 1

        # Make sure box is always square.
        assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def box_in_image(box, image):
        """Check if the box is in image"""
        rows = image.shape[0]
        cols = image.shape[1]
        return box[0] >= 0 and box[1] >= 0 and box[2] <= cols and box[3] <= rows

    def detect_marks(self, image):
        """Detect facial marks from an face image.

        Args:
            image: a face image.

        Returns:
            marks: the facial marks as a numpy array of shape [N, 2].
        """
        # Resize the image into fix size.
        image = cv2.resize(image, (128, 128))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0,
                                tileGridSize=(8, 8))
        image = clahe.apply(image)
        detection = self.face_detector(image, 0)
        # inputs = tf.expand_dims(image, axis=0)

        # Actual detection.
        marks = self.shape_predictor(image, 0)

        # Convert predictions to landmarks.
        marks = np.reshape(marks, (-1, 2))

        return marks

    @staticmethod
    def draw_marks(image, marks, color=(255, 255, 255)):
        """Draw mark points on image"""
        for mark in marks:
            cv2.circle(image, (int(mark[0]), int(
                mark[1])), 1, color, -1, cv2.LINE_AA)
