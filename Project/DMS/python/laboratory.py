from argparse import ArgumentParser

import cv2
from face_detector import MarkDetector
from face_detector import FaceDetector

print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))


parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
args = parser.parse_args()

if __name__ == '__main__':
    video_src = args.cam if args.cam is not None else args.video
    if video_src is None:
        print("video default webcam will be used.")
        video_src = 0

    cap = cv2.VideoCapture(video_src)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    face_detector = FaceDetector()
    mark_detector = MarkDetector()

    while True:
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        if video_src == 0:
            frame = cv2.flip(frame, 2)

        face_detector.preprocessing(frame)

