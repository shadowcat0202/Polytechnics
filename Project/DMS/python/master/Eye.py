import cv2
import numpy as np


def eye_crop_border(img, eye_landmark):
    """
    :argument
        img: 한 프레임(이미지)
        eye_landmark: 눈의 랜드마크 좌표
    :return
        result: img에서 눈 영역의 이미지를 잘라서 반환
        border: 잘라낸 눈의 영역의 [(p1.x, p1.y), (p2.x, p2.y)]
    """
    W = [i[0] for i in eye_landmark]
    H = [i[1] for i in eye_landmark]

    W_min, W_max = min(W), max(W)
    H_min, H_max = min(H), max(H)

    W_btw = W_max - W_min
    H_btw = H_max - H_min

    W_border_size = W_btw * 0.2
    H_border_size = H_btw * 0.5

    p1_W_border = int(W_min - W_border_size)
    p1_H_border = int(H_min - H_border_size)
    p2_W_border = int(W_max + W_border_size)
    p2_H_border = int(H_max + H_border_size)

    border = [
        (p1_W_border, p1_H_border),
        (p2_W_border, p2_H_border)
    ]
    result = np.array(img[p1_H_border:p2_H_border, p1_W_border:p2_W_border])
    return result, border


def eye_crop_none_border(img, eye_landmark):
    """
    :argument
        img: 한 프레임(이미지)
        eye_landmark: 눈의 랜드마크 좌표
    :return
        result: img에서 눈 영역의 이미지를 잘라서 반환
    """
    W = [i[0] for i in eye_landmark]
    H = [i[1] for i in eye_landmark]

    W_min, W_max = min(W), max(W)
    H_min, H_max = min(H), max(H)

    result = img[H_min:H_max, W_min:W_max]
    result = np.expand_dims(result, axis=-1)
    return result


def eye_crop_border_to_center(img, eye_landmark):
    W = [i[0] for i in eye_landmark]
    H = [i[1] for i in eye_landmark]

    W_min, W_max = min(W), max(W)
    H_min, H_max = min(H), max(H)

    H_btw = H_max - H_min
    W_btw = W_max - W_min

    center = (round(H_btw / 2), round(W_btw / 2))

    H_border = round(H_btw * 0.5)
    W_border = round(W_btw * 0.2)

    result = img[H_min - H_border:H_max + H_border, W_min - W_border:W_max + W_border]
    result = np.expand_dims(result, axis=-1)
    return result


class HaarCascadeBlobCapture:
    def __init__(self):
        self.previous_blob_area = [1, 1]
        self.previous_keypoints = [None, None]
        self.blob_detector = None

    def set_SimpleBlod_params(self, H, W, typ):
        detector_params = cv2.SimpleBlobDetector_Params()

        detector_params.filterByArea = True
        detector_params.minArea = round(H * W * 0.2)

        detector_params.filterByColor = True

        detector_params.blobColor = (255 if typ == cv2.THRESH_BINARY_INV else 0)

        # detector_params.maxArea = 1500

        self.blob_detector = cv2.SimpleBlobDetector_create(detector_params)

    def img_eye_processed(self, img):
        img = cv2.pyrUp(img)
        thold = np.min(img) + np.std(img)
        img = cv2.medianBlur(img, 3, 3)
        img = np.where(img < thold, 255, 0).astype("uint8")
        img = cv2.erode(img, None, iterations=3)
        img = cv2.dilate(img, None, iterations=1)
        return img

    def blob_track(self, img, prev_area=None, quantile=0.3, typ=cv2.THRESH_BINARY_INV):
        if img is None:
            return None
        # original ===================================================================
        # img = cv2.medianBlur(img, 5)
        # # img = cv2.bilateralFilter(img, 9, 75, 75)
        # # img = cv2.fastNlMeansDenoising(img, None, 15, 15, 5)
        #
        # flat = np.ndarray.flatten(img)
        # threshold = np.quantile(flat, quantile)
        # _, img = cv2.threshold(img, threshold, 255, typ)
        # cv2.imshow("th", img)
        # img = search(img)
        # cv2.imshow("bfs:", img)
        # plt.hist(flat)
        # plt.show()
        # img = cv2.erode(img, None, iterations=5)
        # # img = cv2.dilate(img, np.ones((5, 5), np.uint8), iterations=7)
        # img = cv2.dilate(img, None, iterations=5)

        # 시영씨 코드 =============================================================================
        img = self.img_eye_processed(img)
        cv2.imshow("t_e_d_mB", img)
        self.set_SimpleBlod_params(img.shape[0], img.shape[1], typ)
        keypoints = self.blob_detector.detect(img)
        # =============================================================================

        ans = None
        if keypoints and len(keypoints) > 1:
            tmp = 1000
            for keypoint in keypoints:  # filter out odd blobs
                if abs(keypoint.size - prev_area) < tmp:
                    ans = keypoint
                    tmp = abs(keypoint.size - prev_area)

            keypoints = (ans,)
        return keypoints

    def draw(self, img, keypoints, dest=None):
        if dest is None:
            dest = img
        if img is None:
            return None
        d = cv2.drawKeypoints(
            img,
            keypoints,
            dest,
            (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        return d

    def eye_direction_process(self, img, mark):
        try:
            if len(img.shape) != 2:
                raise Exception(f"get_pupil(gray) parameter shape length expected 2, but get {len(img.shape)}")
            if type(mark) == np.array:
                raise Exception(f"landmark should be ndarray, but get {type(mark)}")
        except Exception as e:
            print(e)
            exit(1)

        eyes = [eye_crop_none_border(img, mark[36:42]), eye_crop_none_border(img, mark[42:48])]
        eye_looking = ["", ""]
        for i in range(len(eyes)):
            key_points = self.blob_track(eyes[i], self.previous_blob_area[i])
            self.previous_keypoints[i] = key_points or self.previous_keypoints[i]
            if self.previous_keypoints[i] is not None:
                print(f"({self.previous_keypoints[i][0].pt[0]}, {self.previous_keypoints[i][0].pt[1]})")
                d =self.draw(eyes[i], self.previous_keypoints[i])
                cv2.imshow("left" if i == 0 else "right", d)
                look_percent = self.previous_keypoints[i][0].pt[0] / eyes[i].shape[1]
                if look_percent < 0.35:
                    eye_looking[i] = "left"
                elif look_percent > 0.65:
                    eye_looking[i] = "right"
                else:
                    eye_looking[i] = "center"

        if eye_looking[0] == "left" or eye_looking[1] == "left":
            return "left"
        elif eye_looking[1] == "right" or  eye_looking[0] == "right":
            return "right"
        else:
            return "center"
