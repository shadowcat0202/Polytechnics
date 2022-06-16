import pprint

import cv2
import numpy as np
import dlib


class Haar_like:
    def __init__(self):
        self.kernel_e = np.ones((5,5), np.uint8)  # TODO 9행 9열의 1으로 채워진 양수(0~255) array 생성
        self.kernel_d = np.ones((9, 9), np.uint8)
        self.pupil_brightness = 255
        # self.LEFT = [[(0, 1), (0, 1)], [(0, 1), (1, 0)], [(1, 0), (0, 1)], [(1, 0), (1, 0)]]
        # self.CENTER = [[(1, 2), (1, 2)], [(1, 2), (2, 1)], [(2, 1), (1, 2)], [(2, 1), (2, 1)]]
        # self.RIGHT = [[(2, 3), (2, 3)], [(2, 3), (3, 2)], [(3, 2), (2, 3)], [(3, 2), (3, 2)]]
        #
        # 4 등분
        # self.LEFT = [[0, 0], [0, 1], [1, 0], [0, 2]]
        # self.CENTER = [[1, 1], [1, 2], [2, 1], [2, 2]]
        # self.RIGHT = [[2, 3], [3, 3], [3, 2], [1, 3]]
        # 3등분
        self.LEFT = [[0, 0], [0, 1], [1, 0]]
        self.CENTER = [[1, 1], [2, 0],[0, 2]]
        self.RIGHT = [[2, 2], [2, 1], [1, 2]]

    def flatten_array_remove_item(self, array, itemToPop):
        array_flat = np.ndarray.flatten(array)
        array_toPop = np.array(itemToPop)
        array_refined = np.setdiff1d(array_flat, array_toPop)
        return array_refined

    def threshold(self, frame, quantile=0.5, maxValue=255, type=cv2.THRESH_BINARY_INV):
        if type == cv2.THRESH_BINARY_INV:
            self.pupil_brightness = 255
        elif type == cv2.THRESH_BINARY:
            self.pupil_brightness = 0
        mb = cv2.medianBlur(frame, 5)

        # frame_values = self.flatten_array_remove_item(mb, 0)
        frame_values = np.ndarray.flatten(mb)

        qt = np.quantile(frame_values, quantile)
        _, th = cv2.threshold(mb, round(qt), maxValue, type)
        # _, th = cv2.threshold(mb, 42, maxValue, type)
        th = cv2.erode(th, self.kernel_e, iterations=3)
        th = cv2.dilate(th, self.kernel_d, iterations=2)  # cv2.THRESH_BINARY

        # ed = cv2.erode(frame, self.kernel, 0)  # cv2.THRESH_BINARY_INV
        # gb = cv2.GaussianBlur(dl, (21, 7), 0)
        return th

    def shape_to_np(self, shape, dtype="int"):  # TODO 얼굴 랜드마크를 np.array로 만들어주는 함수
        """        
        :param shape: 랜드마크 전부
        :param dtype: 
        :return: 랜드마크 좌표 nparray 변경
        """
        # initialize the list of (x, y)-coordinates #TODO (x, y) 좌표를 넣을 리스트 초기화
        coords = np.zeros((68, 2), dtype=dtype)  # TODO 68행 2열의 0으로 채워진 int array 생성
        # loop over the 68 facial landmarks and convert them #TODO 68개의 얼굴 랜드마크를 돌면서 array를 채울꺼임
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)  # TODO 68개의 얼굴 랜드마크를 돌면서 array를 채우는중
        # return the list of (x, y)-coordinates
        return coords  # TODO (x, y) 좌표 리턴

    def eye_on_mask(self, mask, side, shape):
        """
        :param mask: 마스크
        :param side: 어느쪽 눈
        :param shape: 랜드마크 전부
        :return: 마스크
        """
        points = [shape[i] for i in side]  # TODO 왼쪽, 오른쪽눈에 속하는 랜드마크를 넣을 points list 만듬
        points = np.array(points, dtype=np.int32)  # TODO list를 int array로 바꿈
        # TODO 여러 점의 좌표를 이용해 채워진 블록 다각형을 그린다 (mask = img파일 / points = 좌표 점들(x,y) / 255 = 색상(white)
        mask = cv2.fillConvexPoly(mask, points, 255)
        return mask  # TODO 왼쪽 혹은 오른쪽눈 좌표를 받아 흰색으로 채운 다각형 리턴

    # def contouring(self, thresh, mid, img, right=False):  # TODO 동공에 빨간 원 그리는 함수
    #     # TODO 외각선 검출 (thresh = 입력 영상 /
    #     #  cv2.RETR_EXTERNAL = 외각선 검출 모드(계층 정보 x, 바깥 외곽선만 검출합니다. 단순히 리스트로 묶어줍니다.) /
    #     #  cv2.CHAIN_APPROX_NONE = 외각선 근사화 방법(윤곽점들의 모든 점을 반환합니다.)
    #     cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #     try:
    #         cnt = max(cnts, key=cv2.contourArea)  # TODO cv2.contourArea => 외각선이 감싸는 영역의 면적 반환
    #         M = cv2.moments(cnt)  # TODO cv2.moments => 이미지 모멘트를 계산하고 딕셔너리 형태로 리턴 ('m10', 'm00', 'm01'공간 모멘트)
    #         cx = int(M['m10'] / M['m00'])  # TODO 중심의 x좌표
    #         cy = int(M['m01'] / M['m00'])  # TODO 중심의 y좌표
    #         if right:  # TODO True값을 주면 cx에 mid 더하기
    #             cx += mid
    #         # TODO (img = img파일 / (cx, cy) = 원의 중심 좌표 / 4 = 원의 반지름 / (0, 0, 255) = 색상(red) / 2 = 선 두께)
    #         cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
    #     except:
    #         pass  # TODO try에서 오류가 발생하면 pass

    def get_pupil(self, gray, landmark):
        """
        :param gray: cv2.COLOR_BGR2GRAY로 변경한 이미지
        :param landmark: 랜드마크 리스트
        :return: 동공 검정으로 전처리한 이미지 (원본 이미지와 같은 shape)
        """
        try:
            if len(gray.shape) != 2:
                raise Exception(f"get_pupil(gray) parameter shape length expected 2, but get {len(gray.shape)}")
        except Exception as e:
            print("shape size error:", e)
            exit(1)

        if isinstance(landmark, dlib.full_object_detection):  # nparray가 아닌경우 변경
            landmark = self.shape_to_np(landmark)  # TODO 얼굴 랜드마크를 np.array로 만들어줌
        mask = np.zeros(gray.shape[:2], dtype=np.uint8)  # TODO 프레임이랑 똑같은 크기의 검정 틀을 만든다
        mask = self.eye_on_mask(mask, [36, 37, 38, 39, 40, 41], landmark)  # TODO 왼쪽눈 좌표를 받아 흰색으로 채운 다각형 리턴
        mask = self.eye_on_mask(mask, [42, 43, 44, 45, 46, 47], landmark)  # TODO 오른쪽눈 좌표를 받아 흰색으로 채운 다각형 리턴
        # TODO cv2.dilate => 이미지 변환(객체 외각 팽창) (mask = 입력영상 / kernel = 구조 요소 커널(9, 9) / 5 = 반복해서 몇번 실행할지)
        # mask = cv2.dilate(mask, self.kernel, 1)
        # TODO cv2.bitwise_and => 각 픽셀에 대해 AND연산, 프레임을 합쳐서 모두 흰곳만 흰곳으로 표현 (mask = 적용 영역 지정) /
        #  눈만 뽑아낸 mask를 합쳐 둘다 흰색으로 표현된 곳만 흰색으로 보임
        eyes = cv2.bitwise_and(gray, gray, mask=mask)

        # mask = (eyes == [0, 0, 0]).all(axis=2)
        # eyes[mask] = [255, 255, 255]
        mask = (eyes == 0).all()
        eyes[mask] = 255

        eyes = cv2.GaussianBlur(eyes, (9, 9), 0)

        _, thh = cv2.threshold(eyes, 60, 255, cv2.THRESH_BINARY)
        return thh
    def eye_pixel_count_3(self, eye):
        side_pixel_add_rat = 0.1
        pixel_count = [[0,0],[1,0],[2, 0]]
        H = eye.shape[0]
        l = eye.shape[1] // 3
        c = l * 2
        r = eye.shape[1]
        addition_pixel = round(H * l * side_pixel_add_rat)
        pixel_count[0][1] += addition_pixel
        pixel_count[2][1] += addition_pixel

        for i in range(H):
            for j in range(l):
                if eye[i][j] == self.pupil_brightness:
                    pixel_count[0][1] += 1

            for j in range(l, c):
                if eye[i][j] == self.pupil_brightness:
                    pixel_count[1][1] += 1

            for j in range(c, r):
                if eye[i][j] == self.pupil_brightness:
                    pixel_count[2][1] += 1

        return pixel_count


    def eye_pixel_count_4(self, eye):
        side_pixel_add_rat = 0.3
        pixel_count = [[0, 0], [1, 0], [2, 0], [3, 0]]

        H = eye.shape[0]
        ll = eye.shape[1] // 4
        l = ll * 2
        r = ll * 3
        rr = eye.shape[1]

        addition_pixel = round(H * ll * side_pixel_add_rat)
        pixel_count[0][1] += addition_pixel
        # pixel_count[1][1] += round(addition_pixel * 0.2)
        # pixel_count[2][1] += round(addition_pixel * 0.2)
        pixel_count[3][1] += addition_pixel
        # print(f"left.shape{left.shape}, {W_1}, {W_2}")
        for i in range(H):
            for j in range(ll):
                if eye[i][j] == self.pupil_brightness:
                    pixel_count[0][1] += 1

            for j in range(ll, l):
                if eye[i][j] == self.pupil_brightness:
                    pixel_count[1][1] += 1

            for j in range(l, r):
                if eye[i][j] == self.pupil_brightness:
                    pixel_count[2][1] += 1

            for j in range(r, rr):
                if eye[i][j] == self.pupil_brightness:
                    pixel_count[3][1] += 1

        return pixel_count


    def eye_dir_index(self, pixel_list):
        # 2 가지 모두 체크 경우의 수가 너무 많다   ==============================
        # sort_list = sorted(pixel_list, key=lambda x: -x[1])
        # idx = (sort_list[0][0], sort_list[1][0])

        # ================================================================
        sort_list = sorted(pixel_list, key=lambda x: -x[1])
        idx = sort_list[0][0]
        return idx


    def eye_dir(self, eyes):
        left_eye_pixel_list = self.eye_pixel_count_3(eyes[0])
        right_eye_pixel_list = self.eye_pixel_count_3(eyes[1])
        # print(f"left:{left_eye_pixel_list}, right:{right_eye_pixel_list}")
        check = [self.eye_dir_index(left_eye_pixel_list), self.eye_dir_index(right_eye_pixel_list)]
        # print(f"l:{l}, r:{r}")
        if check in self.LEFT:
            return "left"
        elif check in self.RIGHT:
            return "right"
        elif check in self.CENTER:
            return "center"
        else:
            return "None"
        # if (l == 0 and r == 0) or (l == 0 and r == 1) or (l == 1 and r == 0):
        #     return "left"
        # elif (l == 2 and r == 2) or (l == 1 and r == 2) or (l == 2 and r == 1):
        #     return "right"
        # else:
        #     return "center"
