import cv2
import numpy as np
import dlib


class Haar_like:
    def __init__(self):
        self.kernel = np.ones((9, 9), np.uint8)  # TODO 9행 9열의 1으로 채워진 양수(0~255) array 생성

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

    def get_pupil(self, img, rect, predictor, landmark):
        """
        :param landmark: 랜드마크 리스트
        :param img: 원본 이미지
        :param rect: face detection rectangle
        :param predictor: predictor()
        :return: 동공 검정으로 전처리한 이미지 (원본 이미지와 같은 shape)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = predictor(gray, rect)  # TODO 얼굴의 랜드마크 잡기
        shape = self.shape_to_np(shape)  # TODO 얼굴 랜드마크를 np.array로 만들어줌
        mask = np.zeros(img.shape[:2], dtype=np.uint8)  # TODO 프레임이랑 똑같은 크기의 검정 틀을 만든다
        mask = self.eye_on_mask(mask, [36, 37, 38, 39, 40, 41], landmark)  # TODO 왼쪽눈 좌표를 받아 흰색으로 채운 다각형 리턴
        mask = self.eye_on_mask(mask, [42, 43, 44, 45, 46, 47], landmark)  # TODO 오른쪽눈 좌표를 받아 흰색으로 채운 다각형 리턴
        # TODO cv2.dilate => 이미지 변환(객체 외각 팽창) (mask = 입력영상 / kernel = 구조 요소 커널(9, 9) / 5 = 반복해서 몇번 실행할지)
        mask = cv2.dilate(mask, self.kernel, 5)
        # TODO cv2.bitwise_and => 각 픽셀에 대해 AND연산, 프레임을 합쳐서 모두 흰곳만 흰곳으로 표현 (mask = 적용 영역 지정) /
        #  눈만 뽑아낸 mask를 합쳐 둘다 흰색으로 표현된 곳만 흰색으로 보임
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        eyes = cv2.GaussianBlur(eyes, (9, 9), 0)
        _, threshold = cv2.threshold(cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY), 50, 255, cv2.THRESH_BINARY)
        return threshold
