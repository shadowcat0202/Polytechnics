import matplotlib.pyplot as plt
import cv2
import dlib
import numpy as np
import imutils
import time
from SY import *
from SY2 import *
from my_face_detector import *
from my_mark_detector import *
from pose_estimator import PoseEstimator
from my_tracker import *
from Preprocessing import *
from img_draw import imgMake
from Haar_Like import Haar_like
# import seaborn as sns
import scipy
###
def make_line_function(point1, point2):
    point1_x, point1_y = point1
    point2_x, point2_y = point2

    diff_Y = (point1_y - point2_y)
    diff_X = (point1_x - point2_x)
    intercept = (point1_x * point2_y - point2_x * point1_y) * (-1)

    return diff_Y, diff_X, intercept


def get_point_ofIntersection(line1, line2):
    line1_point1, line1_point2, line1_intercept = line1
    line2_point1, line2_point2, line2_intercept = line2

    D = line1_point1 * line2_point2 - line1_point2 * line2_point1
    Dx = line1_intercept * line2_point2 - line1_point2 * line2_intercept
    Dy = line1_point1 * line2_intercept - line1_intercept * line2_point1
    print(f"함수 d, dx, dy = {D}, {Dx}, {Dy}")

    x = Dx / D
    y = Dy / D
    return (x, y)


###
def eye_crop(img, eye_landmark):
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
        border: 잘라낸 눈의 영역의 [(p1.x, p1.y), (p2.x, p2.y)]
    """
    W = [i[0] for i in eye_landmark]
    H = [i[1] for i in eye_landmark]

    print(W)
    print(H)

    W_min, W_max = min(W), max(W)
    H_min, H_max = min(H), max(H)

    result = img[H_min:H_max, W_min:W_max]
    result = np.expand_dims(result, axis=-1)
    return result


print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
SKYBLUE = (255, 255, 0)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

PIXELS = [(1280, 720), (640, 480), (256, 144), (320, 240), (480, 360)]
PIXEL_NUMBER = 1
RES_W, RES_H = PIXELS[PIXEL_NUMBER][0], PIXELS[PIXEL_NUMBER][1]

FaceDetector = FaceDetector()  # 얼굴 인식 관련
MarkDetector = MarkDetector(
    save_model="C:/Dropbox/PC/Documents/Pupil_Master/assets/shape_predictor_68_face_landmarks.dat")  # 랜드마크 관련
# cv2.matchTemplate()도 해보자
Tracker = Tracker()  # 트래킹 관련
haar_like = Haar_like()
iMake = imgMake()
# eye = Eye

crop = None
cap = None
### SY
###

try:
    # 카메라 or 영상
    # cap = cv2.VideoCapture("05-5.mp4")
    # cap = cv2.VideoCapture("01-5.mp4")
    cap = cv2.VideoCapture("WIN_20220608_17_11_21_Pro.mp4")
    # cap = cv2.VideoCapture(1)
    # cap = cv2.VideoCapture("D:/JEON/dataset/haar-like/WIN_20220601_15_48_25_Pro.mp4")
    # cap = cv2.VideoCapture("D:/JEON/Polytechnics/Project/DMS/dataset/WIN_20220526_15_33_19_Pro.mp4")
    # cap = cv2.VideoCapture("D:/JEON/dataset/look_direction/vid/5/04-5.mp4")


except:
    print("Error opening video stream or file")
###
pupil_x_lftE = []
pupil_y_lftE = []
pupil_x_rytE = []
pupil_y_rytE = []
contour_Ratio = []
prediction_Result = []
cnt_frame = 0
###
while cap.isOpened():
    perv_time = time.time()
    ret, frame = cap.read()
    # print("loop started")
    if ret:
        view = frame

        frame =img_gray_Preprocessing(frame)

        if frame is None:   break

        if Tracker.track_number == 0:
            Tracker.find_tracker(frame, FaceDetector)
        else:  # 트레킹 할 얼굴이 1명 이상 있다면(지금은 1명만 트레킹 하도록 작성함)
            box_rect = None
            if Tracker.frame_counter == 60:  # 60 프레임마다 한번씩 트래커 이탈 방지용 refaceDetection
                box_rect = Tracker.find_tracker(frame, FaceDetector, re=True)
            else:
                box_rect = Tracker.tracking(frame)  # 네모 박스 처준다 원한다면 rectangle타입 반환도 가능

            if box_rect is not None:
                # 랜드마크 type=full_object_detection --> .part().x, .part().x 형식으로 뽑아내기
                landmarks = MarkDetector.get_marks(frame, box_rect)
                # MarkDetector.draw_marks(frame, landmarks, color=GREEN)  # 랜드마크 점 그려주기
                landmarks = MarkDetector.full_object_detection_to_ndarray(landmarks)  # 랜드마크 68개의 (x,y) 좌표

                if landmarks is not None or landmarks == ():
                    ldmk_lftE = landmarks[36:42]
                    ldmk_rytE = landmarks[42:48]

                    lftE = FindPupil(view, frame, ldmk_lftE)
                    # img_thold = lftE.img_eye_threshold()

                    # img_thold = lftE.img_eye_threshold()
                    img_prcd, thold = lftE.img_eye_processed()
                    # img_prcd, thold = lftE.img_eye_processed_test()

                    # v = 5
                    # img = cv2.erode(img_thold, (v,v))
                    # img = cv2.dilate(img, (v,v))

                    # v = 11
                    #
                    # thold1 = cv2.blur(img_thold, (v,v))
                    # thold2 = cv2.medianBlur(img_thold, v)
                    # thold3 = cv2.GaussianBlur(img_thold, (v,v),1)
                    # thold4 = cv2.medianBlur(img_thold, 1, (v,v))

                    # blobs = lftE.blob_detection()
                    # im_with_keypoints = cv2.drawKeypoints(img_thold, blobs, np.array([]), (0, 0, 255),
                    #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



                    # eye = lftE.img_eye_cropped()
                    # img_eye = lftE.img_eye_processed()

                    # img_thold = lftE.img_eye_threshold()
                    # show_cv_img(eye)
                    # show_cv_img(img_eye)
                    # show_cv_img(img_thold)

                    # thold_test  = lftE. optimal_thres_test()


                    # cv2.calcHist(eye, [0], None, )
                    # show_cv_img(cv2.pyrUp(cv2.pyrUp(cv2.pyrUp(lftE.img_eye_cropped()))))
                    # show_cv_img(cv2.pyrUp(cv2.pyrUp(cv2.pyrUp(lftE.img_eye_processed()))))
                    # show_cv_img(cv2.pyrUp(cv2.pyrUp(cv2.pyrUp(lftE.img_eye_threshold()))))

                    # blob = lftE.blob_detection()
                    # for i in blob:
                    #     print(blob[i].pt[0], blob[i].pt[1], blob[i].size)


                    """
                    """
                    # lftE = PupilDetection(view, frame, ldmk_lftE)
                    # rytE = PupilDetection(view, frame, ldmk_rytE)
                    # x_lftPupl, y_lftPupl = lftE.loc_xy_pupils_center_in_original_image()
                    # x_rytPupl, y_rytPupl = rytE.loc_xy_pupils_center_in_original_image()
                    #
                    #
                    # prd_lftE = lftE.classify_look_direction()
                    # prd_rytE = rytE.classify_look_direction()
                    #
                    # result = final_prediction(prd_lftE, prd_rytE)
                    # print(f"F/L/R: [{result}] - {prd_lftE}, {prd_rytE}")
                    # prediction_Result.append(result)
                    #
                    #
                    # cv2.circle(view, (round(x_lftPupl), round(y_lftPupl)), round(lftE.hgt_wth_eye[0]/4), (0, 0, 255), -1)
                    # cv2.circle(view, (round(x_rytPupl), round(y_rytPupl)), round(rytE.hgt_wth_eye[0]/4), (0, 0, 255), -1)
                    #
                    # dvsnL_L, dvsnL_M, dvsnL_R = lftE.loc_x_dvsn
                    # dvsnR_L, dvsnR_M, dvsnR_R = rytE.loc_x_dvsn
                    # dvsnL_Y = lftE.loc_xy_int[1]
                    # dvsnR_Y = rytE.loc_xy_int[1]
                    #
                    # dvsnL_L, dvsnL_M, dvsnL_R = round(dvsnL_L), round(dvsnL_M), round(dvsnL_R)
                    # dvsnR_L, dvsnR_M, dvsnR_R = round(dvsnR_L), round(dvsnR_M), round(dvsnR_R)
                    #
                    # dvsnL_Y = round(dvsnL_Y)
                    # dvsnR_Y = round(dvsnR_Y)
                    # x_msg = landmarks[27][0]
                    # y_msg = landmarks[27][1]
                    #
                    # cv2.putText(view, f"{result}", (round(x_msg), round(y_msg)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
                    # cv2.putText(view, f"gaze direction", (round(x_msg), round(y_msg-50)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
                    # cv2.putText(view, f"L/R: {prd_lftE}/{prd_rytE}", (round(W_VIEW/2), round(H_VIEW/2+300)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
                    # cv2.circle(view, (dvsnL_L, dvsnL_Y), round(rytE.hgt_wth_eye[0]/8), (255, 0, 0), 1)
                    # cv2.circle(view, (dvsnL_M, dvsnL_Y), round(rytE.hgt_wth_eye[0]/8), (255, 0, 0), 1)
                    # cv2.circle(view, (dvsnL_R, dvsnL_Y), round(rytE.hgt_wth_eye[0]/8), (255, 0, 0), 1)
                    # cv2.circle(view, (dvsnR_L, dvsnR_Y), round(rytE.hgt_wth_eye[0]/8), (255, 0, 0), 1)
                    # cv2.circle(view, (dvsnR_M, dvsnR_Y), round(rytE.hgt_wth_eye[0]/8), (255, 0, 0), 1)
                    # cv2.circle(view, (dvsnR_R, dvsnR_Y), round(rytE.hgt_wth_eye[0]/8), (255, 0, 0), 1)
                    # lft = lftE.template_eye_preprocessed()


                    ### 눈감기 판단
                    # ratio = lftE.contour_ratio_wth_hgt()
                    # contour_Ratio.append(ratio)
                    """
                    """

                else:
                    print("랜드마크 없어서 예외처리가 나와야하지만 뭔가 이상하네요 에러 났을때 안나오네요")

            # cv2.putText(frame, f"fps:{int(1. / (time.time() - perv_time))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
            # cv2.imshow("blob", show_result)
            # cv2.imshow("1",lft)
            # cv2.imshow('show_frame', view)
            # cv2.imshow('template(L)', lftE.template_eye_preprocessed())
            # cv2.imshow('template(L-test)', lftE.template_eye_preprocessed_test())
            cnt_frame +=1

            print(cnt_frame)
            # cv2.imshow("1", cv2.pyrUp(cv2.pyrUp(lftE.img_eye_cropped())))
            cv2.imshow("processed", cv2.pyrUp(img_prcd))
            # cv2.imshow("blob", cv2.pyrUp(im_with_keypoints))
            cv2.imshow("orign", cv2.pyrUp(lftE.img_eye_cropped()))
            # cv2.imshow("org_eye", cv2.pyrUp(lftE.img_eye_cropped()))
            # cv2.imshow("1", cv2.pyrUp(cv2.pyrUp(lftE.img_eye_cropped())))
            # cv2.imshow("org", view)

    # if the `esc` key was pressed, break from the loop
    key = cv2.waitKey(1)
    if key == 27 or ret is False:
        break

cv2.destroyAllWindows()
cap.release()

pred_result = np.array(prediction_Result)
result_values, result_counts = np.unique(pred_result, return_counts=True)

print(dict(zip(result_values, result_counts)))
