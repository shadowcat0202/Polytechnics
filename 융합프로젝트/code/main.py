import toy_project as tp
import cv2

def img_cut():
    img = cv2.imread('./icon/Vecteezyanimalgeneralnl0421Icon_generated.jpg')
    img_size = [310, 220]
    img_reshape_size = (50, 50)

    p1 = [180, 90]
    for i in range(10):
        p2 = (p1[0] + img_size[0], p1[0] + img_size[1])
        roi_img = img[p1[1]:p2[1], p1[0]:p2[0]]
        cv2.imshow('img', roi_img)
        cv2.waitKey(0)
        roi_img = cv2.resize(roi_img, img_reshape_size)
        cv2.imwrite(f'./icon/icon_{i}.jpg', roi_img)
        p1[0] = p1[0] + (i * img_size[0] + 50)
        p1[1] = p1[1] + (i * img_size[1] + 50)


if __name__ == '__main__':
    tp.draw_turtle_frame()


    print('PyCharm')




