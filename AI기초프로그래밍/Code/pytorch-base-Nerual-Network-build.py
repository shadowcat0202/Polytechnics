import pprint
import time

import torch
from PIL import Image
import numpy as np
import cv2
from img_draw import imgMake



def d20220513():
    a = [15, 17, 19.2, 22.3, 20, 19, 16]
    temp = torch.FloatTensor(a)

    print(temp.size(), temp.dim())

    print(f"월, 화 평균온도는 : {temp[0]}, {temp[1]}")
    print(f"화 ~ 목 평균온도는 : {temp[1:4]}")

    t = torch.FloatTensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ]
    )

    print(t.size(), t.dim())

    print(t[1:3, 1:3])

    from sklearn.datasets import load_boston

    boston = load_boston()
    # data = torch.from_numpy(boston['data'])
    # feature_names = torch.from_numpy(boston['feature_names'])
    # target = torch.from_numpy(boston['target'])
    # pprint.pprint(boston)
    # pprint.pprint(data)
    # pprint.pprint(feature_names)
    # pprint.pprint(target)

    data_tensor = torch.from_numpy(boston.data)

    print(data_tensor.size(), data_tensor.dim())


def dogdog():
    dog = np.array(Image.open("./dataset/dataset_CatDog/dog.566.jpg"))

    dog_tensor = torch.from_numpy(dog)
    # 3D Tensor print
    print(f"size and demension of the dog color image\n"
          f"size():{dog_tensor.size()}\ndim():{dog_tensor.dim()}")

    # 4D Tensor : color images가 여러개 들어가 있는 데이터셋
    data_path = "./dataset/dataset_CatDog/training_set/dogs/"
    # 특정 문자를 푸함한 모든 것을 가지고 오기 위해서 glob
    from glob import glob
    dogs_path_list = glob(data_path + "*.jpg")

    # 리스트로 변환
    print(dogs_path_list[:64])
    img_list = []
    for dog in dogs_path_list[:64]:
        img_list.append(np.array(Image.open(dog).resize((224, 224))))
        # img_list.append(Image.open(dog).resize((224, 224)))
    print(img_list[0])
    cv2.imshow("dog", img_list[0])

    # dog_imgs = np.array(img_list)
    # imgshow = Image.fromarray(dog_imgs[1])
    # imgshow.show()

    # print(dog_imgs.size, dog_imgs.shape, dog_imgs.ndim)


def show_img(a, b, c, d):
    row, col, _channels = map(int, a.shape)
    while True:
        IM = imgMake()
        dstimage = IM.create_image_multiple(224, 224, 3, 2, 2)
        IM.showMultiImage(dstimage, a, row, col, _channels, 0, 0)
        IM.showMultiImage(dstimage, b, row, col, _channels, 0, 1)
        IM.showMultiImage(dstimage, c, row, col, _channels, 1, 0)
        IM.showMultiImage(dstimage, d, row, col, _channels, 1, 1)

        cv2.imshow("dog", dstimage)
        if cv2.waitKey(1) == 27:
            break


def pytorch_function():
    m1 = torch.FloatTensor([[3, 3]])
    m2 = torch.FloatTensor([[2, 2]])
    print(m1 + m2)

    # m2 1x1 벡터(실수) 정의, m1(1x2)  + m2(1x1)
    m2 = torch.FloatTensor([2])
    # m1 = [3, 3] m2 = [2] ==> m2 = [2, 2]: 자동으로 동일한 값으로 확장
    print(f"shape가 맞지 않을 경우 확장해서 계산\n{m1 + m2}")

    # m1 = 1x2  m2 = 2x1 matrix
    # m1 + m2 = broadcating을 통해서 m1, m2 모두 2x2로 확장해서 계산
    m1 = torch.FloatTensor([[1, 2]])
    m2 = torch.FloatTensor([[3], [4]])
    """
    [[1,2] + [[3, 3] = [[4, 5]
     [1,2]]   [4, 4]]   [5, 6]]
    """
    print(f"{m1 + m2}")

    # in-place 연산(덮어쓰기 연산 : pandas)
    # in-place: _(언더바) 사용
    a = torch.FloatTensor([[1, 2], [3, 4]])
    b = torch.FloatTensor([[1, 2], [3, 4]])
    """
    [[1, 2] + [[2, 2] = [[3, 4]
     [3, 4]]   [2, 2]]   [5, 6]]
    """
    print(f"in-place\n{a.add_(2)}\n{a}")

    # matrix 곱, elements-wise 곱
    # 1) matrix multiplication: tensor.matmul(tensor) (m1.matmul(m2))
    # m1: 2x2 m2: 2x1 ==> matnul: 2x1
    m1 = torch.FloatTensor([[1, 2], [3, 4]])
    m2 = torch.FloatTensor([[1], [2]])
    print(f"matrix multiplication\n{m1.matmul(m2)}")
    # 2) matrix elements-wise: tensor * tensor 원소들 끼리의 곱 연산 (m1 * m2, m1.mul(m2))
    print(f"matrix elements-wise\n{m1 * m2}")
    print(m1.mul(m2))


def GPU():
    from torch import cuda
    print(f"torch_version : {torch.__version__}")
    torch.cuda.device(0)
    use_gpu = cuda.is_available()
    print(use_gpu, torch.cuda.device_count())


GPU()
