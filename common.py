import os
import csv
import cv2
import numpy as np
from PIL import Image


def dict2csv(dict, filename):
    if not os.path.isfile(filename):
        fw = open(filename, 'w', newline='')
        writer = csv.writer(fw, delimiter=',')

        _list = list(dict.keys())
        _list = [str(s).strip() for s in _list]
        writer.writerow(_list)
        fw.close()

    fw = open(filename, 'a', newline='')
    writer = csv.writer(fw, delimiter=',')
    _list = list(dict.values())
    _list = [str(s).strip() for s in _list]
    writer.writerow(_list)
    fw.close()


def cv2pil(img: np.array):
    """
    convert OpenCV-format to PIL-format
    :param img:
    :return:
    """

    img_pil = img.copy()

    if img_pil.ndim == 2:  # モノクロ
        pass
    elif img_pil.shape[2] == 3:  # カラー
        img_pil = cv2.cvtColor(img_pil, cv2.COLOR_BGR2RGB)
    elif img_pil.shape[2] == 4:  # 透過
        img_pil = cv2.cvtColor(img_pil, cv2.COLOR_BGRA2RGBA)

    img_pil = Image.fromarray(img_pil)

    return img_pil


def tensor2cv(img, use_norm: bool = False):
    """
    convert Tensor-format to OpenCV-format
    :param img:
    :param use_norm:
    :return:
    """
    img = img.numpy().transpose((1, 2, 0))

    if use_norm:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean

    img = np.clip(img, 0, 1)
    img = np.uint8(img * 255)

    return img


def show_images(img1: np.array, img2: np.array, size_no: int = 1):
    """
    show 2-images
    :param img1:
    :param img2:
    :param size_no:
    :return:
    """
    if size_no == 1:
        img2 = cv2.resize(img2, (img1.shape[0], img1.shape[1],))
    else:
        img1 = cv2.resize(img1, (img2.shape[0], img2.shape[1],))

    imgs = cv2.hconcat([img1, img2])

    title = 'test'
    cv2.imshow(title, cv2.resize(imgs, dsize=None, fx=1, fy=1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
