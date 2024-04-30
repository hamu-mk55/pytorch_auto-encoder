import os
import random
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import common


def _dummy(img):
    return img


class ImgTransform():
    def __init__(self,
                 func=_dummy,
                 phase='train',
                 img_size=256,
                 mean=None,
                 std=None):
        if std is None:
            std = [0.229, 0.224, 0.225]
        if mean is None:
            mean = [0.485, 0.456, 0.406]

        _img_transform = {
            'train': transforms.Compose([
                transforms.Lambda(func),
                transforms.Resize(img_size),
                transforms.RandomResizedCrop(int(img_size * 0.9),
                                             scale=(0.95, 1.0),
                                             ratio=(0.98, 1.02)),
                transforms.Resize(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std)
            ]),

            'val': transforms.Compose([
                transforms.Lambda(func),
                transforms.Resize(img_size),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std)
            ])
        }

        self.phase = phase
        self.transform = _img_transform[self.phase]

    def __call__(self, img):
        return self.transform(img)

    def show(self, img):
        plt.subplot(1, 2, 1)
        plt.imshow(img)

        plt.subplot(1, 2, 2)
        img_trans = self.transform(img)
        img_trans = common.tensor2cv(img_trans)
        plt.imshow(img_trans)

        plt.show()


def make_data_list(file_ext='jpg',
                   src_dir='./images',
                   ratio=0.75,
                   shuffle=True,
                   mode='both'):
    print('make_data_list===========')
    # open output
    if mode != 'train':
        test = open('test.txt', 'w')
    if mode != 'test':
        train = open('train.txt', 'w')

    # images
    images = glob.glob(f'{src_dir}/**/*.{file_ext}', recursive=True)
    img_num = len(images)

    if shuffle:
        random.shuffle(images)

    cnt = 0
    for image_path in images:
        if cnt < img_num * ratio:
            if mode != 'test':
                train.write(image_path + '\n')
        else:
            if mode != 'train':
                test.write(image_path + '\n')

        cnt += 1

    # close output
    if mode != 'test':
        train.close()
    if mode != 'train':
        test.close()

    return 1


def load_data_list(data_file):
    data_list = []

    with open(data_file) as datafile:
        for line in datafile:
            tmp1 = line.split(' ')[0]
            data_list.append(tmp1)

    return data_list


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, datalist, transform=None):
        self.transform = transform
        self.data_list = datalist

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        _list = self.data_list

        _img_path = _list[index]
        _img_path = _img_path.strip()

        _img = cv2.imread(_img_path, 1)

        _img = common.cv2pil(_img)

        if self.transform is not None:
            _img = self.transform(_img)

        return _img


if __name__ == '__main__':
    trans = ImgTransform(phase='train')

    img = cv2.imread('1.png', 1)

    for i in range(10):
        img1 = trans(common.cv2pil(img))

        common.show_images(img, common.tensor2cv(img1), size_no=2)

    pass
    exit()
