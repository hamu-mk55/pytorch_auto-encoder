import os
import csv
import glob

from ae_app import AEapp
from ae_dataset import make_data_list, load_data_list
from model import AutoEncoderCNN
from common import dict2csv


def train(max_channel=None, use_bn=False, deconv_num=1):
    csvfile = 'results_train.csv'

    # data set
    data_dir = './data2'
    make_data_list(file_ext='png', src_dir=data_dir, ratio=0.75, mode='both')

    train_list = load_data_list('train.txt')
    val_list = load_data_list('test.txt')

    # model
    _name = 'AE'
    model = AutoEncoderCNN(max_channel=max_channel, use_bn=use_bn, deconv_num=deconv_num)

    app = AEapp()
    app.data_set(train_list, val_list)
    app.model_init(model, init_model=None, device="cuda")
    app.train(num_epochs=50, model_name=_name, debug_dir='./debug')

    dict2csv(app.results_dict, csvfile)


def test(max_channel=None, use_bn=False, deconv_num=1):
    image_dir = './data'
    output_dir = './output'

    app = AEapp()

    model = AutoEncoderCNN(max_channel=max_channel, use_bn=use_bn, deconv_num=deconv_num)
    app.model_init(model, init_model='AE_best.pth')

    for path in glob.glob(f'{image_dir}/**/*.png', recursive=True):
        out_path = path.replace(image_dir, output_dir)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        app.test(path, out_path=out_path, norm_factor=2)


if __name__ == '__main__':
    params = {'max_channel': 512,
              'use_bn': True,
              'deconv_num':2}

    # train(**params)

    test(**params)


