import os
import time
import datetime
import glob
import shutil

import cv2
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from model import AutoEncoderCNN
from logger import TrainLogger
from ae_dataset import ImgTransform, CustomDataset
from common import tensor2cv, cv2pil


class AEapp():
    def __init__(self, img_size=256):
        # data
        self.data_transforms = {}
        self.dataset = {}
        self.dataloaders = {}
        self.dataset_sizes = {}

        # model
        self.device = None
        self.model = None
        self.img_size = img_size
        self.batch_size = 3
        self.loss_weights = None
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None

        # train
        self.epoch_num = None
        self.model_name = None

        self.best_score = None
        self.epoch_loss = {}
        self.best_loss = {}

        self.total_minutes = None
        self.train_id = None
        self.results_dict = {}

        self.logger = None

        self.data_transforms = {'train': ImgTransform(phase='train', img_size=self.img_size),
                                'val': ImgTransform(phase='val', img_size=self.img_size)}
        self.metrics_init()

    def params_set(self):
        # self.loss_fn = nn.BCELoss()
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=50,
                                                   gamma=0.3)

    def data_set(self, train_list, val_list):

        _train_dataset = CustomDataset(train_list,
                                       transform=self.data_transforms['train'])
        _val_dataset = CustomDataset(val_list,
                                     transform=self.data_transforms['val'])
        _all_dataset = CustomDataset(train_list + val_list,
                                     transform=self.data_transforms['val'])

        self.dataset = {'train': _train_dataset,
                        'val': _val_dataset,
                        'all': _all_dataset}

        # data-loader
        self.dataloaders = {'train': torch.utils.data.DataLoader(self.dataset['train'],
                                                                 batch_size=self.batch_size,
                                                                 shuffle=True,
                                                                 num_workers=1),
                            'val': torch.utils.data.DataLoader(self.dataset['val'],
                                                               batch_size=self.batch_size,
                                                               shuffle=True,
                                                               num_workers=1),
                            'all': torch.utils.data.DataLoader(self.dataset['all'],
                                                               batch_size=self.batch_size,
                                                               shuffle=True,
                                                               num_workers=1)
                            }

        self.dataset_sizes = {x: len(self.dataset[x]) for x in ['train', 'val', 'all']}

    def model_init(self, net, init_model=None, device=None):
        # Set Device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print("Using {} device".format(self.device))

        # Set Model
        self.model = net
        self.model = self.model.to(self.device)

        # Set pre-training model if possible
        if init_model is not None:
            print(f"load init_model: {init_model}")
            self.model.load_state_dict(torch.load(init_model))

        self.params_set()

    def train(self, num_epochs=10, model_name='model', debug_dir=None):
        if debug_dir is not None:
            if os.path.isdir(debug_dir):
                shutil.rmtree(debug_dir)
            os.makedirs(debug_dir, exist_ok=True)

        self.epoch_num = num_epochs
        self.model_name = model_name
        time_start = time.time()

        print('start train...')
        self.init_logger()
        for epoch in range(self.epoch_num):
            self.metrics_init(epoch)

            for phase in ['train', 'val', 'all']:
                if phase == 'train':
                    self.model.train()
                else:
                    # if (epoch + 1) % 5 != 0: continue
                    self.model.eval()

                # train model
                running_loss = 0.0
                for images in self.dataloaders[phase]:
                    images = images.to(self.device)

                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs, inputs = self.model(images)
                        loss = self.loss_fn(outputs, inputs)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * images.size(0)

                if phase == 'train':
                    self.scheduler.step()

                # log results
                # score: loss @ val
                _score = None
                if phase == 'val' and debug_dir is not None:
                    for images in self.dataloaders[phase]:
                        images = images.to(self.device)
                        outputs, inputs = self.model(images)

                        img_org = self._tensor2cv(inputs[0,])
                        img_out = self._tensor2cv(outputs[0,])

                        img_show = cv2.hconcat([img_org, img_out])
                        cv2.imwrite(f'{debug_dir}/val_epoch{epoch:05d}.jpg', img_show)
                        break

                if phase == 'all':
                    _score = self.epoch_loss['val']

                self.metrics_set(phase, running_loss, score=_score)
                self.set_logger(epoch)

                if phase == 'all':
                    if (epoch + 1) % 5 == 0 or True:
                        self.save_logger()
                        self.logger.save_model(self.model,
                                               f'{model_name}_best.pth',
                                               metrics=self.epoch_loss['all'])

                        print(f'{epoch}epoch: Loss: {self.epoch_loss}')

        self.total_minutes = (time.time() - time_start) // 60
        self.set_results()

        self.logger.save_dict(self.results_dict)
        self.logger.save_model(self.model, f'{model_name}_best.pth', metrics=self.epoch_loss['all'])
        self.logger.save_model(self.model, f'{model_name}.pth', metrics=None)

        torch.save(self.model.state_dict(), f'{model_name}.pth')
        print(f"Saved PyTorch Model State to {model_name}.pth")

        return

    def metrics_init(self, epoch=None):

        self.epoch_loss = dict.fromkeys(['train', 'val', 'all'], None)

        if epoch is None or epoch == 0:
            self.best_score = None
            self.best_loss = dict.fromkeys(['train', 'val', 'all'], None)

    def metrics_set(self, phase, loss, score=None):
        self.epoch_loss[phase] = loss / self.dataset_sizes[phase]

        if score is None:
            return

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score:
            self.best_score = score
        else:
            return

        self.best_loss = self.epoch_loss
        torch.save(self.model.state_dict(), f'{self.model_name}_best.pth')

    def set_results(self):

        self.results_dict = {
            'id': self.train_id,
            'model_name': self.model_name,
            'dataset_sizes': self.dataset_sizes,
            'optimizer': self.optimizer,
            'loss_func': self.loss_fn,
            'loss_weight': self.loss_weights,
            'image_size': self.img_size,
            'batch_size': self.batch_size,
            'epochs': self.epoch_num,
            'train_time': self.total_minutes,
            'loss_train': self.epoch_loss['train'],
            'loss_val': self.epoch_loss['val'],
            'loss_all': self.epoch_loss['all'],
            'loss_train_best': self.best_loss['train'],
            'loss_val_best': self.best_loss['val'],
            'loss_all_best': self.best_loss['all'],
        }

        return self.results_dict

    def init_logger(self):
        id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.train_id = id

        self.logger = TrainLogger()
        self.logger.init(id=id)
        self.logger.print_obj(self.model, filename='model.txt')

        self.logger.init_plot('loss', 'train')
        self.logger.init_plot('loss', 'val')
        self.logger.init_plot('loss', 'all')

    def set_logger(self, epoch):

        self.logger.set_plot('loss', 'train', epoch, self.epoch_loss['train'])
        self.logger.set_plot('loss', 'val', epoch, self.epoch_loss['val'])
        self.logger.set_plot('loss', 'all', epoch, self.epoch_loss['all'])

    def save_logger(self):
        self.logger.save_plot('loss')

    def test(self, img_path,
             out_path=None,
             norm_factor=None,
             phase='val'):

        _device = "cpu"
        if self.device != _device:
            self.model = self.model.to(_device)
        self.model.eval()

        # load image
        img = cv2.imread(img_path, 1)
        img = cv2pil(img)
        img = self.data_transforms[phase](img)
        img = img.unsqueeze(0)
        img = img.to(_device)

        # test model
        output, input = self.model(img)

        # show images
        img_org = self._tensor2cv(input[0,])
        img_out = self._tensor2cv(output[0,])

        img_diff = img_out.astype(int) - img_org.astype(int)

        if norm_factor is None:
            norm_factor = 255 / np.abs(img_diff).max()

        img_diff = img_diff * norm_factor
        img_diff = np.floor_divide(img_diff, 2) + 128
        img_diff = np.clip(img_diff, 0, 255)
        img_diff = np.uint8(img_diff)

        img_show = cv2.hconcat([img_org, img_out, img_diff])

        if out_path is None:
            cv2.imshow('test', img_show)
            cv2.waitKey()
        else:
            cv2.imwrite(out_path, img_show)

    def _tensor2cv(self, tensor_image):

        image = tensor_image.view(3, self.img_size, self.img_size).detach().cpu()

        image = tensor2cv(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (300, 300))

        return image


if __name__ == '__main__':
    app = AEapp()

    model = AutoEncoderCNN()
    app.model_init(model, init_model='CAE_best.pth')

    for path in glob.glob('./data_screw/**/*.png', recursive=True):
        print(path)
        app.test(path, norm_factor=2)
