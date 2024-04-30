import torch
from torch import nn
import numpy as np

from torchinfo import summary


class AutoEncoderCNN(nn.Module):
    def __init__(self, max_channel=None, use_bn=False, deconv_num=1):
        super(AutoEncoderCNN, self).__init__()

        self.max_channel = max_channel
        self.use_bn = use_bn
        self.deconv_num = deconv_num

        # encoder
        self.encoder = nn.Sequential(
            self._make_conv(3, 16),
            nn.MaxPool2d((2, 2)),
            self._make_conv(16, 32),
            nn.MaxPool2d((2, 2)),
            self._make_conv(32, 64),
            nn.MaxPool2d((2, 2)),
            self._make_conv(64, 128),
            nn.MaxPool2d((2, 2)),
            self._make_conv(128, 256),
            nn.MaxPool2d((2, 2)),
            self._make_conv(256, 512),
            nn.MaxPool2d((2, 2)),
            self._make_conv(512, 1024),
        )

        # decoder
        self.decoder = nn.Sequential(
            self._make_deconv(1024, 512),
            self._make_deconv(512, 256),
            self._make_deconv(256, 128),
            self._make_deconv(128, 64),
            self._make_deconv(64, 32),
            self._make_deconv(32, 16),
            nn.Conv2d(in_channels=16,
                      out_channels=3,
                      kernel_size=1,
                      stride=1),
        )

    def forward(self, x, debug=False):
        if debug: print(x.size())

        y = self.encoder(x)
        if debug: print(y.size())

        y = self.decoder(y)
        if debug: print(y.size())

        return y, x

    def _make_conv(self, channel_in, channel_out):
        if self.max_channel is not None:
            channel_in = min(self.max_channel, channel_in)
            channel_out = min(self.max_channel, channel_out)

        conv = nn.Sequential()

        conv.add_module('conv',
                        nn.Conv2d(in_channels=channel_in,
                                  out_channels=channel_out,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1))

        if self.use_bn:
            conv.add_module('bn', nn.BatchNorm2d(channel_out))

        conv.add_module('relu', nn.ReLU())

        return conv

    def _make_deconv(self, channel_in, channel_out):
        if self.max_channel is not None:
            channel_in = min(self.max_channel, channel_in)
            channel_out = min(self.max_channel, channel_out)

        deconv = nn.Sequential()

        deconv.add_module('deconv',
                          nn.ConvTranspose2d(in_channels=channel_in,
                                             out_channels=channel_out,
                                             kernel_size=2,
                                             stride=2))

        if self.use_bn:
            deconv.add_module('bn', nn.BatchNorm2d(channel_out))

        deconv.add_module('relu', nn.ReLU())

        for cnt in range(self.deconv_num):
            deconv.add_module(f'conv{cnt}', self._make_conv(channel_out, channel_out))

        return deconv


if __name__ == '__main__':
    model = AutoEncoderCNN().to('cpu')
    print(model)
    print('========================')
    # for param in model.parameters():
    #     print(param)
    print('========================')
    summary(model, (1, 3, 256, 256))

    img = np.zeros((256, 256, 3))
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img.astype(np.float32))
    img = img.unsqueeze(0)

    model(img, debug=True)
    exit()
