import numpy as np
import torch
import torchvision
import torch.nn as nn
import math

class VGG16_C(nn.Module):
    """"""
    def __init__(self, pretrain=None, logger=None):
        super(VGG16_C, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, (3, 3), stride=1, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv2_1 = nn.Conv2d(64, 128, (3, 3), stride=1, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, (3, 3), stride=1, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv3_1 = nn.Conv2d(128, 256, (3, 3), stride=1, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv4_1 = nn.Conv2d(256, 512, (3, 3), stride=1, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)
        self.conv5_1 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=2, dilation=2)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=2, dilation=2)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=2, dilation=2)
        self.relu5_3 = nn.ReLU(inplace=True)
        if pretrain:
            if '.npy' in pretrain:
                state_dict = np.load(pretrain).item()
                for k in state_dict:
                    state_dict[k] = torch.from_numpy(state_dict[k])
            else:
                state_dict = torch.load(pretrain)
            own_state_dict = self.state_dict()
            for name, param in own_state_dict.items():
                if name in state_dict:
                    if logger:
                        logger.info('copy the weights of %s from pretrained model' % name)
                    param.copy_(state_dict[name])
                else:
                    if logger:
                        logger.info('init the weights of %s from mean 0, std 0.01 gaussian distribution'\
                         % name)
                    if 'bias' in name:
                        param.zero_()
                    else:
                        param.normal_(0, 0.01)
        else:
            self._initialize_weights(logger)

    def forward(self, x):
        conv1_1 = self.relu1_1(self.conv1_1(x))
        conv1_2 = self.relu1_2(self.conv1_2(conv1_1))
        pool1 = self.pool1(conv1_2)
        conv2_1 = self.relu2_1(self.conv2_1(pool1))
        conv2_2 = self.relu2_2(self.conv2_2(conv2_1))
        pool2 = self.pool2(conv2_2)
        conv3_1 = self.relu3_1(self.conv3_1(pool2))
        conv3_2 = self.relu3_2(self.conv3_2(conv3_1))
        conv3_3 = self.relu3_3(self.conv3_3(conv3_2))
        pool3 = self.pool3(conv3_3)
        conv4_1 = self.relu4_1(self.conv4_1(pool3))
        conv4_2 = self.relu4_2(self.conv4_2(conv4_1))
        conv4_3 = self.relu4_3(self.conv4_3(conv4_2))
        pool4 = self.pool4(conv4_3)
        # pool4 = conv4_3
        conv5_1 = self.relu5_1(self.conv5_1(pool4))
        conv5_2 = self.relu5_2(self.conv5_2(conv5_1))
        conv5_3 = self.relu5_3(self.conv5_3(conv5_2))

        side = [conv1_1, conv1_2, conv2_1, conv2_2,
                conv3_1, conv3_2, conv3_3, conv4_1,
                conv4_2, conv4_3, conv5_1, conv5_2, conv5_3]
        return side

    def _initialize_weights(self, logger=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if logger:
                        logger.info('init the weights of %s from mean 0, std 0.01 gaussian distribution'\
                         % m)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

if __name__ == '__main__':
    model = VGG16_C()
    # im = np.zeros((1,3,100,100))
    # out = model(Variable(torch.from_numpy(im)))


