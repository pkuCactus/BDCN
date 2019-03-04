import numpy as np
import torch
import torch.nn as nn

import ablation_vgg16_c

def crop(data1, data2, crop_h, crop_w):
    _, _, h1, w1 = data1.size()
    _, _, h2, w2 = data2.size()
    assert(h2 <= h1 and w2 <= w1)
    data = data1[:, :, crop_h:crop_h+h2, crop_w:crop_w+w2]
    return data

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class MSBlock(nn.Module):
    def __init__(self, c_in, k=3, rate=4):
        super(MSBlock, self).__init__()
        c_out = c_in
        self.k = k
        self.rate = rate

        self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        if k>=1:
            dilation = self.rate*1 if self.rate >= 1 else 1
            self.conv1 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
            self.relu1 = nn.ReLU(inplace=True)
        if k>=2:
            dilation = self.rate*2 if self.rate >= 1 else 1
            self.conv2 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
            self.relu2 = nn.ReLU(inplace=True)
        if k>=3:
            dilation = self.rate*3 if self.rate >= 1 else 1
            self.conv3 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
            self.relu3 = nn.ReLU(inplace=True)
        if k>=4:
            dilation = self.rate*4 if self.rate >= 1 else 1
            self.conv4 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
            self.relu4 = nn.ReLU(inplace=True)

        self._initialize_weights()

    def forward(self, x):
        o = self.relu(self.conv(x))
        if self.k>=1:
            o1 = self.relu1(self.conv1(o))
        if self.k>=2:
            o2 = self.relu2(self.conv2(o))
        if self.k>=3:
            o3 = self.relu3(self.conv3(o))
        if self.k>=4:
            o4 = self.relu4(self.conv4(o))
        if self.k < 1:
            return o
        elif self.k>=4:
            return o+o1+o2+o3+o4
        elif self.k>=3:
            return o + o1 + o2 + o3
        elif self.k>=2:
            return o + o1 + o2
        elif self.k>=1:
            return o + o1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class BDCN(nn.Module):
    def __init__(self, pretrain=None, logger=None, ms=True, block=5, bdcn=True, direction='both', k=3, rate=4):
        super(BDCN, self).__init__()
        if logger:
            logger.info(ms)
            logger.info(block)
            logger.info(bdcn)
        self.pretrain = pretrain
        self.ms = ms
        self.block = block
        self.bdcn = bdcn
        self.dir = direction
        self.k = k
        t = 1

        self.features = ablation_vgg16_c.VGG16_C(pretrain, logger, block=block)
        if ms:
            self.msblock1_1 = MSBlock(64, k, rate)
            self.msblock1_2 = MSBlock(64, k, rate)
        else:
            t = 2
        self.conv1_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv1_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.score_dsn1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn1_1 = nn.Conv2d(21, 1, 1, stride=1)
        if block >= 2:
            if ms:
                self.msblock2_1 = MSBlock(128, k, rate)
                self.msblock2_2 = MSBlock(128, k, rate)
            else:
                t = 4
            self.conv2_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
            self.conv2_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
            self.score_dsn2 = nn.Conv2d(21, 1, (1, 1), stride=1)
            self.score_dsn2_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
            self.upsample_2 = nn.ConvTranspose2d(1, 1, 4, stride=2, bias=False)
        if block >= 3:
            if ms:
                self.msblock3_1 = MSBlock(256, k, rate)
                self.msblock3_2 = MSBlock(256, k, rate)
                self.msblock3_3 = MSBlock(256, k, rate)
            else:
                t = 8
            self.conv3_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
            self.conv3_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
            self.conv3_3_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
            self.score_dsn3 = nn.Conv2d(21, 1, (1, 1), stride=1)
            self.score_dsn3_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
            self.upsample_4 = nn.ConvTranspose2d(1, 1, 8, stride=4, bias=False)
        if block >= 4:
            if ms:
                self.msblock4_1 = MSBlock(512, k, rate)
                self.msblock4_2 = MSBlock(512, k, rate)
                self.msblock4_3 = MSBlock(512, k, rate)
            else:
                t = 16
            self.conv4_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
            self.conv4_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
            self.conv4_3_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
            self.score_dsn4 = nn.Conv2d(21, 1, (1, 1), stride=1)
            self.score_dsn4_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
            self.upsample_8 = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        if block >=5:
            if ms:
                self.msblock5_1 = MSBlock(512, k, rate)
                self.msblock5_2 = MSBlock(512, k, rate)
                self.msblock5_3 = MSBlock(512, k, rate)
            else:
                t = 16
            self.conv5_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
            self.conv5_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
            self.conv5_3_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
            self.score_dsn5 = nn.Conv2d(21, 1, (1, 1), stride=1)
            self.score_dsn5_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
            self.upsample_8_5 = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        if bdcn and self.dir == 'both':
            c = block * 2
        else:
            c = block
        self.fuse = nn.Conv2d(c, 1, 1, stride=1)

        self._initialize_weights(logger)

    def forward(self, x):
        features = self.features(x)
        if self.ms:
            sum1 = self.conv1_1_down(self.msblock1_1(features[0])) + \
                    self.conv1_2_down(self.msblock1_2(features[1]))
        else:
            sum1 = self.conv1_1_down(features[0]) + \
                    self.conv1_2_down(features[1])
        s1 = self.score_dsn1(sum1)
        if self.bdcn:
            s11 = self.score_dsn1_1(sum1)
        if self.block >= 2:
            if self.ms:
                sum2 = self.conv2_1_down(self.msblock2_1(features[2])) + \
                    self.conv2_2_down(self.msblock2_2(features[3]))
            else:
                sum2 = self.conv2_1_down(features[2]) + \
                    self.conv2_2_down(features[3])
            s2 = self.score_dsn2(sum2)
            s2 = self.upsample_2(s2)
            s2 = crop(s2, x, 1, 1)
            if self.bdcn:
                s21 = self.score_dsn2_1(sum2)
                s21 = self.upsample_2(s21)
                s21 = crop(s21, x, 1, 1)
        if self.block >= 3:
            if self.ms:
                sum3 = self.conv3_1_down(self.msblock3_1(features[4])) + \
                    self.conv3_2_down(self.msblock3_2(features[5])) + \
                    self.conv3_3_down(self.msblock3_3(features[6]))
            else:
                sum3 = self.conv3_1_down(features[4]) + \
                    self.conv3_2_down(features[5]) + \
                    self.conv3_3_down(features[6])
            s3 = self.score_dsn3(sum3)
            s3 =self.upsample_4(s3)
            s3 = crop(s3, x, 2, 2)
            if self.bdcn:
                s31 = self.score_dsn3_1(sum3)
                s31 =self.upsample_4(s31)
                s31 = crop(s31, x, 2, 2)
        if self.block >= 4:
            if self.ms:
                sum4 = self.conv4_1_down(self.msblock4_1(features[7])) + \
                    self.conv4_2_down(self.msblock4_2(features[8])) + \
                    self.conv4_3_down(self.msblock4_3(features[9]))
            else:
                sum4 = self.conv4_1_down(features[7]) + \
                    self.conv4_2_down(features[8]) + \
                    self.conv4_3_down(features[9])
            s4 = self.score_dsn4(sum4)
            s4 = self.upsample_8(s4)
            s4 = crop(s4, x, 4, 4)
            if self.bdcn:
                s41 = self.score_dsn4_1(sum4)
                s41 = self.upsample_8(s41)
                s41 = crop(s41, x, 4, 4)
        if self.block >= 5:
            if self.ms:
                sum5 = self.conv5_1_down(self.msblock5_1(features[10])) + \
                    self.conv5_2_down(self.msblock5_2(features[11])) + \
                    self.conv5_3_down(self.msblock5_3(features[12]))
            else:
                sum5 = self.conv5_1_down(features[10]) + \
                    self.conv5_2_down(features[11]) + \
                    self.conv5_3_down(features[12])
            s5 = self.score_dsn5(sum5)
            s5 = self.upsample_8_5(s5)
            s5 = crop(s5, x, 0, 0)
            if self.bdcn:
                s51 = self.score_dsn5_1(sum5)
                s51 = self.upsample_8_5(s51)
                s51 = crop(s51, x, 0, 0)
        if self.bdcn:
            if self.block >= 5:
                o1, o2, o3, o4, o5 = s1.detach(), s2.detach(), s3.detach(), s4.detach(), s5.detach()
                o11, o21, o31, o41, o51 = s11.detach(), s21.detach(), s31.detach(), s41.detach(), s51.detach()
                p1_1 = s1
                p2_1 = s2 + o1
                p3_1 = s3 + o2 + o1
                p4_1 = s4 + o3 + o2 + o1
                p5_1 = s5 + o4 + o3 + o2 + o1
                p1_2 = s11 + o21 + o31 + o41 + o51
                p2_2 = s21 + o31 + o41 + o51
                p3_2 = s31 + o41 + o51
                p4_2 = s41 + o51
                p5_2 = s51
                if self.dir == 'both':
                    fuse = self.fuse(torch.cat([p1_1, p2_1, p3_1, p4_1, p5_1, p1_2, p2_2, p3_2, p4_2, p5_2], 1))
                    return [p1_1, p2_1, p3_1, p4_1, p5_1, p1_2, p2_2, p3_2, p4_2, p5_2, fuse]
                if self.dir == 'd2s':
                    fuse = self.fuse(torch.cat([p1_1, p2_1, p3_1, p4_1, p5_1], 1))
                    return [p1_1, p2_1, p3_1, p4_1, p5_1, fuse]
                elif self.dir == 's2d':
                    fuse = self.fuse(torch.cat([p1_2, p2_2, p3_2, p4_2, p5_2], 1))
                    return [p1_2, p2_2, p3_2, p4_2, p5_2, fuse]
            elif self.block >= 4:
                o1, o2, o3, o4 = s1.detach(), s2.detach(), s3.detach(), s4.detach()
                o11, o21, o31, o41 = s11.detach(), s21.detach(), s31.detach(), s41.detach()
                p1_1 = s1
                p2_1 = s2 + o1
                p3_1 = s3 + o2 + o1
                p4_1 = s4 + o3 + o2 + o1
                p1_2 = s11 + o21 + o31 + o41
                p2_2 = s21 + o31 + o41
                p3_2 = s31 + o41
                p4_2 = s41
                fuse = self.fuse(torch.cat([p1_1, p2_1, p3_1, p4_1,p1_2, p2_2, p3_2, p4_2], 1))
                return [p1_1, p2_1, p3_1, p4_1, p1_2, p2_2, p3_2, p4_2, fuse]
            elif self.block >= 3:
                o1, o2, o3 = s1.detach(), s2.detach(), s3.detach()
                o11, o21, o31 = s11.detach(), s21.detach(), s31.detach()
                p1_1 = s1
                p2_1 = s2 + o1
                p3_1 = s3 + o2 + o1
                p1_2 = s11 + o21 + o31
                p2_2 = s21 + o31
                p3_2 = s31
                fuse = self.fuse(torch.cat([p1_1, p2_1, p3_1, p1_2, p2_2, p3_2], 1))
                return [p1_1, p2_1, p3_1, p1_2, p2_2, p3_2, fuse]
            elif self.block >= 2:
                o1, o2 = s1.detach(), s2.detach()
                o11, o21 = s11.detach(), s21.detach()
                p1_1 = s1
                p2_1 = s2 + o1
                p1_2 = s11 + o21
                p2_2 = s21
                fuse = self.fuse(torch.cat([p1_1, p2_1, p1_2, p2_2], 1))
                return [p1_1, p2_1, p1_2, p2_2, fuse]

        concat = s1
        res = [s1]
        if self.block >= 2:
            concat = torch.cat([concat, s2], 1)
            res = [s1, s2]
        if self.block >= 3:
            concat = torch.cat([concat, s3], 1)
            res = [s1, s2, s3]
        if self.block >= 4:
            concat = torch.cat([concat, s4], 1)
            res = [s1, s2, s3, s4]
        if self.block >= 5:
            concat = torch.cat([concat, s5], 1)
            res = [s1, s2, s3, s4, s5]
        fuse = self.fuse(concat)
        res.append(fuse)
        return res

    def _initialize_weights(self, logger=None):
        for name, param in self.state_dict().items():
            if self.pretrain and 'features' in name:
                continue
            elif 'down' in name:
                param.zero_()
            elif 'upsample' in name:
                if logger:
                    logger.info('init upsamle layer %s ' % name)
                k = int(name.split('.')[0].split('_')[1])
                param.copy_(get_upsampling_weight(1, 1, k*2))
            elif 'fuse' in name:
                if logger:
                    logger.info('init params %s ' % name)
                if 'bias' in name:
                    param.zero_()
                else:
                    nn.init.constant(param, 0.080)
            else:
                if logger:
                    logger.info('init params %s ' % name)
                if 'bias' in name:
                    param.zero_()
                else:
                    param.normal_(0, 0.01)

if __name__ == '__main__':
    model = BDCN('./caffemodel2pytorch/vgg16.pth')
    for name, param in model.state_dict().items():
        print name, param
