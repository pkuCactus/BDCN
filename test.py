import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import time
import re
import os
import sys
import cv2
import bdcn
from datasets.dataset import Data
import argparse
import cfg
from matplotlib import pyplot as plt

def sigmoid(x):
    return 1./(1+np.exp(np.array(-1.*x)))


def test(model, args):
    test_root = cfg.config_test[args.dataset]['data_root']
    test_lst = cfg.config_test[args.dataset]['data_lst']
    test_name_lst = os.path.join(test_root, 'voc_valtest.txt')
    # test_name_lst = os.path.join(test_root, 'test_id.txt')
    if 'Multicue' in args.dataset:
        test_lst = test_lst % args.k
        test_name_lst = os.path.join(test_root, 'test%d_id.txt'%args.k)
    mean_bgr = np.array(cfg.config_test[args.dataset]['mean_bgr'])
    test_img = Data(test_root, test_lst, mean_bgr=mean_bgr)
    testloader = torch.utils.data.DataLoader(
        test_img, batch_size=1, shuffle=False, num_workers=8)
    nm = np.loadtxt(test_name_lst, dtype=str)
    print(len(testloader), len(nm))
    assert len(testloader) == len(nm)
    save_res = True
    save_dir = args.res_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if args.cuda:
        model.cuda()
    model.eval()
    data_iter = iter(testloader)
    iter_per_epoch = len(testloader)
    start_time = time.time()
    all_t = 0
    for i, (data, _) in enumerate(testloader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        tm = time.time()
        out = model(data)
        fuse = F.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]
        if not os.path.exists(os.path.join(save_dir, 'fuse')):
            os.mkdir(os.path.join(save_dir, 'fuse'))
        cv2.imwrite(os.path.join(save_dir, 'fuse', '%s.png'%nm[i]), 255-fuse*255)
        all_t += time.time() - tm
    print all_t
    print 'Overall Time use: ', time.time() - start_time

def main():
    import time
    print time.localtime()
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = bdcn.BDCN()
    model.load_state_dict(torch.load('%s' % (args.model)))
    test(model, args)

def parse_args():
    parser = argparse.ArgumentParser('test BDCN')
    parser.add_argument('-d', '--dataset', type=str, choices=cfg.config_test.keys(),
        default='bsds500', help='The dataset to train')
    parser.add_argument('-c', '--cuda', action='store_true',
        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
        help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str, default='params/bdcn_final.pth',
        help='the model to test')
    parser.add_argument('--res-dir', type=str, default='result',
        help='the dir to store result')
    parser.add_argument('-k', type=int, default=1,
        help='the k-th split set of multicue')
    return parser.parse_args()

if __name__ == '__main__':
    main()
