import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import argparse
import time
import re
import os
import sys
import ablation
from datasets.dataset import Data
import cfg
import log

def adjust_learning_rate(optimizer, steps, step_size, gamma=0.1, logger=None):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * gamma # (0.1 ** (steps // step_size))
        if logger:
            logger.info('%s: %s' % (param_group['name'], param_group['lr']))


def cross_entropy_loss2d(inputs, targets, cuda=False, balance=1.1):
    """
    :param inputs: inputs is a 4 dimensional data nx1xhxw
    :param targets: targets is a 3 dimensional data nx1xhxw
    :return:
    """
    n, c, h, w = inputs.size()

    weights = np.zeros((n, c, h, w))
    for i in xrange(n):
        t = targets[i, :, :, :].cpu().data.numpy()
        pos = (t == 1).sum()
        neg = (t == 0).sum()
        valid = neg + pos
        weights[i, t == 1] = neg * 1. / valid
        weights[i, t == 0] = pos * balance / valid

    weights = torch.Tensor(weights)
    if cuda:
        weights = weights.cuda()
    weights = Variable(weights)
    inputs = F.sigmoid(inputs)
    loss = nn.BCELoss(weights, size_average=False)(inputs, targets)

    return loss

def train(model, args):
    data_root = cfg.config[args.dataset]['data_root']
    data_lst = cfg.config[args.dataset]['data_lst']
    if 'Multicue' in args.dataset:
        data_lst = data_lst % args.k
    mean_bgr = np.array(cfg.config[args.dataset]['mean_bgr'])
    yita = args.yita if args.yita else cfg.config[args.dataset]['yita']
    crop_size = args.crop_size
    train_img = Data(data_root, data_lst, yita, mean_bgr=mean_bgr, crop_size=crop_size)
    trainloader = torch.utils.data.DataLoader(train_img,
        batch_size=args.batch_size, shuffle=True, num_workers=5)
    params_dict = dict(model.named_parameters())
    base_lr = args.base_lr
    weight_decay = args.weight_decay
    logger = args.logger
    params = []
    for key, v in params_dict.items():
        if re.match(r'conv[1-5]_[1-3]_down', key):
            if 'weight' in key:
                params += [{'params': v, 'lr': base_lr*0.1, 'weight_decay': weight_decay*1, 'name': key}]
            elif 'bias' in key:
                params += [{'params': v, 'lr': base_lr*0.2, 'weight_decay': weight_decay*0, 'name': key}]
        elif re.match(r'.*conv[1-4]_[1-3]', key):
            if 'weight' in key:
                params += [{'params': v, 'lr': base_lr*1, 'weight_decay': weight_decay*1, 'name': key}]
            elif 'bias' in key:
                params += [{'params': v, 'lr': base_lr*2, 'weight_decay': weight_decay*0, 'name': key}]
        elif re.match(r'.*conv5_[1-3]', key):
            if 'weight' in key:
                params += [{'params': v, 'lr': base_lr*100, 'weight_decay': weight_decay*1, 'name': key}]
            elif 'bias' in key:
                params += [{'params': v, 'lr': base_lr*200, 'weight_decay': weight_decay*0, 'name': key}]
        elif re.match(r'score_dsn[1-5]', key):
            if 'weight' in key:
                params += [{'params': v, 'lr': base_lr*0.01, 'weight_decay': weight_decay*1, 'name': key}]
            elif 'bias' in key:
                params += [{'params': v, 'lr': base_lr*0.02, 'weight_decay': weight_decay*0, 'name': key}]
        elif re.match(r'upsample_[248](_5)?', key):
            if 'weight' in key:
                params += [{'params': v, 'lr': base_lr*0, 'weight_decay': weight_decay*0, 'name': key}]
            elif 'bias' in key:
                params += [{'params': v, 'lr': base_lr*0, 'weight_decay': weight_decay*0, 'name': key}]
        elif re.match(r'.*msblock[1-5]_[1-3]\.conv', key):
            if 'weight' in key:
                params += [{'params': v, 'lr': base_lr*1, 'weight_decay': weight_decay*1, 'name': key}]
            elif 'bias' in key:
                params += [{'params': v, 'lr': base_lr*2, 'weight_decay': weight_decay*0, 'name': key}]
        else:
            if 'weight' in key:
                params += [{'params': v, 'lr': base_lr*0.001, 'weight_decay': weight_decay*1, 'name': key}]
            elif 'bias' in key:
                params += [{'params': v, 'lr': base_lr*0.002, 'weight_decay': weight_decay*0, 'name': key}]
    optimizer = torch.optim.SGD(params, momentum=args.momentum,
        lr=args.base_lr, weight_decay=args.weight_decay)
    start_step = 1
    mean_loss = []
    cur = 0
    pos = 0
    data_iter = iter(trainloader)
    iter_per_epoch = len(trainloader)
    logger.info('*'*40)
    logger.info('train images in all are %d ' % iter_per_epoch)
    logger.info('*'*40)

    start_time = time.time()
    if args.cuda:
        model.cuda()
    if args.resume:
        logger.info('resume from %s' % args.resume)
        state = torch.load(args.resume)
        start_step = state['step']
        optimizer.load_state_dict(state['solver'])
    model.train()
    batch_size = args.iter_size * args.batch_size
    for step in xrange(start_step, args.max_iter + 1):
        optimizer.zero_grad()
        batch_loss = 0
        for i in xrange(args.iter_size):
            if cur == iter_per_epoch:
                cur = 0
                data_iter = iter(trainloader)
            images, labels = next(data_iter)
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()
            images, labels = Variable(images), Variable(labels)
            out = model(images)
            loss = 0
            for k in range(len(out) - 1):
                loss += args.side_weight*cross_entropy_loss2d(out[k], labels, args.cuda, args.balance)/batch_size
            loss += args.fuse_weight*cross_entropy_loss2d(out[-1], labels, args.cuda, args.balance)/batch_size
            loss.backward()
            batch_loss += loss.data[0]
            cur += 1
        # update parameter
        optimizer.step()
        if len(mean_loss) < args.average_loss:
            mean_loss.append(batch_loss)
        else:
            mean_loss[pos] = batch_loss
            pos = (pos + 1) % args.average_loss
        if step % args.step_size == 0:
            adjust_learning_rate(optimizer, step, args.step_size)
        if step % args.snapshots == 0:
            torch.save(model.state_dict(), '%s/bdcn_%d.pth' % (args.param_dir, step))
            # state = {'step': step+1,'param':model.state_dict(),'solver':optimizer.state_dict()}
            # torch.save(state, '%s/bdcn_%d.pth.tar' % (args.param_dir, step))
        if step % args.display == 0:
            tm = time.time() - start_time
            logger.info('iter: %d, lr: %e, loss: %f, time using: %f(%fs/iter)' % (step,
                optimizer.param_groups[0]['lr'], np.mean(mean_loss), tm, tm/args.display))
            start_time = time.time()

def main():
    args = parse_args()
    logger = log.get_logger(args.log)
    args.logger = logger
    logger.info('*'*80)
    logger.info('the args are the below')
    logger.info('*'*80)
    for x in args.__dict__:
        logger.info(x+','+str(args.__dict__[x]))
    logger.info(cfg.config[args.dataset])
    logger.info('*'*80)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if not os.path.exists(args.param_dir):
        os.mkdir(args.param_dir)
    torch.manual_seed(long(time.time()))
    model = ablation.BDCN(pretrain=args.pretrain, logger=logger,
        ms=args.ms, block=args.block, bdcn=not args.no_bdcn, direction=args.dir,
        k=args.num_conv, rate=args.rate)
    if args.complete_pretrain:
        model.load_state_dict(torch.load(args.complete_pretrain))
    logger.info(model)
    train(model, args)

def parse_args():
    parser = argparse.ArgumentParser(description='Train BDCN for different args')
    parser.add_argument('-d', '--dataset', type=str, choices=cfg.config.keys(),
        default='bsds500', help='The dataset to train')
    parser.add_argument('--param-dir', type=str, default='params',
        help='the directory to store the params')
    parser.add_argument('--lr', dest='base_lr', type=float, default=1e-6,
        help='the base learning rate of model')
    parser.add_argument('-m', '--momentum', type=float, default=0.9,
        help='the momentum')
    parser.add_argument('-c', '--cuda', action='store_true',
        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
        help='the gpu id to train net')
    parser.add_argument('--weight-decay', type=float, default=0.0002,
        help='the weight_decay of net')
    parser.add_argument('-r', '--resume', type=str, default=None,
        help='whether resume from some, default is None')
    parser.add_argument('-p', '--pretrain', type=str, default=None,
        help='init net from pretrained model default is None')
    parser.add_argument('--max-iter', type=int, default=40000,
        help='max iters to train network, default is 40000')
    parser.add_argument('--iter-size', type=int, default=10,
        help='iter size equal to the batch size, default 10')
    parser.add_argument('--average-loss', type=int, default=50,
        help='smoothed loss, default is 50')
    parser.add_argument('-s', '--snapshots', type=int, default=1000,
        help='how many iters to store the params, default is 1000')
    parser.add_argument('--step-size', type=int, default=10000,
        help='the number of iters to decrease the learning rate, default is 10000')
    parser.add_argument('--display', type=int, default=20,
        help='how many iters display one time, default is 20')
    parser.add_argument('-b', '--balance', type=float, default=1.1,
        help='the parameter to balance the neg and pos, default is 1.1')
    parser.add_argument('-l', '--log', type=str, default='log.txt',
        help='the file to store log, default is log.txt')
    parser.add_argument('-k', type=int, default=1,
        help='the k-th split set of multicue')
    parser.add_argument('--batch-size', type=int, default=1,
        help='batch size of one iteration, default 1')
    parser.add_argument('--crop-size', type=int, default=None,
        help='the size of image to crop, default not crop')
    parser.add_argument('--yita', type=float, default=None,
        help='the param to operate gt, default is data in the config file')
    parser.add_argument('--complete-pretrain', type=str, default=None,
        help='finetune on the complete_pretrain, default None')
    parser.add_argument('--side-weight', type=float, default=0.5,
        help='the loss weight of sideout, default 0.5')
    parser.add_argument('--fuse-weight', type=float, default=1.1,
        help='the loss weight of fuse, default 1.1')
    parser.add_argument('--ms', action='store_true',
        help='whether employ the ms blocks, default False')
    parser.add_argument('--block', type=int, default=5,
        help='how many blocks of the model, default 5')
    parser.add_argument('--no-bdcn', action='store_true',
        help='whether to employ our policy to train the model, default False')
    parser.add_argument('--dir', type=str, choices=['both', 's2d', 'd2s'], default='both',
        help='the direction of cascade, default both')
    parser.add_argument('--num-conv', type=int, choices=[0,1,2,3,4], default=3,
        help='the number of convolution of SEB, default 3')
    parser.add_argument('--rate', type=int, default=4,
        help='the dilation rate of scale enhancement block, default 4')
    return parser.parse_args()

if __name__ == '__main__':
    main()


