import argparse
import os
import sys
import torch
import torch.nn as nn
from collections import OrderedDict

def parse_args():
	parser = argparse.ArgumentParser('Translation the caffe model to pytorch')
	parser.add_argument('--caffe_root', required=True, type=str,
        help='the caffe root')
	parser.add_argument('--caffe_model', required=True, type=str,
		help='the dir of caffemodel')
	parser.add_argument('--caffe_proto', required=True, type=str,
		help='the filepath of prototxt')
	return parser.parse_args()

def bulid(net):
    new_net = []
    for i, x in enumerate(net._layer_names):
    	if net.layers[i].type == 'Convolution':
    		shape = net.params[x][0].shape
    		new_net.append((x, nn.Conv2d(shape[1], shape[0], (shape[2], shape[3]), stride=1, padding=1))) 
    	elif net.layers[i].type == 'ReLU':
    		new_net.append((x, nn.ReLU(inplace=True)))
    	elif net.layers[i].type == 'Pooling':
    		new_net.append((x, nn.MaxPool2d(2, stride=2)))
    	elif net.layers[i].type == 'Dropout':
    		new_net.append((x, nn.Dropout(inplace=True)))
    model = nn.Sequential(OrderedDict(new_net))
    for name, param in model.state_dict().items():
    	nm = name.split('.')
    	if 'weight' in name:
    		param.copy_(torch.from_numpy(net.params[nm[0]][0].data).float())
    	else:
    		param.copy_(torch.from_numpy(net.params[nm[0]][1].data).float())
    return model
    
def main():
	args = parse_args()
	sys.path.append(args.caffe_root)
	import caffe
	net = caffe.Net(args.caffe_proto, args.caffe_model, caffe.TEST)
	print dir(net.layers[1].blobs[0])
	# for i, x in enumerate(net._layer_names):
	# 	print x, net.layers[i].type,
	# 	if x in net.params:
	# 		print net.params[x][0].shape
	# 	print '\n'
	model = bulid(net)
	torch.save(model.state_dict(), args.caffe_proto.split('.')[0]+'.pth')
	f = open(args.caffe_proto.split('.')[0]+'.py', 'w')
	stdout = sys.stdout
	sys.stdout = f
	print 'model = ', model
	sys.stdout = stdout
	f.close()

if __name__ == '__main__':
	main()
