from __future__ import print_function
import sys
import os
pjoin = os.path.join
import shutil
import time
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob
import scipy.io as sio
import math
# torch
import torch
import torch.nn as nn
from torch.utils.serialization import load_lua
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
# my libs
from model_vdsr_v2 import Autoencoders
from dataset import DatasetFromHdf5 # vdsr data loader

# Passed-in params
parser = argparse.ArgumentParser(description="VDSR Compression")
parser.add_argument('--train_data', type=str, help='the directory of train images', default="../Data/train_data/train.h5")
parser.add_argument('--test_data', type=str, help='the directory of test images', default="../Data/test_data/Set5_mat")
parser.add_argument('--e1', type=str, help='path of pretrained encoder1', default="model/64filter_192-20181019-0832_E50.pth")
parser.add_argument('--e2', type=str, help='path of pretrained encoder2', default=None)
parser.add_argument('--gpu', type=str, help="which gpu to run on. default is 0", default="0")
parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=4)
parser.add_argument('--lr', type=float, help='learning rate', default=0.1)
parser.add_argument('--ploss_weight', type=float, help='loss weight to balance multi-losses', default=1.0)
parser.add_argument('--iloss_weight', type=float, help='loss weight to balance multi-losses', default=1.0)
parser.add_argument('--layer_ploss_weight', type=str, default="1-0.01-0.1-1-100") # It will be parsed by sep "-".
parser.add_argument('-p', '--project', type=str, default="test", help='the name of project, to save logs etc., will be set in the directory "Experiments"')
parser.add_argument('-m', '--mode', type=str, help='the training mode name.')
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--resume', action="store_true")
parser.add_argument("--num_filter", default=64, type=int)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--num_pos", type=int, default=8)
parser.add_argument("--patch_size", type=int, default=7)
parser.add_argument("--pixel_threshold", type=float, default=5./255)
parser.add_argument("--log")
parser.add_argument("--TIME_ID")
opt = parser.parse_args()

prune_ratio = {
"conv1": 11,
"conv2": 0,
"conv3": 0,
"conv4": 0,
"conv5": 0,
"conv6": 0,
"conv7": 0,
"conv8": 0,
"conv9": 0,
"conv10": 0,
"conv11": 0,
"conv12": 0,
"conv13": 0,
"conv14": 0,
"conv15": 0,
"conv16": 0,
"conv17": 0,
"conv18": 0,
"conv19": 0,
}
model = Autoencoders[opt.mode](opt.e1, opt.e2).e2
dict_param = dict(model.named_parameters())
for p in model.named_parameters():
  layer_name, weight = p
  weight = weight.data.cpu().numpy()
  num_filter = weight.shape[0]
  if num_filter == 1: continue
  w_sum = np.sum(np.abs(weight.reshape(num_filter, -1)), axis=1)
  print("\n" + "*"*20, layer_name)
  print(np.sort(w_sum))
  order = np.argsort(w_sum)[:prune_ratio[layer_name.split(".weight")[0]]]
  weight[order] = 0
  dict_param[layer_name].data.copy_(torch.from_numpy(weight))
torch.save(dict_param, "pruned_model.pth")
  