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
import re
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
from model_vdsr_v2 import Autoencoders, SmallVDSR_16x, VDSR, SmallVDSR_F8, VDSR_F64B6
from dataset import DatasetFromHdf5 # vdsr data loader

# Passed-in params
parser = argparse.ArgumentParser(description="VDSR Compression")
parser.add_argument('-m', '--mode', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--prune_mode', type=str)
parser.add_argument('--prune_index', type=str)
opt = parser.parse_args()

prune_ratio = {
"conv1": 0,
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

prune_index = {}
tmp = opt.prune_index.split("-")
for t in tmp:
  layer, index = t.split(":")
  kk = []
  index = [int(ix) for ix in re.split("[\[,\]]", index) if ix.isdigit()]
  prune_index[layer] = index
print(prune_index)

if opt.mode in ["F16", "16x"]:
  model = SmallVDSR_16x(opt.model)
elif opt.mode == "original":
  model = VDSR(opt.model)
elif opt.mode == "F8":
  model = SmallVDSR_F8(opt.model)
elif opt.mode == "F64B6":
  model = VDSR_F64B6(opt.model)
dict_param = dict(model.named_parameters())
for p in model.named_parameters():
  layer_name, weight = p
  weight = weight.data.cpu().numpy()
  num_filter = weight.shape[0]
  if num_filter == 1: continue
  if opt.prune_mode == "ratio":
    w_sum = np.sum(np.abs(weight.reshape(num_filter, -1)), axis=1)
    print("\n" + "*"*20, layer_name)
    print(np.sort(w_sum))
    order = np.argsort(w_sum)[:prune_ratio[layer_name.split(".weight")[0]]]
    weight[order] = 0
  elif opt.prune_mode == "direct_set":
    layer_name2 = layer_name.split(".weight")[0]
    if layer_name2 in prune_index:
      index = prune_index[layer_name2]
      print(index)
      weight[index] = 0
  dict_param[layer_name].data.copy_(torch.from_numpy(weight))
torch.save(dict_param, "pruned_model.pth")
  