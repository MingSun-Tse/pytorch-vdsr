import argparse, os
import torch
from torch.autograd import Variable
from scipy.ndimage import imread
from PIL import Image
import numpy as np
import time, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave
from model_vdsr import SmallVDSR_16x, VDSR
import cv2
import threading # plot figures using multi-threading
from multiprocessing import Process
from multiprocessing import Pool
pjoin = os.path.join


parser = argparse.ArgumentParser(description="Local Structure Visualization")
parser.add_argument("--in_gt_img")
parser.add_argument("--in_lr_img")
parser.add_argument("--out_dir", type=str, default="./")
parser.add_argument("--fm_F16", type=str)
parser.add_argument("--fm_F64", type=str)
parser.add_argument("--num_pos", type=int, default=64)
parser.add_argument("--pic", type=str)
parser.add_argument("--feature_dir", type=str)
parser.add_argument("--filter_size", type=int, default=5)
parser.add_argument("--pixel_threshold", type=float, default=50./255)
opt = parser.parse_args()

def get_structure_map(residuals, THRESHOLD=opt.pixel_threshold):
  assert(len(residuals.shape) == 3) # [batch, height, width], because of only using Y channel, there is no dimension for channel.
  structure_maps = []
  for res in residuals:
    h, w = np.where(np.abs(res) > THRESHOLD)
    structure_maps.append(list(zip(h, w)))
    res[np.abs(res) <= THRESHOLD] = 0
  return np.array(structure_maps), residuals

# Get structure_maps
im_gt_ycbcr = imread(opt.in_gt_img, mode="YCbCr") # GT
im_b_ycbcr  = imread(opt.in_lr_img, mode="YCbCr") # LR
im_gt_y = im_gt_ycbcr[:, :, 0].astype(float) / 255.
im_b_y  = im_b_ycbcr[:, :, 0].astype(float) / 255.
residual = im_gt_y - im_b_y
residual = residual[np.newaxis, :] # add a dimension for batch
structure_maps, residual_thresholded = get_structure_map(residual) # shape: [batch, some, 2]. 2: coordinate x and y
cnt = 0
cmap = matplotlib.cm.jet
for res in residual_thresholded:
  plt.imshow(res)
  plt.colorbar()
  plt.savefig("residual_thresholded_{}.png".format(cnt))
  cnt += 1
  
if not os.path.exists(opt.out_dir):
  os.makedirs(opt.out_dir)
  
assert(opt.num_pos <= structure_maps.shape[1])
picked_pos = np.random.permutation(structure_maps.shape[1])[:opt.num_pos]
np.save("structure_maps.npy", structure_maps)
np.save("picked_pos.npy", picked_pos)

num_layer = 20
for layer_index in range(num_layer):
  script = "python local_structure.py --in_lr_img ../Data/test_data/Set5/butterfly_GT_scale_4.bmp \
                                      --in_gt_img ../Data/test_data/Set5/butterfly_GT.bmp \
                                      --feature_dir Feature \
                                      --num_pos 10 \
                                      --structure_maps structure_maps.npy \
                                      --picked_pos picked_pos.npy \
                                      --out_dir LocalStructVisualization --pic butterfly --layer_index " + str(layer_index)
  os.system(script)