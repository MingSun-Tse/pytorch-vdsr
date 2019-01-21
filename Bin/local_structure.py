import argparse, os
import torch
from torch.autograd import Variable
from scipy.ndimage import imread
from PIL import Image
import numpy as np
import time, math
import matplotlib
matplotlib.use("Agg")
from matplotlib.pyplot import imsave
import matplotlib.pyplot as plt
from model_vdsr import SmallVDSR_16x, VDSR
import cv2
import threading # plot figures using multi-threading
from multiprocessing import Process, Lock, Pool
pjoin = os.path.join


parser = argparse.ArgumentParser(description="Local Structure Visualization")
parser.add_argument("--in_gt_img")
parser.add_argument("--in_lr_img")
parser.add_argument("--out_dir", type=str, default="./")
parser.add_argument("--fm_F16", type=str)
parser.add_argument("--fm_F64", type=str)
parser.add_argument("--num_pos", type=int, default=64)
parser.add_argument("--pic", type=str)
parser.add_argument("--feature_dir", type=str, default="./Feature")
parser.add_argument("--filter_size", type=int, default=5)
parser.add_argument("--pixel_threshold", type=float, default=50./255)
parser.add_argument("--layer_index", type=int)
parser.add_argument("--structure_maps", type=str)
parser.add_argument("--picked_pos", type=str)

opt = parser.parse_args()


def covariance(x): # batch x channel x height x width
  channel, height, width = x.shape
  x = x.reshape(channel, height*width)
  x = np.matmul(x.T, x)
  return x
  
def get_structure_map(residuals, THRESHOLD=opt.pixel_threshold):
  assert(len(residuals.shape) == 3) # [batch, height, width], because of only using Y channel, there is no dimension for channel.
  structure_maps = []
  for res in residuals:
    h, w = np.where(np.abs(res) > THRESHOLD)
    structure_maps.append(list(zip(h, w)))
    res[np.abs(res) <= THRESHOLD] = 0
  return np.array(structure_maps), residuals
  
def get_local_structure(structure_maps, feature_maps):
  """
    structure_maps -- shape: [batch, some, 2]
    feature_maps   -- shape: [batch, channel, height, width]
    out            -- shape: [batch, some, filter_size*filter_size, filter_size*filter_size)
  """
  # print("smap shape: %s, fmap shape: %s" % (structure_maps.shape, feature_maps.shape))
  out = []
  margin = int((opt.filter_size - 1) / 2)
  for smap, fmap in zip(structure_maps, feature_maps):
    fmap = np.pad(fmap, ((0, 0), (margin, margin), (margin, margin)), "reflect") # padding
    # ------------------------------------------------------------
    covar = []
    for h, w in smap:
      fm_patch = fmap[:, h:h+2*margin+1, w:w+2*margin+1]
      covar.append(covariance(fm_patch)) # covar for an example
    # covar = [covariance(fmap[:, x:x+2*margin+1, y:y+2*margin+1]) for (x,y) in smap] # another way, not faster...
    # ------------------------------------------------------------
    out.append(covar) # covar for a batch
  return np.array(out)


def visualize(struct1, struct2, picked_pos, layer_index, struct_map, LR):
  cnt = 0
  cmap = matplotlib.cm.jet
  margin = int((opt.filter_size - 1) / 2)
  title_fs = 20
  sup_title_fs = 50
  norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
  for p in picked_pos:
    print("plotting %s / %s (layer %s)" % (cnt+1, len(picked_pos), layer_index)); cnt += 1
    struct1[p], struct2[p] = struct1[p]/struct1[p].max(), struct2[p]/struct2[p].max() # normalized to [0,1]
    h, w = struct_map[p]  
    
    fig = plt.figure(layer_index, figsize=(22, 20))
    ax00 = fig.add_subplot(2,2,1)
    ax01 = fig.add_subplot(2,2,2)
    ax10 = fig.add_subplot(2,2,3)
    ax11 = fig.add_subplot(2,2,4)
    # fig, ax = plt.subplots(2, 2, figsize=(22, 20))
    
    im0 = ax00.imshow(struct1[p], cmap=cmap, norm=norm)
    im1 = ax01.imshow(struct2[p], cmap=cmap, norm=norm)
    im2 = ax10.imshow(LR[h-margin : h+margin+1, w-margin : w+margin+1], cmap="gray", norm=norm)
    im3 = ax11.imshow(LR, cmap="gray", norm=norm)
    ax11.plot(w,h, marker="x", color="r", ms=20, mew=4) # Note that the coordinate order for "plot" (horizon, vertical) is opposite to that of "imshow" (height, width).
    
    # ax00.set_title("F16", size=title_fs)
    # ax01.set_title("F64", size=title_fs)
    # ax10.set_title("LR (zoomed in)", size=title_fs)
    # ax11.set_title("LR", size=title_fs)
    # fig.suptitle("Layer {} Position {}: (h={}, w={})".format(layer_index, p, h, w), size=sup_title_fs)
    
    # fig.colorbar(im0, ax=ax00, fraction=0.046, pad=0.04)
    # fig.colorbar(im1, ax=ax01, fraction=0.046, pad=0.04)
    # fig.colorbar(im2, ax=ax10, fraction=0.046, pad=0.04)
    # fig.colorbar(im3, ax=ax11, fraction=0.046, pad=0.04)
    fig.savefig("{}/position{}_layer{}.png".format(opt.out_dir, p, layer_index), bbox_inches="tight")
    plt.close(layer_index)


def get_local_structure_and_visualize(layer_index, structure_maps, picked_pos, im_b_y):
  print("===> Visualizing layer %s (pid = %s)" % (layer_index, os.getpid()))
  fm_F16 = pjoin(opt.feature_dir, "%s_F16_layer%s.npy" % (opt.pic, layer_index))
  fm_F64 = pjoin(opt.feature_dir, "%s_F64_layer%s.npy" % (opt.pic, layer_index))
  struct_F16 = get_local_structure(structure_maps, np.load(fm_F16))
  struct_F64 = get_local_structure(structure_maps, np.load(fm_F64))
  assert(struct_F16.shape == struct_F64.shape)
  visualize(struct_F16[0], struct_F64[0], picked_pos, layer_index, structure_maps[0], im_b_y)
  
    

# Multi-threading. Seems not faster. ----------------------------------------------  
# class PlotThread(threading.Thread):
  # def __init__(self, threadID, name, layer_index, picked_pos, structure_maps):
    # threading.Thread.__init__(self)
    # self.threadID = threadID
    # self.name = name
    # self.layer_index = layer_index
    # self.picked_pos = picked_pos
    # self.structure_maps = structure_maps
    
  # def run(self):
    # print("===> start %s: visualizing layer %s" % (self.name, self.layer_index))
    # fm_F16 = pjoin(opt.feature_dir, "%s_F16_layer%s.npy" % (opt.pic, self.layer_index))
    # fm_F64 = pjoin(opt.feature_dir, "%s_F64_layer%s.npy" % (opt.pic, self.layer_index))
    # struct_F16 = get_local_structure(self.structure_maps, np.load(fm_F16))
    # struct_F64 = get_local_structure(self.structure_maps, np.load(fm_F64))
    # assert(struct_F16.shape == struct_F64.shape)
    # visualize(struct_F16[0], struct_F64[0], self.picked_pos, self.layer_index, self.structure_maps[0], im_b_y)
    # print("===> close %s: visualizing layer %s" % (self.name, self.layer_index))
# ---------------------------------------------------------------------------------
  
if __name__ == "__main__":
  opt.in_gt_img = "../Data/test_data/Set5/%s_GT.bmp" % opt.pic
  opt.in_lr_img = "../Data/test_data/Set5/%s_GT_scale_4.bmp" % opt.pic

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
  
  # Multi-threading. Seems not faster. ----------------------------------------------  
  # for layer_index in range(num_layer):
    # thread = PlotThread(layer_index, "Thread-%s" % layer_index, layer_index, picked_pos, structure_maps)
    # thread.start()
  # ---------------------------------------------------------------------------------
  
  num_layer = 20
  t1 = time.time()
  # ----------------------------------------------------------------------------------
  # Multi-processing (1) -- using Pool. The text part (title) in the plot is strange.
  print("===> Parent pid = %s" % os.getpid())
  p = Pool(num_layer)
  for layer_index in range(num_layer):
    p.apply_async(get_local_structure_and_visualize, args=(layer_index, structure_maps, picked_pos, im_b_y))
  p.close()
  p.join()
  # # Multi-processing (2) -- using Process.
  # for layer_index in range(num_layer):
    # p = Process(target=get_local_structure_and_visualize, args=(layer_index, structure_maps, picked_pos, im_b_y))
    # p.start()
  # p.join()
  # ----------------------------------------------------------------------------------
  t2 = time.time()
  print("It takes {:.4f}s for all visualization".format(t2-t1))
    
  


    
    