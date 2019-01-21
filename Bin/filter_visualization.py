import argparse, os
import torch
from torch.autograd import Variable
from scipy.ndimage import imread
import numpy as np
import time, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from model_vdsr import SmallVDSR_16x, VDSR
import cv2
pjoin = os.path.join
import math

def visualize_filter(model, out_dir):
  plt.rcParams['figure.dpi'] = 50
  norm = matplotlib.colors.Normalize(vmin=-0.2, vmax=0.2)
  cmap = matplotlib.cm.jet
  for layer in model.named_parameters():
    tensor_name = layer[0] # e.g., "conv1.weight"
    # if int(tensor_name.split(".")[0].split("conv")[1]) < 15: continue # use this to generate visualization for some specific layers
    conv_weight = layer[1].data.cpu().numpy() # npy, shape example = [filter_out, filter_in, height, width]
    print("visualizing layer '{}', shape = {}".format(tensor_name, conv_weight.shape))
    if len(conv_weight.shape) != 4: continue
    num_filter, num_channel, height, width = conv_weight.shape
    if height != 3 or width !=3: continue # only consider the layers of filter size 3x3
    for i in range(num_filter):
      if num_channel == 1:
        plt.imshow(conv_weight[i, 0], cmap=cmap, norm=norm)
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)
        title_font_size = 20
      else:
        fig, ax = plt.subplots(max(num_channel//8, 1), 8, figsize=(112, 32))
        title_font_size = 50
        for j in range(num_channel):
          current_ax = ax[j//8, j%8]
          im = current_ax.imshow(conv_weight[i, j], cmap=cmap, norm=norm)
          current_ax.get_xaxis().set_visible(False); current_ax.get_yaxis().set_visible(False)
          fig.colorbar(im, ax=current_ax, fraction=0.046, pad=0.04)
      plt.suptitle("{}_filter{}".format(tensor_name, i), size=title_font_size)
      plt.savefig("{}/{}_filter{}.png".format(out_dir, tensor_name, i), bbox_inches='tight')
      plt.close('all')

def main():
  parser = argparse.ArgumentParser(description="PyTorch VDSR Filter Visualization")
  parser.add_argument("--model", default="model/model_epoch_50.pth", type=str, help="model path")
  parser.add_argument("-m", "--mode", type=str)
  parser.add_argument("--out_dir", type=str)
  opt = parser.parse_args()
  
  # Load model
  if opt.mode:
    assert(opt.model != "")
    if opt.mode == "16x":
      model = SmallVDSR_16x(opt.model)
    elif opt.mode == "original":
      model = VDSR(opt.model)
  else:
    model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]
  
  # Visualize
  if not os.path.exists(opt.out_dir):
    os.mkdir(opt.out_dir)
  visualize_filter(model, opt.out_dir)

if __name__ == "__main__":
  main()
