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
pjoin = os.path.join

def sharpen(in_image):
    k = -0.4 # original k = -1. The larger abs(k), the sharp the result.
    kernel = np.array([[0, k, 0], [k, -4*k+1, k], [0, k, 0]], np.float32) # sharpen
    dst = cv2.filter2D(in_image, -1, kernel=kernel)
    return dst

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)
    
def colorize(y, ycbcr): 
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycbcr[:,:,1]
    img[:,:,2] = ycbcr[:,:,2]
    img = Image.fromarray(img, "YCbCr").convert("RGB")
    return img

def visualize_luminace(y, save_path=None):
    img = Image.fromarray(y)
    if save_path:
      img.save(save_path)
    else:
      imshow(img)

parser = argparse.ArgumentParser(description="PyTorch VDSR Demo")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_epoch_50.pth", type=str, help="model path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("-m", "--mode", type=str)
parser.add_argument("--in_gt_img")
parser.add_argument("--in_lr_img")
parser.add_argument("--out_hr_img")
parser.add_argument("--sharpen", action="store_true")
parser.add_argument("--num_stage", type=int, default=1)
parser.add_argument("--out_dir", type=str)
parser.add_argument("--save_feature", action="store_true")
opt = parser.parse_args()
cuda = opt.cuda

if cuda:
  print("=> use gpu id: '{}'".format(opt.gpus))
  os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
  if not torch.cuda.is_available():
    raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

if opt.mode:
  assert(opt.model != "")
  if opt.mode == "16x":
    model = SmallVDSR_16x(opt.model)
  elif opt.mode == "original":
    model = VDSR(opt.model)
else:
  model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]

im_gt_ycbcr = imread(opt.in_gt_img, mode="YCbCr") # Ground Truth
im_b_ycbcr  = imread(opt.in_lr_img, mode="YCbCr") # LR

im_gt_y = im_gt_ycbcr[:,:,0].astype(float)
im_b_y = im_b_ycbcr[:,:,0].astype(float)
psnr_bicubic = PSNR(im_gt_y, im_b_y, shave_border=opt.scale)

im_input = im_b_y/255.
im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])
if cuda:
  model = model.cuda()
  im_input = im_input.cuda()
else:
  model = model.cpu()
  
def feature_display(fms, mark, save_feature=False):
  # Save
  if save_feature:
    if opt.mode == "16x":
      model_mark = "F16"  
    elif opt.mode == "original":
      model_mark = "F64"
    else:
      print("mode wrong")
      exit(1)
    num_layer = len(fms)
    for i in range(num_layer):
      np.save("%s_%s_layer%s.npy" % (mark, model_mark, i), fms[i].cpu().data.numpy())
    return
  
  # Plot  
  if not os.path.exists(opt.out_dir):
    os.mkdir(opt.out_dir)
  plt.rcParams['figure.dpi'] = 200
  norm_feat = matplotlib.colors.Normalize(vmin=0, vmax=1.6)
  norm_resi = matplotlib.colors.Normalize(vmin=-0.5, vmax=0.5)
  cmap = matplotlib.cm.jet
  for cnt in range(len(fms)):
    norm = norm_resi if cnt == len(fms)-1 else norm_feat
    print("visualizing layer %s" % cnt)
    fm = fms[cnt][0].cpu().data.numpy()
    num_channel = fm.shape[0]
    for i in range(num_channel):
      channel = fm[i]
      plt.imshow(channel, cmap=cmap, norm=norm)
      plt.colorbar()
      plt.title("layer{}_fm{}.png".format(cnt, i))
      plt.savefig("{}/layer{}_fm{}_{}.png".format(opt.out_dir, cnt, i, mark))
      plt.close("all")

######### Inference
start_time = time.time()
for _ in range(opt.num_stage):
  # im_input = torch.add(model(im_input), im_input) if opt.mode else model(im_input) # compressed model only output the residual
  fms = model.forward_dense(im_input)
  mark = os.path.basename(opt.in_lr_img).split("_")[0] # example: xx/butterfly_GT_scale_4.bmp
  feature_display(fms, mark, opt.save_feature)

out = im_input
elapsed_time = time.time() - start_time
###################

im_h_y = out.cpu().data[0].numpy().astype(np.float32)
im_h_y = im_h_y * 255.
im_h_y[im_h_y < 0] = 0
im_h_y[im_h_y > 255.] = 255.

# Get new PSNR
psnr_predicted = PSNR(im_gt_y, im_h_y[0,:,:], shave_border=opt.scale)

# Put Y channel back into a color image and save it
im_h = colorize(im_h_y[0,:,:], sharpen(im_b_ycbcr)) if opt.sharpen else colorize(im_h_y[0,:,:], im_b_ycbcr)
im_h.save(opt.out_hr_img)

print("Scale {} num_stage {}: PSNR_predicted = {:.4f} PSNR_bicubic = {:.4f} Processing time {:.3f}s".format(opt.scale, opt.num_stage, psnr_predicted, psnr_bicubic, elapsed_time))