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
pjoin = os.path.join


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
parser.add_argument("-m", "--mode", default="")
parser.add_argument("--in_gt_img")
parser.add_argument("--in_lr_img")
parser.add_argument("--out_hr_img")
opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

if opt.mode:
  assert(opt.model != "")
  model = SmallVDSR_16x(opt.model)
  # model = VDSR(opt.model) # test big model
else:
  model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]

im_gt_ycbcr = imread(opt.in_gt_img, mode="YCbCr") # HR baseline
im_b_ycbcr  = imread(opt.in_lr_img, mode="YCbCr") # LR

im_gt_y = im_gt_ycbcr[:,:,0].astype(float)
im_b_y = im_b_ycbcr[:,:,0].astype(float)
visualize_luminace(im_gt_y.astype(np.uint8), "luminance_gt.png")
visualize_luminace(im_b_y.astype(np.uint8), "luminance_bi.png")

psnr_bicubic = PSNR(im_gt_y, im_b_y, shave_border=opt.scale)

im_input = im_b_y/255.

im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

if cuda:
    model = model.cuda()
    im_input = im_input.cuda()
else:
    model = model.cpu()

start_time = time.time()
out = torch.add(model(im_input), im_input) if opt.mode else model(im_input) # compressed model only output the residual
elapsed_time = time.time() - start_time

out = out.cpu()
im_h_y = out.data[0].numpy().astype(np.float32)

im_h_y = im_h_y * 255.
im_h_y[im_h_y < 0] = 0
im_h_y[im_h_y > 255.] = 255.
visualize_luminace(im_h_y[0,:,:].astype(np.uint8), "luminance_hr.png")

psnr_predicted = PSNR(im_gt_y, im_h_y[0,:,:], shave_border=opt.scale)

im_h = colorize(im_h_y[0,:,:], im_b_ycbcr)
im_gt = Image.fromarray(im_gt_ycbcr, "YCbCr").convert("RGB")
im_b = Image.fromarray(im_b_ycbcr, "YCbCr").convert("RGB")
im_h.save(opt.out_hr_img)

print("Scale=", opt.scale)
print("PSNR_predicted=", psnr_predicted)
print("PSNR_bicubic=", psnr_bicubic)
print("It takes {}s for processing".format(elapsed_time))

fig = plt.figure()
ax = plt.subplot("131")
ax.imshow(im_gt)
ax.set_title("GT")

ax = plt.subplot("132")
ax.imshow(im_b)
ax.set_title("Input(bicubic)")

ax = plt.subplot("133")
ax.imshow(im_h)
ax.set_title("Output(vdsr)")

TIMEID = time.strftime("%Y%m%d-%H%M")
plt.savefig("result/" + TIMEID + "_output.png")
