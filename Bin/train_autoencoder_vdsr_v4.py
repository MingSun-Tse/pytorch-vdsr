from __future__ import print_function
import sys
import os
pjoin = os.path.join
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1] # The args MUST has an option "--gpu".
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

"""
@ Date: 2019-01-22
@ Author: Huan Wang (huanw@zju.edu.cn)
v4: improve speed
"""

# Passed-in params
parser = argparse.ArgumentParser(description="VDSR Compression")
parser.add_argument('--train_data', type=str, help='the directory of train images', default="../Data/train_data/train.h5")
parser.add_argument('--test_data', type=str, help='the directory of test images', default="../Data/test_data/Set5_mat")
parser.add_argument('--e1', type=str, help='path of pretrained encoder1', default="model/64filter_192-20181019-0832_E50.pth")
parser.add_argument('--e2', type=str, help='path of pretrained encoder2', default=None)
parser.add_argument('--gpu', type=str, help="which gpu to run on. default is 0", default="0")
parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=128)
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
parser.add_argument("--pic", type=str)
parser.add_argument("--patch_size", type=int, default=7)
parser.add_argument("--pixel_threshold", type=float, default=5./255)
parser.add_argument("--log")
opt = parser.parse_args()


def logprint(some_str):
  print(time.strftime("[%s" % os.getpid() + "-%Y/%m/%d-%H:%M] ") + str(some_str), file=opt.log, flush=True)

def PSNR(pred, gt, shave_border=0):
  height, width = pred.shape[:2]
  pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
  gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
  imdff = pred - gt
  rmse = math.sqrt(np.mean(imdff ** 2))
  return 100 if rmse == 0 else 20 * math.log10(255.0 / rmse)

def covariance(x): # channel x height x width
  x = x.reshape(x.shape[0], -1)
  x = torch.mm(x.t(), x)
  return x

kk = 0
def get_structure_map(residuals, THRESHOLD=opt.pixel_threshold, num_pos=opt.num_pos):
  picked_positions = [] # residuals shape: # [batch, 1, height, width]
  for res in residuals:
    res = res[0]
    h, w = np.where(np.abs(res) > THRESHOLD)
    positions = list(zip(h, w))
    if len(positions) >= num_pos:
      rand = np.random.permutation(len(positions))[:num_pos]
      positions = np.array(positions)
      picked_positions.append(positions[rand])
    elif 0 < len(positions) < num_pos:
      k = (num_pos // len(positions)) + 1
      positions = (positions * k)[:num_pos]
      picked_positions.append(positions)
      # global kk; kk+=1; print("wooops kk=%s" % kk)
    else:
      picked_positions.append([(10, 10)] * num_pos)
  # print("picked pos shape:", np.shape(picked_positions))
  return np.array(picked_positions) # [batch, num_pos, 2]
  
  
def get_local_structure_loss(structure_maps, fms1, fms2, loss_func):
  """
    structure_maps -- shape: [batch, num_pos, 2], ndarray
    feature_maps   -- shape: [batch, channel, height, width], Tensor
    out            -- shape: [batch, num_pos, patch_size*patch_size, patch_size*patch_size)
  """
  margin = int((opt.patch_size - 1) / 2)
  batch_size = structure_maps.shape[0]
  out1 = torch.zeros([batch_size, opt.num_pos, opt.patch_size*opt.patch_size, opt.patch_size*opt.patch_size]).cuda()
  out2 = torch.zeros([batch_size, opt.num_pos, opt.patch_size*opt.patch_size, opt.patch_size*opt.patch_size]).cuda()
  for i in range(batch_size):
    smap, fm1, fm2 = structure_maps[i], fms1[i], fms2[i]
    # print("fm1 shape before padding", fm1.shape)
    fm1 = nn.functional.pad(fm1, pad=(margin, margin, margin, margin)) # padding zero
    fm2 = nn.functional.pad(fm2, pad=(margin, margin, margin, margin)) # padding zero
    # print("fm1 shape after padding", fm1.shape)
    for j in range(opt.num_pos):
      h, w = smap[j]
      out1[i, j] = covariance(fm1[:, h:h+2*margin+1, w:w+2*margin+1])
      out2[i, j] = covariance(fm2[:, h:h+2*margin+1, w:w+2*margin+1])
  # out1 = nn.functional.normalize(out1)
  # out2 = nn.functional.normalize(out2)
  loss = loss_func(out1, out2.data)
  return loss
  

def test(model, epoch=-1, step=-1):
  scales = [2, 3, 4]
  image_list = glob.glob(opt.test_data + "/*.*")
  for scale in scales:
    avg_psnr_predicted = 0.0
    avg_psnr_bicubic = 0.0
    avg_elapsed_time = 0.0
    count = 0.0
    for image_name in image_list:
      if str(scale) in image_name:
        count += 1
        # logprint("Processing %s" % image_name)
        im_gt_y = sio.loadmat(image_name)['im_gt_y']
        im_b_y = sio.loadmat(image_name)['im_b_y']
                   
        im_gt_y = im_gt_y.astype(float)
        im_b_y = im_b_y.astype(float)

        psnr_bicubic = PSNR(im_gt_y, im_b_y, shave_border=scale)
        avg_psnr_bicubic += psnr_bicubic

        im_input = im_b_y / 255.
        im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])
        im_input = im_input.cuda()
        
        HR = model.e2(im_input) + im_input
        im_h_y = HR.data[0].cpu().numpy().astype(np.float32)
        im_h_y = im_h_y * 255.
        im_h_y[im_h_y < 0] = 0; im_h_y[im_h_y > 255.] = 255.
        im_h_y = im_h_y[0,:,:]

        psnr_predicted = PSNR(im_gt_y, im_h_y, shave_border=scale)
        avg_psnr_predicted += psnr_predicted

    logprint("E{}S{} Scale{} PSNR_predicted = {:.4f} PSNR_bicubic = {:.4f}".format(epoch, step, scale, avg_psnr_predicted/count, avg_psnr_bicubic/count))
   
def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def train(training_data_loader, optimizer, model, loss_func, epoch):
    lr = adjust_learning_rate(epoch)
  
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    logprint("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))
    
    model.train()
    layer_ploss_weight = [float(x) for x in opt.layer_ploss_weight.split("-")]
    loss_func2 = nn.MSELoss()
    for step, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        input = input.cuda() # LR: [batch size, height, width]
        target = target.cuda() # GT
        # -----------------------------------------------------
        # t0 = time.time()
        feats_F64, feats_F16 = model(input)
        # print("inference: {:.3f}".format(time.time() - t0))
        residual_batch = (feats_F16[-1] + input - target).data.cpu().numpy()
        struct_map_batch = get_structure_map(residual_batch)
        # print("get positions: {:.3f}".format(time.time() - t0))
        
        ploss_2  = get_local_structure_loss(struct_map_batch, feats_F16[2],  feats_F64[2],  loss_func2) * 0.5
        ploss_6  = get_local_structure_loss(struct_map_batch, feats_F16[6],  feats_F64[6],  loss_func2) * 5
        ploss_10 = get_local_structure_loss(struct_map_batch, feats_F16[10], feats_F64[10], loss_func2) * 5e3
        ploss_14 = get_local_structure_loss(struct_map_batch, feats_F16[14], feats_F64[14], loss_func2) * 5e6
        ploss_18 = get_local_structure_loss(struct_map_batch, feats_F16[18], feats_F64[18], loss_func2) * 5e8
        # print("get ploss: {:.3f}".format(time.time() - t0))
        iloss = loss_func(feats_F16[-1]+input, target.data) * opt.iloss_weight
        loss  = ploss_2 + ploss_6 + ploss_10 + ploss_14 + ploss_18 + iloss
        # -----------------------------------------------------
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), opt.clip) 
        optimizer.step()
        # print("backward: {:.3f}".format(time.time() - t0))
        if step % SHOW_INTERVAL == 0:
          global t1          
          format_str = "E{}S{} loss: {:.1f} | iloss: {:.3f} | {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} ({:.1f}s/step)"
          logprint(format_str.format(epoch, step, loss.data.cpu().numpy(), iloss.data.cpu().numpy(), \
            ploss_2, ploss_6, ploss_10, ploss_14, ploss_18, (time.time()-t1)/SHOW_INTERVAL))          
          t1 = time.time()
        if step % 100 == 0:
          test(model, epoch, step)

def save_checkpoint(ae, epoch, TIME_ID, weights_path):
  model_index = 0
  for model in [ae.e1, ae.e2]:
    model_index += 1
    if not model.fixed:
      torch.save(model.state_dict(), pjoin(weights_path, "%s_%s_E%s.pth" % (TIME_ID, opt.mode, epoch)))

SHOW_INTERVAL = 2
t1 = 0
if __name__ == "__main__":
  # Set up data
  train_set = DatasetFromHdf5(opt.train_data)
  training_data_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=opt.batch_size, shuffle=True) # 'num_workers' need to be 1, otherwise will cause read error.
  
  # Set up directories and logs etc
  if opt.debug:
    opt.project = "test" # debug means it's just a test demo
  project_path = pjoin("../Experiments", opt.project)
  rec_img_path = pjoin(project_path, "reconstructed_images")
  weights_path = pjoin(project_path, "weights") # to save torch model
  if not opt.resume:
    if os.path.exists(project_path):
      respond = "Y" # input("The appointed project name has existed. Do you want to overwrite it (everything inside will be removed)? (y/n) ")
      if str.upper(respond) in ["Y", "YES"]:
        shutil.rmtree(project_path)
      else:
        exit(1)
    if not os.path.exists(rec_img_path):
      os.makedirs(rec_img_path)
    if not os.path.exists(weights_path):
      os.makedirs(weights_path)
  TIME_ID = os.environ["SERVER"] + time.strftime("-%Y%m%d-%H%M")
  log_path = pjoin(weights_path, "log_" + TIME_ID + ".txt")
  opt.log = sys.stdout if opt.debug else open(log_path, "w+")
  logprint("===> Use gpu id: {}".format(opt.gpu))

  # Set up model
  model = Autoencoders[opt.mode](opt.e1, opt.e2)
  model.cuda()

  # print setting for later check
  logprint(str(opt._get_kwargs()))
  
  # get previous step
  previous_epoch = 0
  previous_step = 0
  previous_total_step = 0
  if opt.e2 and opt.resume:
    previous_epoch = int(os.path.basename(opt.e2).split("_")[-1].split("E")[1].split("S")[0]) # the name of model must end with "_ExxSxx.xx"
    previous_step = int(os.path.basename(opt.e2).split("_")[-1].split("S")[1].split(".")[0])
    previous_total_step = previous_epoch * num_step_per_epoch + previous_step

  # Optimize
  optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
  loss_func = nn.MSELoss(reduction="sum") # old: nn.MSELoss(size_average=False)
  global t1; t1 = time.time()
  test(model) # initial test
  for epoch in range(opt.epoch):
    train(training_data_loader, optimizer, model, loss_func, epoch)
    save_checkpoint(model, epoch, TIME_ID, weights_path)
    test(model, epoch, "end")
    