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
from model_vdsr import Autoencoders
from dataset import DatasetFromHdf5 # vdsr data loader

def logprint(some_str, f=sys.stdout):
  print(time.strftime("[%s" % os.getpid() + "-%Y/%m/%d-%H:%M] ") + str(some_str), file=f, flush=True)

def PSNR(pred, gt, shave_border=0):
  height, width = pred.shape[:2]
  pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
  gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
  imdff = pred - gt
  rmse = math.sqrt(np.mean(imdff ** 2))
  return 100 if rmse == 0 else 20 * math.log10(255.0 / rmse)

def covariance(x): # batch x channel x height x width
  batch, channel, height, width = x.size()
  x = x.view(batch, channel, height*width)
  x = torch.stack([torch.mm(i.t(), i) for i in x])
  print(x.shape)
  return x
  
def get_structure_map(residuals):
  structure_map = []
  for res in residuals:
    x, y = np.where(res > 10.0/255)
    structure_map.append(zip(x, y))
  return structure_map
  
def local_structure(structure_map, residuals, feature_maps, filter_size=5):
  """
  """
  structure_maps = get_structure_map(residuals)
  batch = len(residuals)
  out = []
  for index in range(batch): # feature_maps are in batch
    smap = structure_maps[index]
    convar = []
    for (x, y) in smap:
      fm = feature_maps[index][:, x-filter_size : x+filter_size, y-filter_size : y+filter_size]
      convar.append(covariance(fm))
    out.append(convar)
  return out
  

def test(model, args, log, epoch):
  scales = [2, 3, 4]
  image_list = glob.glob(args.test_data + "/*.*")
  for scale in scales:
    avg_psnr_predicted = 0.0
    avg_psnr_bicubic = 0.0
    avg_elapsed_time = 0.0
    count = 0.0
    for image_name in image_list:
      if str(scale) in image_name:
        count += 1
        # logprint("Processing %s" % image_name, log)
        im_gt_y = sio.loadmat(image_name)['im_gt_y']
        im_b_y = sio.loadmat(image_name)['im_b_y']
                   
        im_gt_y = im_gt_y.astype(float)
        im_b_y = im_b_y.astype(float)

        psnr_bicubic = PSNR(im_gt_y, im_b_y, shave_border=scale)
        avg_psnr_bicubic += psnr_bicubic

        im_input = im_b_y / 255.
        im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])
        im_input = im_input.cuda()
        
        HR = model(im_input)[3].cpu() # index 3 is the predicted HR by small model
        im_h_y = HR.data[0].numpy().astype(np.float32)
        im_h_y = im_h_y * 255.
        im_h_y[im_h_y < 0] = 0
        im_h_y[im_h_y > 255.] = 255.
        im_h_y = im_h_y[0,:,:]

        psnr_predicted = PSNR(im_gt_y, im_h_y, shave_border=scale)
        avg_psnr_predicted += psnr_predicted

    logprint("Epoch {} Scale {} PSNR_predicted = {:.4f} PSNR_bicubic = {:.4f}".format(epoch, scale, avg_psnr_predicted/count, avg_psnr_bicubic/count), log)
   
def adjust_learning_rate(epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.step))
    return lr

def train(training_data_loader, optimizer, model, loss_func, epoch, args, log):
    lr = adjust_learning_rate(epoch, args)
  
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    logprint("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]), log)
    
    model.train()
    ploss1 = ploss2 = ploss3 = ploss4 = ploss5 = torch.FloatTensor(0).cuda()
    layer_ploss_weight = [float(x) for x in args.layer_ploss_weight.split("-")]
    for step, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        input = input.cuda()
        target = target.cuda()
        
        # -----------------------------------------------------
        # deeply-supervised perceptual loss
        feats_1, feats2_1, predictedHR_1, predictedHR2_1, \
        feats_2, feats2_2, predictedHR_2, predictedHR2_2, \
        feats_3, feats2_3, predictedHR_3, predictedHR2_3 = model(input)
        
        ploss1_1 = loss_func(feats2_1[0], feats_1[0].data) * args.ploss_weight * layer_ploss_weight[0]
        ploss2_1 = loss_func(feats2_1[1], feats_1[1].data) * args.ploss_weight * layer_ploss_weight[1]
        ploss3_1 = loss_func(feats2_1[2], feats_1[2].data) * args.ploss_weight * layer_ploss_weight[2]
        ploss4_1 = loss_func(feats2_1[3], feats_1[3].data) * args.ploss_weight * layer_ploss_weight[3]
        ploss5_1 = loss_func(feats2_1[4], feats_1[4].data) * args.ploss_weight * layer_ploss_weight[4]
        
        ploss1_2 = loss_func(feats2_2[0], feats_2[0].data) * args.ploss_weight * layer_ploss_weight[0]
        ploss2_2 = loss_func(feats2_2[1], feats_2[1].data) * args.ploss_weight * layer_ploss_weight[1]
        ploss3_2 = loss_func(feats2_2[2], feats_2[2].data) * args.ploss_weight * layer_ploss_weight[2]
        ploss4_2 = loss_func(feats2_2[3], feats_2[3].data) * args.ploss_weight * layer_ploss_weight[3]
        ploss5_2 = loss_func(feats2_2[4], feats_2[4].data) * args.ploss_weight * layer_ploss_weight[4]
        
        ploss1_3 = loss_func(feats2_3[0], feats_3[0].data) * args.ploss_weight * layer_ploss_weight[0]
        ploss2_3 = loss_func(feats2_3[1], feats_3[1].data) * args.ploss_weight * layer_ploss_weight[1]
        ploss3_3 = loss_func(feats2_3[2], feats_3[2].data) * args.ploss_weight * layer_ploss_weight[2]
        ploss4_3 = loss_func(feats2_3[3], feats_3[3].data) * args.ploss_weight * layer_ploss_weight[3]
        ploss5_3 = loss_func(feats2_3[4], feats_3[4].data) * args.ploss_weight * layer_ploss_weight[4]
        
        HR_iloss_1 = loss_func(predictedHR2_1, target.data) * args.iloss_weight
        HR_iloss_2 = loss_func(predictedHR2_2, target.data) * args.iloss_weight
        HR_iloss_3 = loss_func(predictedHR2_3, target.data) * args.iloss_weight
        GT_iloss   = loss_func(predictedHR2_1, target.data) * args.iloss_weight
        
        # loss = ploss1_3 + ploss2_3 + ploss3_3 + ploss4_3 + ploss5_3
        loss = ploss1_1 + ploss2_1 + ploss3_1 + ploss4_1 + ploss5_1 + \
               ploss1_2 + ploss2_2 + ploss3_2 + ploss4_2 + ploss5_2 + \
               ploss1_3 + ploss2_3 + ploss3_3 + ploss4_3 + ploss5_3 + \
               HR_iloss_1 + HR_iloss_2 + HR_iloss_3 #+ GT_iloss
        # -----------------------------------------------------
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.clip) 
        optimizer.step()

        if step % SHOW_INTERVAL == 0:
          global t1
          # format_str = "E{}S{} loss={:.3f} | iloss={:.5f} | ploss1={:.5f} ploss2={:.5f} ploss3={:.5f} ploss4={:.5f} ploss5={:.5f} ({:.3f}s/step)"
          # logprint(format_str.format(epoch, step, loss.data.cpu().numpy(), iloss.data.cpu().numpy(), ploss1.data.cpu().numpy()), log), ploss2.data.cpu().numpy(),
              # ploss3.data.cpu().numpy(), ploss4.data.cpu().numpy(), ploss5.data.cpu().numpy(), (time.time()-t1)/SHOW_INTERVAL), log)
          
          format_str = "E{}S{} loss={:.3f} | iloss=({:.3f} | {:.3f} {:.3f} {:.3f}) | ploss1=({:.3f} {:.3f} {:.3f}) | \
ploss2=({:.3f} {:.3f} {:.3f}) | ploss3=({:.3f} {:.3f} {:.3f}) | ploss4=({:.3f} {:.3f} {:.3f}) | ploss5=({:.3f} {:.3f} {:.3f}) ({:.2f}s/step)"
          logprint(format_str.format(epoch, step, loss.data.cpu().numpy(), \
              GT_iloss.data.cpu().numpy(), HR_iloss_1.data.cpu().numpy(), HR_iloss_2.data.cpu().numpy(), HR_iloss_3.data.cpu().numpy(), \
              ploss1_1.data.cpu().numpy(), ploss1_2.data.cpu().numpy(), ploss1_3.data.cpu().numpy(), \
              ploss2_1.data.cpu().numpy(), ploss2_2.data.cpu().numpy(), ploss2_3.data.cpu().numpy(), \
              ploss3_1.data.cpu().numpy(), ploss3_2.data.cpu().numpy(), ploss3_3.data.cpu().numpy(), \
              ploss4_1.data.cpu().numpy(), ploss4_2.data.cpu().numpy(), ploss4_3.data.cpu().numpy(), \
              ploss5_1.data.cpu().numpy(), ploss5_2.data.cpu().numpy(), ploss5_3.data.cpu().numpy(), \
              (time.time()-t1)/SHOW_INTERVAL), log)
          t1 = time.time()

def save_checkpoint(ae, epoch, TIME_ID, weights_path, args):
  model_index = 0
  for model in [ae.e1, ae.e2]:
    model_index += 1
    if not model.fixed:
      torch.save(model.state_dict(), pjoin(weights_path, "%s_%s_E%s.pth" % (TIME_ID, args.mode, epoch)))

SHOW_INTERVAL = 100
SAVE_INTERVAL = 1000
t1 = 0
if __name__ == "__main__":
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
  args = parser.parse_args()

  # Set up data
  train_set = DatasetFromHdf5(args.train_data)
  training_data_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=args.batch_size, shuffle=True) # 'num_workers' need to be 1, otherwise will cause read error.
  
  # Set up directories and logs etc
  if args.debug:
    args.project = "test" # debug means it's just a test demo
  project_path = pjoin("../Experiments", args.project)
  rec_img_path = pjoin(project_path, "reconstructed_images")
  weights_path = pjoin(project_path, "weights") # to save torch model
  if not args.resume:
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
  log = sys.stdout if args.debug else open(log_path, "w+")
  logprint("===> use gpu id: {}".format(args.gpu), log)

  # Set up model
  model = Autoencoders[args.mode](args.e1, args.e2)
  model.cuda()

  # print setting for later check
  logprint(str(args._get_kwargs()), log)
  
  # get previous step
  previous_epoch = 0
  previous_step = 0
  previous_total_step = 0
  if args.e2 and args.resume:
    previous_epoch = int(os.path.basename(args.e2).split("_")[-1].split("E")[1].split("S")[0]) # the name of model must end with "_ExxSxx.xx"
    previous_step = int(os.path.basename(args.e2).split("_")[-1].split("S")[1].split(".")[0])
    previous_total_step = previous_epoch * num_step_per_epoch + previous_step

  # Optimize
  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
  loss_func = nn.MSELoss(reduction="sum") # old: nn.MSELoss(size_average=False)
  t1 = time.time()
  loss_log = []
  num_stage = int(args.mode[0])
  ploss1 = ploss2 = ploss3 = ploss4 = ploss5 = torch.FloatTensor(0).cuda()
  test(model, args, log, -1) # initial test
  for epoch in range(args.epoch):
    train(training_data_loader, optimizer, model, loss_func, epoch, args, log)
    save_checkpoint(model, epoch, TIME_ID, weights_path, args)
    test(model, args, log, epoch)
    