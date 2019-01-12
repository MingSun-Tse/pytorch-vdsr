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
from utils import logprint
from dataset import DatasetFromHdf5 # vdsr data loader

def adjust_learning_rate(epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.step))
    return lr

def train(training_data_loader, optimizer, model, loss_func, epoch, args, log):
    lr = adjust_learning_rate(epoch-1, args)
  
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    logprint("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]), log)

    model.train()
    ploss1 = ploss2 = ploss3 = ploss4 = ploss5 = torch.FloatTensor(0).cuda(args.gpu)
    for step, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        input = input.cuda(args.gpu)
        target = target.cuda(args.gpu)
        
        # -----------------------------------------------------
        # deeply-supervised perceptual loss
        feats, feats2, HR_predicted, feats3 = model(input)
        # ploss1 = loss_func(feats2[0], feats[0].data) * args.ploss_weight
        # ploss2 = loss_func(feats2[1], feats[1].data) * args.ploss_weight * 0.1
        # ploss3 = loss_func(feats2[2], feats[2].data) * args.ploss_weight
        # ploss4 = loss_func(feats2[3], feats[3].data) * args.ploss_weight
        # ploss5 = loss_func(feats2[4], feats[4].data) * args.ploss_weight
        # ploss6 = loss_func(feats2[5], feats[5].data) * args.ploss_weight
        # ploss1 = loss_func(feats2, feats.data) * args.ploss_weight
        
        # dploss1 = loss_func(feats3[0], feats[0].data) * args.dploss_weight
        # dploss2 = loss_func(feats3[1], feats[1].data) * args.dploss_weight * 0.1
        # dploss3 = loss_func(feats3[2], feats[2].data) * args.dploss_weight
        # dploss4 = loss_func(feats3[3], feats[3].data) * args.dploss_weight
        # dploss5 = loss_func(feats3[4], feats[4].data) * args.dploss_weight
        # dploss6 = loss_func(feats3[5], feats[5].data) * args.dploss_weight
        
        ploss1 = loss_func(feats2, feats.data) * args.ploss_weight
        iloss = loss_func(HR_predicted, target) * args.iloss_weight
        loss = iloss + ploss1 #+ ploss2 + ploss3 + ploss4 + ploss5 #+ dploss1 + dploss2 + dploss3 + dploss4 + dploss5
        # -----------------------------------------------------
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(),args.clip) 
        optimizer.step()

        if step % SHOW_INTERVAL == 0:
          format_str = "E{}S{} loss={:.3f} | iloss={:.5f} | ploss1={:.5f}" #ploss2={:.5f} ploss3={:.5f} ploss4={:.5f} ploss5={:.5f} ({:.3f}s/step)"
          logprint(format_str.format(epoch, step, loss.data.cpu().numpy(), iloss.data.cpu().numpy(), ploss1.data.cpu().numpy()), log)#, ploss2.data.cpu().numpy(),
              #ploss3.data.cpu().numpy(), ploss4.data.cpu().numpy(), ploss5.data.cpu().numpy(), (time.time()-t1)/SHOW_INTERVAL), log)
          global t1; t1 = time.time()

def save_checkpoint(ae, epoch, TIME_ID, weights_path, args):
  # save model
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
  parser = argparse.ArgumentParser(description="Autoencoder")
  parser.add_argument('--train_data', type=str, help='the directory of train images', default="../Data/train_data/train.h5")
  parser.add_argument('--test_data', type=str, help='the directory of test images', default="../Data/test_data")
  parser.add_argument('--e1', type=str, help='path of pretrained encoder1', default=None)
  parser.add_argument('--e2', type=str, help='path of pretrained encoder2', default=None)
  parser.add_argument('-d', '--decoder', type=str, help='path of pretrained decoder', default=None)
  parser.add_argument('-g', '--gpu', type=int, help="which gpu to run on. default is 0", default=0)
  parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=128)
  parser.add_argument('--lr', type=float, help='learning rate', default=0.1)
  parser.add_argument('--ploss_weight', type=float, help='loss weight to balance multi-losses', default=1.0)
  parser.add_argument('--dploss_weight', type=float, help='loss weight to balance multi-losses', default=1.0)
  parser.add_argument('--iloss_weight', type=float, help='loss weight to balance multi-losses', default=1.0)
  parser.add_argument('-p', '--project_name', type=str, help='the name of project, to save logs etc., will be set in directory, "Experiments"')
  parser.add_argument('-r', '--resume', action='store_true', help='if resume, default=False')
  parser.add_argument('-m', '--mode', type=str, help='the training mode name.')
  parser.add_argument('--epoch', type=int, default=50)
  parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
  parser.add_argument("--cuda", action="store_false", help="Use cuda?")
  parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
  parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
  parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
  parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
  parser.add_argument("--num_filter", default=64, type=int)
  parser.add_argument("--debug", action="store_true")
  args = parser.parse_args()

  # Set up data
  train_set = DatasetFromHdf5(args.train_data)
  training_data_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=args.batch_size, shuffle=True) # 'num_workers' need to be 1, otherwise will cause read error.
  
  # Set up directories and logs etc
  project_path = pjoin("../Experiments", args.project_name)
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
  
  # Set up model
  model = Autoencoders[args.mode](args.e1, args.e2)
  model.cuda(args.gpu)

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
  loss_func = nn.MSELoss(size_average=False)
  t1 = time.time()
  loss_log = []
  num_stage = int(args.mode[0])
  ploss1 = ploss2 = ploss3 = ploss4 = ploss5 = torch.FloatTensor(0).cuda(args.gpu)
  for epoch in range(1, args.epoch+1):
    train(training_data_loader, optimizer, model, loss_func, epoch, args, log)
    save_checkpoint(model, epoch, TIME_ID, weights_path, args)
  log.close()

  

  
