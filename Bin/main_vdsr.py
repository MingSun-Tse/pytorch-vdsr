import argparse, os, sys, time
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import shutil
from torch.autograd import Variable
from torch.utils.data import DataLoader
from vdsr import Net
from dataset import DatasetFromHdf5
import cv2
import glob
import scipy.io as sio
import math
import numpy as np
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


def logprint(some_str, f=sys.stdout):
    print(time.strftime("[%s" % os.getpid() + "-%Y/%m/%d-%H:%M] ") + some_str, file=f, flush=True)
  
def main():
  # Training settings
  parser = argparse.ArgumentParser(description="PyTorch VDSR")
  parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
  parser.add_argument("--nEpochs", type=int, default=50, help="Number of epochs to train for")
  parser.add_argument("--lr", type=float, default=0.1, help="Learning Rate. Default=0.1")
  parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
  parser.add_argument("--cuda", action="store_true", help="Use cuda?")
  parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
  parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
  parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
  parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
  parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
  parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
  parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
  parser.add_argument("--gpu", default="0", type=str, help="gpu ids (default: 0)")
  parser.add_argument("--num_filter", default=64, type=int)
  parser.add_argument("--num_block", default=18, type=int)
  parser.add_argument("--train_data", type=str, default="../Data/train_data/train.h5")
  parser.add_argument("--test_data", type=str, default="../Data/test_data/Set5_mat")
  parser.add_argument("-p", "--project_name", type=str)
  parser.add_argument("--debug", action="store_true")
  parser.add_argument("--sharpen", action="store_true")
  parser.add_argument("--drop_ratio", type=float, default=0)
  opt = parser.parse_args()
  
  # Set up directories and logs etc
  if opt.debug:
    opt.project_name = "test"
  project_path = pjoin("../Experiments", opt.project_name)
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
  log = sys.stdout if opt.debug else open(log_path, "w+")
  logprint(str(opt._get_kwargs()), log)
  
  cuda = opt.cuda
  if cuda:
    logprint("=> use gpu id: '{}'".format(opt.gpu), log)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    if not torch.cuda.is_available():
      raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

  opt.seed = random.randint(1, 10000)
  logprint("Random Seed: %s" % opt.seed, log)
  torch.manual_seed(opt.seed)
  if cuda:
    torch.cuda.manual_seed(opt.seed)

  cudnn.benchmark = True

  logprint("===> Loading datasets", log)
  train_set = DatasetFromHdf5(opt.train_data)
  training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

  logprint("===> Building model", log)
  model = Net(opt.num_filter, opt.num_block, opt.sharpen, opt.drop_ratio) ##### creat model
  criterion = nn.MSELoss(size_average=False)

  logprint("===> Setting GPU", log)
  if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

  # optionally resume from a checkpoint
  if opt.resume:
    if os.path.isfile(opt.resume):
      logprint("=> loading checkpoint '{}'".format(opt.resume), log)
      checkpoint = torch.load(opt.resume)
      opt.start_epoch = checkpoint["epoch"] + 1
      model.load_state_dict(checkpoint["model"].state_dict())
    else:
      logprint("=> no checkpoint found at '{}'".format(opt.resume), log)

  # optionally copy weights from a checkpoint
  if opt.pretrained:
    if os.path.isfile(opt.pretrained):
      logprint("=> loading model '{}'".format(opt.pretrained), log)
      weights = torch.load(opt.pretrained)
      model.load_state_dict(weights['model'].state_dict())
    else:
      logprint("=> no model found at '{}'".format(opt.pretrained), log)  

  logprint("===> Setting Optimizer", log)
  optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

  logprint("===> Training", log)
  test(model, opt, log)
  for epoch in range(opt.start_epoch, opt.nEpochs + 1):
    train(training_data_loader, optimizer, model, criterion, epoch, opt, log)
    save_checkpoint(model, epoch, log, weights_path, TIME_ID)
    test(model, opt, log)

def sharpen(in_image):
    k = -0.4 # original k = -1. The larger abs(k), the sharp the result.
    kernel = np.array([[0, k, 0], [k, -4*k+1, k], [0, k, 0]], np.float32) # sharpen
    dst = cv2.filter2D(in_image, -1, kernel=kernel)
    return dst
    
def test(model, opt, log):
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
        
        HR = model(im_input).cpu()
        im_h_y = HR.data[0].numpy().astype(np.float32)
        im_h_y = im_h_y * 255.
        im_h_y[im_h_y < 0] = 0
        im_h_y[im_h_y > 255.] = 255.
        im_h_y = im_h_y[0,:,:]

        psnr_predicted = PSNR(im_gt_y, im_h_y, shave_border=scale)
        avg_psnr_predicted += psnr_predicted

    logprint("Scale = {}, PSNR_predicted = {:.4f}, PSNR_bicubic = {:.4f}".format(scale, avg_psnr_predicted/count, avg_psnr_bicubic/count), log)
    
def adjust_learning_rate(optimizer, epoch, opt):
  """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
  lr = opt.lr * (0.1 ** (epoch // opt.step))
  return lr

def train(training_data_loader, optimizer, model, criterion, epoch, opt, log):
  lr = adjust_learning_rate(optimizer, epoch-1, opt)
  for param_group in optimizer.param_groups:
    param_group["lr"] = lr
  logprint("Epoch = {}, lr = {:.7f}".format(epoch, optimizer.param_groups[0]["lr"]), log)

  model.train()
  for iteration, batch in enumerate(training_data_loader, 1):
    input, target = Variable(batch[0]).cuda(), Variable(batch[1], requires_grad=False).cuda()
    loss = criterion(model(input), target)
    optimizer.zero_grad()
    loss.backward() 
    nn.utils.clip_grad_norm_(model.parameters(), opt.clip) 
    optimizer.step()

    if iteration % 100 == 0:
      logprint("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.data[0]), log)

def save_checkpoint(model, epoch, log, weights_path, TIME_ID):
  model_out_path = pjoin(weights_path, "{}_Epoch{}.pth".format(TIME_ID, epoch))
  state = {"epoch": epoch, "model": model}
  torch.save(state, model_out_path)
  logprint("Checkpoint saved to {}".format(model_out_path), log)

if __name__ == "__main__":
  main()
