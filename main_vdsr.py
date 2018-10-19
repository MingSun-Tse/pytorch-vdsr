import argparse, os, sys, time
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from vdsr import Net
from dataset import DatasetFromHdf5
pjoin = os.path.join

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
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--num_filter", default=64, type=int)

weights_path = "./checkpoint"
TIME_ID = os.environ["SERVER"] + time.strftime("-%Y%m%d-%H%M")
log_path = pjoin(weights_path, "log_" + TIME_ID + ".txt")
log = open(log_path, "w+")

def logprint(some_str, f=sys.stdout):
  print(time.strftime("[%s" % os.getpid() + "-%Y/%m/%d-%H:%M] ") + some_str, file=f, flush=True)
  
def main():
  global opt, model
  opt = parser.parse_args()
  logprint(str(opt._get_kwargs()), log)
  
  cuda = opt.cuda
  if cuda:
      logprint("=> use gpu id: '{}'".format(opt.gpus), log)
      os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
      if not torch.cuda.is_available():
              raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

  opt.seed = random.randint(1, 10000)
  logprint("Random Seed: %s" % opt.seed, log)
  torch.manual_seed(opt.seed)
  if cuda:
      torch.cuda.manual_seed(opt.seed)

  cudnn.benchmark = True

  logprint("===> Loading datasets", log)
  train_set = DatasetFromHdf5("data/train3.h5")
  training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

  logprint("===> Building model", log)
  model = Net(opt.num_filter)
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
  for epoch in range(opt.start_epoch, opt.nEpochs + 1):
      train(training_data_loader, optimizer, model, criterion, epoch)
      save_checkpoint(model, epoch)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def train(training_data_loader, optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(optimizer, epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    logprint("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]), log)

    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        loss = criterion(model(input), target)
        optimizer.zero_grad()
        loss.backward() 
        nn.utils.clip_grad_norm(model.parameters(),opt.clip) 
        optimizer.step()

        if iteration%100 == 0:
            logprint("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.data[0]), log)

def save_checkpoint(model, epoch):
    model_out_path = "checkpoint/" + "%s_model_epoch_{}.pth".format(TIME_ID, epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    logprint("Checkpoint saved to {}".format(model_out_path), log)

if __name__ == "__main__":
    main()
