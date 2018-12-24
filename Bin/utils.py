import os
import time
import sys
import torch
import numpy as np

def is_img(x):
  return any(x.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def logprint(some_str, f=sys.stdout):
  print(time.strftime("[%s" % os.getpid() + "-%Y/%m/%d-%H:%M] ") + str(some_str), file=f, flush=True)

def load_param_from_t7(model, in_layer_index, out_layer):
  out_layer.weight = torch.nn.Parameter(model.get(in_layer_index).weight.float())
  out_layer.bias = torch.nn.Parameter(model.get(in_layer_index).bias.float())

def smooth(L, window = 50):
    num = len(L)
    L1 = list(L[:window]) + list(L)
    out = [np.average(L1[i:i+window]) for i in range(num)]
    return np.array(out) if type(L) == type(np.array([0])) else out
 
# take model1's params to model2
# for each layer of model2, if model1 has the same layer, then copy the params.
def cut_pth(model1, model2):
  params1 = model1.named_parameters()
  params2 = model2.named_parameters()
  dict_params1 = dict(params1)
  dict_params2 = dict(params2)
  for name2, _ in params2:
    if name2 in dict_params1:
      dict_params2[name2].data.copy_(dict_params1[name2].data)
  model2.load_state_dict(dict_params2)
  torch.save(model2.state_dict(), "model2.pth")
