import torch
import torch.nn as nn
import vdsr
from model_vdsr import VDSR, SmallVDSR_16x
import sys

original_vdsr_model = sys.argv[1]
vdsr_params = list(torch.load(original_vdsr_model)["model"].named_parameters())

encoder = VDSR() # Change this to your demand
params = encoder.named_parameters()
dict_params = dict(params)
for i in range(len(vdsr_params)):
  original_tensor_name, tensor_data = vdsr_params[i][0], vdsr_params[i][1]
  if original_tensor_name == "output.weight":
    new_tensor_name = "conv20.weight"
  elif original_tensor_name == "input.weight":
    new_tensor_name = "conv1.weight"
  else:
    new_tensor_name = "conv%d.weight" % (int(original_tensor_name.split(".")[1]) + 2)
  print("===> original_tensor_name: %s  vs.  new_tensor_name: %s" % (original_tensor_name, new_tensor_name))
  dict_params[new_tensor_name].data.copy_(tensor_data)

encoder.load_state_dict(dict_params)
torch.save(encoder.state_dict(), "my_vdsr_model.pth")

