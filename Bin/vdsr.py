import torch
import torch.nn as nn
from math import sqrt
import numpy as np

class Conv_ReLU_Block(nn.Module):
    def __init__(self, num_filter=64):
      super(Conv_ReLU_Block, self).__init__()
      self.conv = nn.Conv2d(in_channels=num_filter, out_channels=num_filter, kernel_size=3, stride=1, padding=1, bias=False)
      self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
      return self.relu(self.conv(x))
      
class Sharpen_Block(nn.Module):
  def __init__(self):
    super(Sharpen_Block, self).__init__()
    self.pad = nn.ReflectionPad2d((1,1,1,1)) # simulate cv2.filter2D, which uses reflection padding
    self.conv = nn.Conv2d(1,1,3,1,0, bias=False)
    self.conv.weight = nn.Parameter(torch.from_numpy(np.array([[[[0, -0.4, 0], [0, 2.6, 0], [0, -0.4, 0]]]])).float())
    self.conv.weight.requires_grad = False
    
  def forward(self, x):
    return self.conv(self.pad(x))
      
class Net(nn.Module):
    def __init__(self, num_filter=64, num_block=18, sharpen=False):
      super(Net, self).__init__()
      self.sharpen = sharpen
      self.residual_layer = self.make_layer(Conv_ReLU_Block, num_block, num_filter)
      self.input  = nn.Conv2d(in_channels=1, out_channels=num_filter, kernel_size=3, stride=1, padding=1, bias=False)
      self.output = nn.Conv2d(in_channels=num_filter, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
      self.relu = nn.ReLU(inplace=True)
      self.sharpen_layer = Sharpen_Block()
      
      # normalize the weights
      for m in self.modules():
        if isinstance(m, nn.Conv2d):
          if m.weight.requires_grad == False: 
            print("sharpen layer, do not normalize")
            continue # do not normalize the sharpen layer
          n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
          m.weight.data.normal_(0, sqrt(2. / n))
    
    def make_layer(self, block, num_block, num_filter):
      layers = []
      for _ in range(num_block):
        layers.append(block(num_filter))    
      return nn.Sequential(*layers)

    def forward(self, x):
      residual = x
      out = self.relu(self.input(x))
      out = self.residual_layer(out)
      out = self.output(out) # there is no ReLU in the output layer
      out = torch.add(out, residual)
      return self.sharpen_layer(out) if self.sharpen else out
 
