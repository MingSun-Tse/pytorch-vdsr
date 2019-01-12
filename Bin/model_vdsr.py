import numpy as np
import os
import torch.nn as nn
import torch

# Original VDSR model
class VDSR(nn.Module):
  def __init__(self, model=False, fixed=False):
    super(VDSR, self).__init__()
    self.fixed = fixed
    self.conv1  = nn.Conv2d( 1,64,3,1,1,bias=False)
    self.conv2  = nn.Conv2d(64,64,3,1,1,bias=False)
    self.conv3  = nn.Conv2d(64,64,3,1,1,bias=False)
    self.conv4  = nn.Conv2d(64,64,3,1,1,bias=False)
    self.conv5  = nn.Conv2d(64,64,3,1,1,bias=False)
    self.conv6  = nn.Conv2d(64,64,3,1,1,bias=False)
    self.conv7  = nn.Conv2d(64,64,3,1,1,bias=False)
    self.conv8  = nn.Conv2d(64,64,3,1,1,bias=False)
    self.conv9  = nn.Conv2d(64,64,3,1,1,bias=False)
    self.conv10 = nn.Conv2d(64,64,3,1,1,bias=False)
    self.conv11 = nn.Conv2d(64,64,3,1,1,bias=False)
    self.conv12 = nn.Conv2d(64,64,3,1,1,bias=False)
    self.conv13 = nn.Conv2d(64,64,3,1,1,bias=False)
    self.conv14 = nn.Conv2d(64,64,3,1,1,bias=False)
    self.conv15 = nn.Conv2d(64,64,3,1,1,bias=False)
    self.conv16 = nn.Conv2d(64,64,3,1,1,bias=False)
    self.conv17 = nn.Conv2d(64,64,3,1,1,bias=False)
    self.conv18 = nn.Conv2d(64,64,3,1,1,bias=False)
    self.conv19 = nn.Conv2d(64,64,3,1,1,bias=False)
    self.conv20 = nn.Conv2d(64, 1,3,1,1,bias=False)
    self.relu = nn.ReLU(inplace=True)
    
    if model:
      self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
  def forward(self, y):
    y = self.relu(self.conv1(y))
    y = self.relu(self.conv2(y))
    y = self.relu(self.conv3(y))
    y = self.relu(self.conv4(y))
    y = self.relu(self.conv5(y))
    y = self.relu(self.conv6(y))
    y = self.relu(self.conv7(y))
    y = self.relu(self.conv8(y))
    y = self.relu(self.conv9(y))
    y = self.relu(self.conv10(y))
    y = self.relu(self.conv11(y))
    y = self.relu(self.conv12(y))
    y = self.relu(self.conv13(y))
    y = self.relu(self.conv14(y))
    y = self.relu(self.conv15(y))
    y = self.relu(self.conv16(y))
    y = self.relu(self.conv17(y))
    y = self.relu(self.conv18(y))
    y = self.relu(self.conv19(y))
    y = self.conv20(y) # note there is no relu in the output layer
    return y
  
  def forward_stem(self, y):
    out1 = self.relu(self.conv1(y)); y = self.relu(self.conv2(out1))
    out3 = self.relu(self.conv3(y)); y = self.relu(self.conv4(out3))
    out5 = self.relu(self.conv5(y)); y = self.relu(self.conv6(out5))
    out7 = self.relu(self.conv7(y)); y = self.relu(self.conv8(out7))
    out9 = self.relu(self.conv9(y)); y = self.relu(self.conv10(out9))
    out11 = self.relu(self.conv11(y)); y = self.relu(self.conv12(out11))
    y = self.relu(self.conv13(y))
    y = self.relu(self.conv14(y))
    y = self.relu(self.conv15(y))
    y = self.relu(self.conv16(y))
    y = self.relu(self.conv17(y))
    y = self.relu(self.conv18(y))
    y = self.relu(self.conv19(y))
    y = self.conv20(y)
    return out1, out3, out5, out7, out9, out11, y
    
class SmallVDSR_16x(nn.Module):
  def __init__(self, model=False, fixed=False):
    super(SmallVDSR_16x, self).__init__()
    self.fixed = fixed
    self.conv1  = nn.Conv2d( 1,16,3,1,1,bias=False)
    self.conv2  = nn.Conv2d(16,16,3,1,1,bias=False)
    self.conv3  = nn.Conv2d(16,16,3,1,1,bias=False)
    self.conv4  = nn.Conv2d(16,16,3,1,1,bias=False)
    self.conv5  = nn.Conv2d(16,16,3,1,1,bias=False)
    self.conv6  = nn.Conv2d(16,16,3,1,1,bias=False)
    self.conv7  = nn.Conv2d(16,16,3,1,1,bias=False)
    self.conv8  = nn.Conv2d(16,16,3,1,1,bias=False)
    self.conv9  = nn.Conv2d(16,16,3,1,1,bias=False)
    self.conv10 = nn.Conv2d(16,16,3,1,1,bias=False)
    self.conv11 = nn.Conv2d(16,16,3,1,1,bias=False)
    self.conv12 = nn.Conv2d(16,16,3,1,1,bias=False)
    self.conv13 = nn.Conv2d(16,16,3,1,1,bias=False)
    self.conv14 = nn.Conv2d(16,16,3,1,1,bias=False)
    self.conv15 = nn.Conv2d(16,16,3,1,1,bias=False)
    self.conv16 = nn.Conv2d(16,16,3,1,1,bias=False)
    self.conv17 = nn.Conv2d(16,16,3,1,1,bias=False)
    self.conv18 = nn.Conv2d(16,16,3,1,1,bias=False)
    self.conv19 = nn.Conv2d(16,16,3,1,1,bias=False)
    self.conv20 = nn.Conv2d(16, 1,3,1,1,bias=False)
        
    self.relu = nn.ReLU(inplace=True)
    self.conv1_aux  = nn.Conv2d(16,64,1,1,0,bias=False)
    # self.conv3_aux  = nn.Conv2d(16,64,1,1,0,bias=False)
    self.conv5_aux  = nn.Conv2d(16,64,1,1,0,bias=False)
    # self.conv7_aux  = nn.Conv2d(16,64,1,1,0,bias=False)
    # self.conv9_aux  = nn.Conv2d(16,64,1,1,0,bias=False)
    self.conv10_aux  = nn.Conv2d(16,64,1,1,0,bias=False)
    # self.conv11_aux = nn.Conv2d(16,64,1,1,0,bias=False)
    self.conv15_aux  = nn.Conv2d(16,64,1,1,0,bias=False)
    
    if model:
      self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
    
  def forward_aux(self, y):
    y = self.relu(self.conv1(y)); out1_aux = self.relu(self.conv1_aux(y))
    y = self.relu(self.conv2(y))
    y = self.relu(self.conv3(y)); # out3_aux = self.relu(self.conv3_aux(y))
    y = self.relu(self.conv4(y))
    y = self.relu(self.conv5(y)); out5_aux = self.relu(self.conv5_aux(y))
    y = self.relu(self.conv6(y))
    y = self.relu(self.conv7(y)); # out7_aux = self.relu(self.conv7_aux(y))
    y = self.relu(self.conv8(y))
    y = self.relu(self.conv9(y)); # out9_aux = self.relu(self.conv9_aux(y))
    y = self.relu(self.conv10(y)); out10_aux = self.relu(self.conv10_aux(y))
    y = self.relu(self.conv11(y)); # out11_aux = self.relu(self.conv11_aux(y))
    y = self.relu(self.conv12(y))
    y = self.relu(self.conv13(y))
    y = self.relu(self.conv14(y))
    y = self.relu(self.conv15(y)); out15_aux = self.relu(self.conv15_aux(y))
    y = self.relu(self.conv16(y))
    y = self.relu(self.conv17(y))
    y = self.relu(self.conv18(y))
    y = self.relu(self.conv19(y))
    y = self.conv20(y)
    return out1_aux, out3_aux, out5_aux, out7_aux, out9_aux, out11_aux, y
    
  def forward(self, y):
    y = self.relu(self.conv1(y))
    y = self.relu(self.conv2(y))
    y = self.relu(self.conv3(y))
    y = self.relu(self.conv4(y))
    y = self.relu(self.conv5(y))
    y = self.relu(self.conv6(y))
    y = self.relu(self.conv7(y))
    y = self.relu(self.conv8(y))
    y = self.relu(self.conv9(y))
    y = self.relu(self.conv10(y))
    y = self.relu(self.conv11(y))
    y = self.relu(self.conv12(y))
    y = self.relu(self.conv13(y))
    y = self.relu(self.conv14(y))
    y = self.relu(self.conv15(y))
    y = self.relu(self.conv16(y))
    y = self.relu(self.conv17(y))
    y = self.relu(self.conv18(y))
    y = self.relu(self.conv19(y))
    y = self.conv20(y)
    return y

class KTSmallVDSR_16x(nn.Module):
  def __init__(self, e1, e2):
    super(KTSmallVDSR_16x, self).__init__()
    self.e1 = VDSR(e1, fixed=True)
    self.e2 = SmallVDSR_16x(e2)
  def forward(self, input):
    feats  = self.e1.forward(input)
    feats2 = self.e2.forward(input)
    HR_predicted = torch.add(feats2, input)
    feats3 = feats # self.e1.forward_stem(HR_predicted)
    return feats, feats2, HR_predicted, feats3

Autoencoders = {
"16x": KTSmallVDSR_16x,
}
