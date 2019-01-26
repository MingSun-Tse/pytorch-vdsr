import numpy as np
import os
import torch.nn as nn
import torch


# Load param from model1 to model2
# For each layer of model2, if model1 has the same layer, then copy the params.
def load_param(model1_path, model2):
  dict_param1 = torch.load(model1_path) # model1_path: .pth model path
  dict_param2 = dict(model2.named_parameters())
  for name2 in dict_param2:
    if name2 in dict_param1:
      # print("tensor '%s' found in both models, so copy it from model 1 to model 2" % name2)
      dict_param2[name2].data.copy_(dict_param1[name2].data)
  model2.load_state_dict(dict_param2)
  return model2
  

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
      load_param(model, self) # self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
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
    y = self.relu(self.conv1(y)); out1 = y
    y = self.relu(self.conv2(y)); out2 = y
    y = self.relu(self.conv3(y)); out3 = y
    y = self.relu(self.conv4(y)); out4 = y
    y = self.relu(self.conv5(y)); out5 = y
    y = self.relu(self.conv6(y)); out6 = y
    y = self.relu(self.conv7(y)); out7 = y
    y = self.relu(self.conv8(y)); out8 = y
    y = self.relu(self.conv9(y)); out9 = y
    y = self.relu(self.conv10(y)); out10 = y
    y = self.relu(self.conv11(y)); out11 = y
    y = self.relu(self.conv12(y)); out12 = y
    y = self.relu(self.conv13(y)); out13 = y
    y = self.relu(self.conv14(y)); out14 = y
    y = self.relu(self.conv15(y)); out15 = y
    y = self.relu(self.conv16(y)); out16 = y
    y = self.relu(self.conv17(y)); out17 = y
    y = self.relu(self.conv18(y)); out18 = y
    y = self.relu(self.conv19(y)); out19 = y
    y = self.conv20(y)
    return out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, \
           out11, out12, out13, out14, out15, out16, out17, out18, out19, y
    
  def forward_dense(self, y):
    y = self.relu(self.conv1(y)); out1 = y
    y = self.relu(self.conv2(y)); out2 = y
    y = self.relu(self.conv3(y)); out3 = y
    y = self.relu(self.conv4(y)); out4 = y
    y = self.relu(self.conv5(y)); out5 = y
    y = self.relu(self.conv6(y)); out6 = y
    y = self.relu(self.conv7(y)); out7 = y
    y = self.relu(self.conv8(y)); out8 = y
    y = self.relu(self.conv9(y)); out9 = y
    y = self.relu(self.conv10(y)); out10 = y
    y = self.relu(self.conv11(y)); out11 = y
    y = self.relu(self.conv12(y)); out12 = y
    y = self.relu(self.conv13(y)); out13 = y
    y = self.relu(self.conv14(y)); out14 = y
    y = self.relu(self.conv15(y)); out15 = y
    y = self.relu(self.conv16(y)); out16 = y
    y = self.relu(self.conv17(y)); out17 = y
    y = self.relu(self.conv18(y)); out18 = y
    y = self.relu(self.conv19(y)); out19 = y
    y = self.conv20(y); out20 = y
    return out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, \
           out11, out12, out13, out14, out15, out16, out17, out18, out19, out20
    
    
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
    self.relu  = nn.ReLU()
    
    if model:
      load_param(model, self)
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
    
  def forward_stem(self, y):
    y = self.relu(self.conv1(y)); out1 = y
    y = self.relu(self.conv2(y)); out2 = y
    y = self.relu(self.conv3(y)); out3 = y
    y = self.relu(self.conv4(y)); out4 = y
    y = self.relu(self.conv5(y)); out5 = y
    y = self.relu(self.conv6(y)); out6 = y
    y = self.relu(self.conv7(y)); out7 = y
    y = self.relu(self.conv8(y)); out8 = y
    y = self.relu(self.conv9(y)); out9 = y
    y = self.relu(self.conv10(y)); out10 = y
    y = self.relu(self.conv11(y)); out11 = y
    y = self.relu(self.conv12(y)); out12 = y
    y = self.relu(self.conv13(y)); out13 = y
    y = self.relu(self.conv14(y)); out14 = y
    y = self.relu(self.conv15(y)); out15 = y
    y = self.relu(self.conv16(y)); out16 = y
    y = self.relu(self.conv17(y)); out17 = y
    y = self.relu(self.conv18(y)); out18 = y
    y = self.relu(self.conv19(y)); out19 = y
    y = self.conv20(y)
    return out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, \
           out11, out12, out13, out14, out15, out16, out17, out18, out19, y
        
  def forward_dense(self, y):
    y = self.relu(self.conv1(y)); out1 = y
    y = self.relu(self.conv2(y)); out2 = y
    y = self.relu(self.conv3(y)); out3 = y
    y = self.relu(self.conv4(y)); out4 = y
    y = self.relu(self.conv5(y)); out5 = y
    y = self.relu(self.conv6(y)); out6 = y
    y = self.relu(self.conv7(y)); out7 = y
    y = self.relu(self.conv8(y)); out8 = y
    y = self.relu(self.conv9(y)); out9 = y
    y = self.relu(self.conv10(y)); out10 = y
    y = self.relu(self.conv11(y)); out11 = y
    y = self.relu(self.conv12(y)); out12 = y
    y = self.relu(self.conv13(y)); out13 = y
    y = self.relu(self.conv14(y)); out14 = y
    y = self.relu(self.conv15(y)); out15 = y
    y = self.relu(self.conv16(y)); out16 = y
    y = self.relu(self.conv17(y)); out17 = y
    y = self.relu(self.conv18(y)); out18 = y
    y = self.relu(self.conv19(y)); out19 = y
    y = self.conv20(y); out20 = y
    return out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, \
           out11, out12, out13, out14, out15, out16, out17, out18, out19, out20
    
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

    
class SmallVDSR_F8(nn.Module):
  def __init__(self, model=False, fixed=False):
    super(SmallVDSR_F8, self).__init__()
    self.fixed  = fixed
    self.conv1  = nn.Conv2d(1,8,3,1,1,bias=False)
    self.conv2  = nn.Conv2d(8,8,3,1,1,bias=False)
    self.conv3  = nn.Conv2d(8,8,3,1,1,bias=False)
    self.conv4  = nn.Conv2d(8,8,3,1,1,bias=False)
    self.conv5  = nn.Conv2d(8,8,3,1,1,bias=False)
    self.conv6  = nn.Conv2d(8,8,3,1,1,bias=False)
    self.conv7  = nn.Conv2d(8,8,3,1,1,bias=False)
    self.conv8  = nn.Conv2d(8,8,3,1,1,bias=False)
    self.conv9  = nn.Conv2d(8,8,3,1,1,bias=False)
    self.conv10 = nn.Conv2d(8,8,3,1,1,bias=False)
    self.conv11 = nn.Conv2d(8,8,3,1,1,bias=False)
    self.conv12 = nn.Conv2d(8,8,3,1,1,bias=False)
    self.conv13 = nn.Conv2d(8,8,3,1,1,bias=False)
    self.conv14 = nn.Conv2d(8,8,3,1,1,bias=False)
    self.conv15 = nn.Conv2d(8,8,3,1,1,bias=False)
    self.conv16 = nn.Conv2d(8,8,3,1,1,bias=False)
    self.conv17 = nn.Conv2d(8,8,3,1,1,bias=False)
    self.conv18 = nn.Conv2d(8,8,3,1,1,bias=False)
    self.conv19 = nn.Conv2d(8,8,3,1,1,bias=False)
    self.conv20 = nn.Conv2d(8,1,3,1,1,bias=False)
    self.relu   = nn.ReLU()
    
    if model:
      load_param(model, self)
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
    
  def forward_stem(self, y):
    y = self.relu(self.conv1(y)); out1 = y
    y = self.relu(self.conv2(y)); out2 = y
    y = self.relu(self.conv3(y)); out3 = y
    y = self.relu(self.conv4(y)); out4 = y
    y = self.relu(self.conv5(y)); out5 = y
    y = self.relu(self.conv6(y)); out6 = y
    y = self.relu(self.conv7(y)); out7 = y
    y = self.relu(self.conv8(y)); out8 = y
    y = self.relu(self.conv9(y)); out9 = y
    y = self.relu(self.conv10(y)); out10 = y
    y = self.relu(self.conv11(y)); out11 = y
    y = self.relu(self.conv12(y)); out12 = y
    y = self.relu(self.conv13(y)); out13 = y
    y = self.relu(self.conv14(y)); out14 = y
    y = self.relu(self.conv15(y)); out15 = y
    y = self.relu(self.conv16(y)); out16 = y
    y = self.relu(self.conv17(y)); out17 = y
    y = self.relu(self.conv18(y)); out18 = y
    y = self.relu(self.conv19(y)); out19 = y
    y = self.conv20(y)
    return out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, \
           out11, out12, out13, out14, out15, out16, out17, out18, out19, y
    
  def forward_dense(self, y):
    y = self.relu(self.conv1(y)); out1 = y
    y = self.relu(self.conv2(y)); out2 = y
    y = self.relu(self.conv3(y)); out3 = y
    y = self.relu(self.conv4(y)); out4 = y
    y = self.relu(self.conv5(y)); out5 = y
    y = self.relu(self.conv6(y)); out6 = y
    y = self.relu(self.conv7(y)); out7 = y
    y = self.relu(self.conv8(y)); out8 = y
    y = self.relu(self.conv9(y)); out9 = y
    y = self.relu(self.conv10(y)); out10 = y
    y = self.relu(self.conv11(y)); out11 = y
    y = self.relu(self.conv12(y)); out12 = y
    y = self.relu(self.conv13(y)); out13 = y
    y = self.relu(self.conv14(y)); out14 = y
    y = self.relu(self.conv15(y)); out15 = y
    y = self.relu(self.conv16(y)); out16 = y
    y = self.relu(self.conv17(y)); out17 = y
    y = self.relu(self.conv18(y)); out18 = y
    y = self.relu(self.conv19(y)); out19 = y
    y = self.conv20(y); out20 = y
    return out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, \
           out11, out12, out13, out14, out15, out16, out17, out18, out19, out20
           
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
  
  def forward(self, LR):
    feats_F64 = self.e1.forward_stem(LR) # [batch, channel, height, width] x 20 layers
    feats_F16 = self.e2.forward_stem(LR)
    return feats_F64, feats_F16
    
class KTSmallVDSR_F8(nn.Module):
  def __init__(self, e1, e2):
    super(KTSmallVDSR_F8, self).__init__()
    self.e1 = VDSR(e1, fixed=True)
    self.e2 = SmallVDSR_F8(e2)
  
  def forward(self, LR):
    feats_F64 = self.e1.forward_stem(LR) # [batch, channel, height, width] x 20 layers
    feats_F8  = self.e2.forward_stem(LR)
    return feats_F64, feats_F8
    
Autoencoders = {
"16x": KTSmallVDSR_16x,
"F16": KTSmallVDSR_16x,
 "F8": KTSmallVDSR_F8,
}
