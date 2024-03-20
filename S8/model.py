
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.optim.lr_scheduler import StepLR
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np


## For S8 of EraV2

class Net_LN_S8(nn.Module):
    @staticmethod
    def calc_activation_shape(dim, ksize=(3, 3), dilation=(1, 1), stride=(1, 1), padding=(0, 0)):
        def shape_each_dim(i):
            odim_i = dim[i] + 2 * padding[i] - dilation[i] * (ksize[i] - 1) - 1
            return int((odim_i / stride[i]) + 1)
        return shape_each_dim(0), shape_each_dim(1)

    #  https://wandb.ai/wandb_fc/LayerNorm/reports/Layer-Normalization-in-Pytorch-With-Examples---VmlldzoxMjk5MTk1

    def __init__(self):
        super(Net_LN_S8, self).__init__()
        set1 = 8 #channels
        set2 = 8 #channels
        out = 10 #channels
        avg = 5 #channels
        drop = 0.25 #dropout
        mom = 0.1
        self.conv1 = nn.Conv2d(3, set1, 3, padding=1) #first 3 => input channels(R,G,B) last 3 => kernel size (3x3)
        ln_shape = self.calc_activation_shape(dim=(32,32), padding=(1, 1))
        self.ln1 = nn.LayerNorm(normalized_shape=[set1, *ln_shape])
        self.conv2 = nn.Conv2d(set1, set1, 3)
        ln_shape = self.calc_activation_shape(dim=(32,32))
        self.ln2 = nn.LayerNorm(normalized_shape=[set1, *ln_shape])
        self.conv3 = nn.Conv2d(set1, set1, 1, padding=1)
        ln_shape = self.calc_activation_shape(dim=(34,34), padding=(1, 1))
        self.ln3 =  nn.LayerNorm(normalized_shape=[set1, *ln_shape])
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(set1, set1, 3, padding=1)
        ln_shape = self.calc_activation_shape(dim=(16,16), padding=(1, 1))
        self.ln4 =  nn.LayerNorm(normalized_shape=[set1, *ln_shape])
        self.conv5 = nn.Conv2d(set1, set2, 3, padding=1)
        ln_shape = self.calc_activation_shape(dim=(16,16), padding=(1, 1))
        self.ln5 =  nn.LayerNorm(normalized_shape=[set2, *ln_shape])
        self.conv6 = nn.Conv2d(set2, set2, 3)
        ln_shape = self.calc_activation_shape(dim=(16,16))
        self.ln6 =  nn.LayerNorm(normalized_shape=[set2, *ln_shape])
        self.conv7 = nn.Conv2d(set2, set2, 1)
        ln_shape = self.calc_activation_shape(dim=(19,19))
        self.ln7 =  nn.LayerNorm(normalized_shape=[set2, *ln_shape])
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv8 = nn.Conv2d(set2, set2, 3)
        ln_shape = self.calc_activation_shape(dim=(7,7))
        self.ln8 =  nn.LayerNorm(normalized_shape=[set2, *ln_shape])
        self.conv9 = nn.Conv2d(set2, set2, 3)
        ln_shape = self.calc_activation_shape(dim=(5,5))
        self.ln9 =  nn.LayerNorm(normalized_shape=[set2, *ln_shape])
        self.conv10 = nn.Conv2d(set2, out, 3)
        ln_shape = self.calc_activation_shape(dim=(3,3))
        self.ln10 =  nn.LayerNorm(normalized_shape=[out, *ln_shape])

        self.gap = nn.AvgPool2d(kernel_size=[avg,avg], stride=[avg,avg], padding=0, ceil_mode=False, count_include_pad=False)
        self.conv11 = nn.Conv2d(10, out, 1)


        self.drop = nn.Dropout(drop)

    def forward(self, x):

        # print(" 1. " + str(x.shape))
        x = self.ln1(F.relu(self.conv1(x)))
        # print(" 2. " + str(x.shape))
        x = self.pool1((F.relu(self.conv3(self.ln2(F.relu(self.conv2(x)))))))
        # print(" 3. " + str(x.shape))
        x = self.conv4(x)
        # print(" 4. " + str(x.shape))
        x = self.conv5(self.ln4(F.relu(x)))
        # print(" 5. " + str(x.shape))
        x = self.ln5(F.relu(x))
        # print(" 6. " + str(x.shape))
        x = self.conv7(self.ln6(F.relu(self.conv6(x)))) #ln for conv7 layer is avoided; if ln is provided, the params are tremendously high. May be due to 1x1.
        # print(" 7. " + str(x.shape))
        x = self.pool2((F.relu(x)))
        # print(" 8. " + str(x.shape))
        x = self.conv8(x)
        # print(" 9. " + str(x.shape))
        x = self.conv9((self.ln8(F.relu(x))))
        # print("10. " + str(x.shape))
        x = self.conv10(self.ln9(F.relu(x)))
        # print("11. " + str(x.shape))
        x = self.ln10(F.relu(x))
        # print("12. " + str(x.shape))
        x = self.conv11(x)
        # print("13. " + str(x.shape))
        #print(x.shape)
        #x = self.gap(x)
        #print(x.shape)
        x = x.view(-1, 10) # Raja ToDo Try printing shape here
        return F.log_softmax(x)