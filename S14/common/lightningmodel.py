import os
import math
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import datasets, transforms, utils
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


seed_everything(7)



class Net_S13(nn.Module):
#class ResNet(nn.Module):
    def __init__(self):
        super(Net_S13, self).__init__()
        #super(ResNet, self).__init__()

        # Control Variable
        self.printShape = False

        #Common :-
        set1 = 64 #prepLayer
        set2 = 128 #Layer2
        set3 = 256 #Layer3
        set4 = 512 #Layer4
        avg = 1024 #channels
        drop = 0.1 #dropout
        S = 1 #stride
        K = 3 #kernel_size

        # PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
        I = 3
        O = set1
        P = 1 #padding
        self.prepLayer = self.convBlock(in_channels = I, out_channels = O, kernel_size = K, stride = S, padding = P)

        # Layer1 -
        # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        # R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
        # Add(X, R1)
        I = O
        O = set2
        P = 1 #padding
        self.Layer1 = self.convMPBlock(in_channels = I, out_channels = O, kernel_size = K, stride = S, padding = P)

        I = O
        O = I
        P = 1 #padding
        self.resNetLayer1Part1 = self.convBlock(in_channels = I, out_channels = O, kernel_size = K, stride = S, padding = P)

        I = O
        O = I
        P = 1 #padding
        self.resNetLayer1Part2 = self.convBlock(in_channels = I, out_channels = O, kernel_size = K, stride = S, padding = P)

        # Layer 2 -
        # Conv 3x3 [256k]
        # MaxPooling2D
        # BN
        # ReLU
        I = O
        O = set3
        P = 1 #padding
        self.Layer2 = self.convMPBlock(in_channels = I, out_channels = O, kernel_size = K, stride = S, padding = P)

        # Layer 3 -
        # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
        # R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
        # Add(X, R2)
        I = O
        O = set4
        P = 1 #padding
        self.Layer3 = self.convMPBlock(in_channels = I, out_channels = O, kernel_size = K, stride = S, padding = P)

        I = O
        O = I
        P = 1 #padding
        self.resNetLayer2Part1 = self.convBlock(in_channels = I, out_channels = O, kernel_size = K, stride = S, padding = P)

        I = O
        O = I
        P = 1 #padding
        self.resNetLayer2Part2 = self.convBlock(in_channels = I, out_channels = O, kernel_size = K, stride = S, padding = P)

        # MaxPooling with Kernel Size 4
        self.pool = nn.MaxPool2d(kernel_size = 4, stride = 4)

        # FC Layer
        I = 512
        O = 10
        self.lastLayer = nn.Linear(I, O)

        # self.aGAP = nn.AdaptiveAvgPool2d((1, 1))
        # self.flat = nn.Flatten(1, -1)
        # self.gap = nn.AvgPool2d(avg)
        self.drop = nn.Dropout(drop)

    # convolution Block
    def convBlock(self, in_channels, out_channels, kernel_size, stride, padding, last_layer = False, bias = False):
      if(False == last_layer):
        return nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, stride = stride, padding = padding, kernel_size = kernel_size, bias = bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
      else:
        return nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, stride = stride, padding = padding, kernel_size = kernel_size, bias = bias))

    # convolution-MP Block
    def convMPBlock(self, in_channels, out_channels, kernel_size, stride, padding, bias = False):
        return nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, stride = stride, padding = padding, kernel_size = kernel_size, bias = bias),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def printf(self, n, x, string1=""):
      if(self.printShape):
        print(f"{n} " f"{x.shape = }" f" {string1}") ##  Comment / Uncomment this line towards the no need of print or needed print
        pass
    def printEmpty(self,):
      if(self.printShape):
        print("") ##  Comment / Uncomment this line towards the no need of print or needed print
        pass

    def forward(self, x):
        self.printf(0.0, x, "prepLayer input")
        x = self.prepLayer(x)
        x = self.drop(x)
        self.printf(0.1, x, "prepLayer output")
        self.printEmpty()

        self.printf(1.0, x, "Layer1 input")
        x = self.Layer1(x)
        self.printf(1.1, x, "Layer1 output --> sacroscant")
        y = x #sacrosanct path1
        self.printf(1.2, x, "Layer1 resnet input")
        x = self.resNetLayer1Part1(x) #residual path1
        x = self.drop(x)
        x = self.resNetLayer1Part2(x) #residual path1
        self.printf(1.3, x, "Layer1 resnet output")
        x = x + y  #adding sacrosanct path1 and residual path1
        x = self.drop(x)
        self.printf(1.4, x, "res+sacrosanct output")
        self.printEmpty()

        self.printf(2.0, x, "Layer2 input")
        x = self.Layer2(x)
        x = self.drop(x)
        self.printf(2.1, x, "Layer2 output")
        self.printEmpty()

        self.printf(3.0, x, "Layer3 input")
        x = self.Layer3(x)
        self.printf(3.1, x, "Layer3 output --> sacroscant")
        y = x  #sacrosanct path2
        self.printf(3.2, x, "Layer3 resnet input")
        x = self.resNetLayer2Part1(x) #residual path2
        x = self.drop(x)
        x = self.resNetLayer2Part2(x) #residual path2
        self.printf(3.3, x, "Layer3 resnet output")
        x = x + y #adding sacrosanct path2 and residual path2
        x = self.drop(x)
        self.printf(3.4, x, "res+sacrosanct output")
        self.printEmpty()

        self.printf(4.0, x, "pool input")
        x = self.pool(x)
        self.printf(4.1, x, "pool output")
        x = x.view(x.size(0), -1)
        self.printf(4.2, x, "after view")
        self.printEmpty()
        
        self.printf(5.0, x, "last layer input") #512, 1, 1
        x = self.lastLayer(x)
        # x = self.gap(x)
        self.printf(5.1, x, "last layer output") #10, 1, 1
        self.printEmpty()

        # self.printf(7.0, x)
        return F.log_softmax(x)
        
def create_model():
    model = Net_S13()
    return model
    
    

class LitResnet(LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task='MULTICLASS', num_classes=10)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        BATCH_SIZE = 256 if torch.cuda.is_available() else 64
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // BATCH_SIZE
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}