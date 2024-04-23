"""
ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch.nn as nn
import torch.nn.functional as F


# class BasicBlock(nn.Module):
    # expansion = 1

    # def __init__(self, in_planes, planes, stride=1):
        # super(BasicBlock, self).__init__()
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)

        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion*planes:
            # self.shortcut = nn.Sequential(
                # nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(self.expansion*planes)
            # )

    # def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        # out += self.shortcut(x)
        # out = F.relu(out)
        # return out


# class ResNet(nn.Module):
    # def __init__(self, block, num_blocks, num_classes=10):
        # super(ResNet, self).__init__()
        # self.in_planes = 64

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512*block.expansion, num_classes)

    # def _make_layer(self, block, planes, num_blocks, stride):
        # strides = [stride] + [1]*(num_blocks-1)
        # layers = []
        # for stride in strides:
            # layers.append(block(self.in_planes, planes, stride))
            # self.in_planes = planes * block.expansion
        # return nn.Sequential(*layers)

    # def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        # return out


# def ResNet18():
    # return ResNet(BasicBlock, [2, 2, 2, 2])


# def ResNet34():
    # return ResNet(BasicBlock, [3, 4, 6, 3])


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

        self.aGAP = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten(1, -1)
        self.gap = nn.AvgPool2d(avg)
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
        y = x #sacrosanct path
        self.printf(1.2, x, "Layer1 resnet input")
        x = self.resNetLayer1Part1(x)
        x = self.drop(x)
        x = self.resNetLayer1Part2(x)
        self.printf(1.3, x, "Layer1 resnet output")
        x = x + y
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
        y = x  #sacrosanct path
        self.printf(3.2, x, "Layer3 resnet input")
        x = self.resNetLayer2Part1(x)
        x = self.drop(x)
        x = self.resNetLayer2Part2(x)
        self.printf(3.3, x, "Layer3 resnet output")
        x = x + y
        x = self.drop(x)
        self.printf(3.4, x, "res+sacrosanct output")
        self.printEmpty()

        self.printf(4.0, x, "pool input")
        x = self.pool(x)
        self.printf(4.1, x, "pool output")
        self.printEmpty()

        # x = x.view(-1, 10)
        self.printf(4.1, x, "For showing before last layer")
        x = x.view(x.size(0), -1)
        self.printf(6.0, x, "last layer input") #512, 1, 1
        x = self.lastLayer(x)
        # x = self.gap(x)
        self.printf(6.1, x, "last layer input") #10, 1, 1
        self.printEmpty()

        # self.printf(7.0, x)
        return F.log_softmax(x)