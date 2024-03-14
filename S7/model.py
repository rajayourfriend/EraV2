
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        set1 = 8 #channels
        set2 = 10 #channels
        out = 10 #channels
        avg = 7 #channels
        drop = 0.25 #dropout
        mom = 0.1 #momentum in bn
        self.conv1 = nn.Conv2d(1, set1, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=set1, eps=1e-05, momentum=mom, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(set1, set1, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=set1, eps=1e-05, momentum=mom, affine=True, track_running_stats=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(set1, set2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=set2, eps=1e-05, momentum=mom, affine=True, track_running_stats=True)
        self.conv4 = nn.Conv2d(set2, out, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=out, eps=1e-05, momentum=mom, affine=True, track_running_stats=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(out, out, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=out, eps=1e-05, momentum=mom, affine=True, track_running_stats=True)
        self.conv6 = nn.Conv2d(out, out, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=out, eps=1e-05, momentum=mom, affine=True, track_running_stats=True)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv7 = nn.Conv2d(out, out, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(num_features=out, eps=1e-05, momentum=mom, affine=True, track_running_stats=True)
        self.conv8 = nn.Conv2d(out, out, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(num_features=out, eps=1e-05, momentum=mom, affine=True, track_running_stats=True)
        self.conv9 = nn.Conv2d(out, out, 3)
        self.bn9 = nn.BatchNorm2d(num_features=out, eps=1e-05, momentum=mom, affine=True, track_running_stats=True)


        self.drop = nn.Dropout(drop)
        self.gap = nn.AvgPool2d(kernel_size=[avg,avg], stride=[avg,avg], padding=0, ceil_mode=False, count_include_pad=False)

    def forward(self, x):
        x = self.drop(self.pool1(self.bn2(F.relu(self.conv2(self.bn1(F.relu(self.conv1(x))))))))
        x = self.drop(self.pool2(self.bn4(F.relu(self.conv4(self.bn3(F.relu(self.conv3(x))))))))
        x = self.drop(self.pool3(self.bn6(F.relu(self.conv6(self.bn5(F.relu(self.conv5(x)))))))) # ToDo Try adding MP here
        x = self.drop(self.bn8(F.relu(self.conv8(self.bn7(F.relu(self.conv7(x))))))) # ToDo Try adding MP here
        x = self.conv9(x)
        #x = self.gap(x) # Raja ToDo Try printing shape here
        #print(x.shape)
        x = x.view(-1, 10) # Raja ToDo Try printing shape here
        return F.log_softmax(x)


class Net4orig(nn.Module):
    def __init__(self):
        super(Net4orig, self).__init__()
        set1 = 8 #channels
        set2 = 16 #channels
        out = 10 #channels
        avg = 7 #channels
        drop = 0.25 #dropout
        mom = 0.1 #momentum in bn
        self.conv1 = nn.Conv2d(1, set1, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=set1, eps=1e-05, momentum=mom, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(set1, set1, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=set1, eps=1e-05, momentum=mom, affine=True, track_running_stats=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(set1, set2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=set2, eps=1e-05, momentum=mom, affine=True, track_running_stats=True)
        self.conv4 = nn.Conv2d(set2, out, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=out, eps=1e-05, momentum=mom, affine=True, track_running_stats=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(out, out, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=out, eps=1e-05, momentum=mom, affine=True, track_running_stats=True)
        self.conv6 = nn.Conv2d(out, out, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=out, eps=1e-05, momentum=mom, affine=True, track_running_stats=True)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv7 = nn.Conv2d(out, out, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(num_features=out, eps=1e-05, momentum=mom, affine=True, track_running_stats=True)
        self.conv8 = nn.Conv2d(out, out, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(num_features=out, eps=1e-05, momentum=mom, affine=True, track_running_stats=True)
        self.conv9 = nn.Conv2d(out, out, 3)
        self.bn9 = nn.BatchNorm2d(num_features=out, eps=1e-05, momentum=mom, affine=True, track_running_stats=True)


        self.drop = nn.Dropout(drop)
        self.gap = nn.AvgPool2d(kernel_size=[avg,avg], stride=[avg,avg], padding=0, ceil_mode=False, count_include_pad=False)

    def forward(self, x):
        x = self.drop(self.pool1(self.bn2(F.relu(self.conv2(self.bn1(F.relu(self.conv1(x))))))))
        x = self.drop(self.pool2(self.bn4(F.relu(self.conv4(self.bn3(F.relu(self.conv3(x))))))))
        x = self.drop(self.pool3(self.bn6(F.relu(self.conv6(self.bn5(F.relu(self.conv5(x)))))))) # ToDo Try adding MP here
        x = self.drop(self.bn8(F.relu(self.conv8(self.bn7(F.relu(self.conv7(x))))))) # ToDo Try adding MP here
        x = self.conv9(x)
        #x = self.gap(x) # Raja ToDo Try printing shape here
        #print(x.shape)
        x = x.view(-1, 10) # Raja ToDo Try printing shape here
        return F.log_softmax(x)
