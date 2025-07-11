import os

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torchvision import datasets, transforms
from functorch.dim import Tensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ConvBN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1,nl:nn.Module=nn.ReLU6()):
        super(ConvBN, self).__init__()

        layers = []
        layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=int((kernel_size-1)/2),groups=groups, bias=False))
        layers.append(nn.BatchNorm2d(out_channel))
        layers.append(nl)

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)



class Bottleneck(nn.Module):
    def __init__(self,in_channel:int,out_channel:int,middle_channel:int=0,kernel_size=3,stride=1,nl:nn.Module=nn.ReLU6()):
        super(Bottleneck, self).__init__()

        self.use_residual = stride==1 and in_channel==out_channel
        if middle_channel == 0: middle_channel = in_channel


        layers = []
        #PW
        layers.append(ConvBN(in_channel,middle_channel,kernel_size=1,stride=1,nl=nl))

        #DW
        layers.append(ConvBN(middle_channel,middle_channel,kernel_size=kernel_size,stride=stride,groups=middle_channel,nl=nl))

        #PW
        layers.append(nn.Conv2d(middle_channel,out_channel,kernel_size=1,stride=1,padding=0,bias=False))
        layers.append(nn.BatchNorm2d(out_channel))

        self.conv = nn.Sequential(*layers)



    def forward(self,x):
        if self.use_residual:
            return self.conv(x) + x
        else:
            return self.conv(x)


mn2=(
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
)

class MobileNetV2(nn.Module):
    def __init__(self,class_num,struct=mn2):
        super().__init__()

        in_channel = 3
        out_channel = 32


        self.conv1=nn.Conv2d(in_channel,out_channel,3,stride=2,padding=1)
        self.bn1=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU6()

        convs=[]
        in_channel = out_channel
        for t,c,n,s in struct:
            out_channel=c
            for i in range(n):
                convs.append(Bottleneck(in_channel,out_channel,t*in_channel,stride=s if i==0 else 1))
                in_channel=out_channel

        self.features=nn.Sequential(*convs)


        self.conv2=nn.Conv2d(in_channel,1280,1,stride=1,padding=0)
        self.bn2=nn.BatchNorm2d(1280)
        self.relu2=nn.ReLU6()

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.flatten=nn.Flatten()
        self.dropout=nn.Dropout(0.2)
        self.fc=nn.Linear(1280,class_num)



    def forward(self,x:Tensor):
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.features(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu2(out)

        out=self.avgpool(out)
        out=self.flatten(out)
        out=self.dropout(out)
        out=self.fc(out)
        return out


