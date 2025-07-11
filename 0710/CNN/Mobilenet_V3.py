import os

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torchvision import datasets, transforms
from functorch.dim import Tensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from CNN.Mobilenet_V2 import Bottleneck,ConvBN







class SqueezeExcitation(nn.Module):
    def __init__(self, channel:int,multiplier:int=4):
        super(SqueezeExcitation, self).__init__()
        self.channel = channel

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1=nn.Conv2d(channel,channel*multiplier,kernel_size=1)
        self.relu=nn.ReLU()
        self.fc2=nn.Conv2d(channel*multiplier,channel,kernel_size=1)
        self.hsigmoid=nn.Hardsigmoid()


    def forward(self, x):
        out = x
        out = self.avg_pool(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.hsigmoid(out)
        return out*x



class BottleneckSE(nn.Module):
    def __init__(self,in_channel:int,out_channel:int,middle_channel:int=0,kernel_size=3,stride=1,nl:nn.Module=nn.ReLU()):
        super(BottleneckSE, self).__init__()
        if middle_channel==0:middle_channel=in_channel
        self.use_residual = stride==1 and in_channel==out_channel

        layers = []
        #PW
        self.PWup=ConvBN(in_channel,middle_channel,kernel_size=1,stride=1,nl=nl)

        #DW
        self.DW=ConvBN(middle_channel, middle_channel, kernel_size=kernel_size, stride=stride,groups=middle_channel,nl=nl)

        self.SE=SqueezeExcitation(middle_channel)

        # PW
        self.PWdown = nn.Sequential(
            nn.Conv2d(middle_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self,x):
        out = self.PWup(x)
        out = self.DW(out)
        out = self.SE(out)
        out = self.PWdown(out)

        if self.use_residual:
            return out + x
        else:
            return out


mnv3=(
    #in  k  hid out   SE    nl   s
    [16, 3, 16, 16, False, "RE", 1],
    [16, 3, 64, 24, False, "RE", 2],
    [24, 3, 72, 24, False, "RE", 1],
    [24, 5, 72, 40, True, "RE", 2],
    [40, 5, 120, 40, True, "RE", 1],
    [40, 5, 120, 40, True, "RE", 1],
    [40, 3, 240, 80, False, "HS", 2],
    [80, 3, 200, 80, False, "HS", 1],
    [80, 3, 184, 80, False, "HS", 1],
    [80, 3, 184, 80, False, "HS", 1],
    [80, 3, 480, 112, True, "HS", 1],
    [112, 3, 672, 112, True, "HS", 1],
    [112, 5, 672, 160, True, "HS", 2],
    [160, 5, 960, 160, True, "HS", 1],
    [160, 5, 960, 160, True, "HS", 1]
)

class MobileNetV3(nn.Module):
    def __init__(self,class_num,struct=mnv3):
        super().__init__()

        in_channel = 3
        out_channel = 16

        self.convbn1=ConvBN(in_channel,out_channel,kernel_size=3,stride=2,nl=nn.Hardswish())

        convs=[]
        for inc,k,midc,outc,use_se,nl,stride in struct:
            in_channel = inc
            out_channel = outc
            if use_se:
                convs.append(BottleneckSE(inc, outc, midc,kernel_size=k, stride=stride, nl=nn.ReLU() if nl == "RE" else nn.Hardswish()))
            else:
                convs.append(Bottleneck(inc, outc, midc,kernel_size=k, stride=stride, nl=nn.ReLU() if nl == "RE" else nn.Hardswish()))

        self.features=nn.Sequential(*convs)

        self.convbn2=ConvBN(in_channel,960,kernel_size=1,stride=1,nl=nn.Hardswish())
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.flatten = nn.Flatten()

        self.fc=nn.Sequential(nn.Linear(960,1280),nn.Hardswish(),nn.Dropout(0.2),nn.Linear(1280,class_num))




        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


    def forward(self,x:Tensor):
        out=self.convbn1(x)


        out=self.features(out)


        out=self.convbn2(out)


        out=self.avgpool(out)

        out=self.flatten(out)
        out=self.fc(out)
        return out


