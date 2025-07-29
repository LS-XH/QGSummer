import os

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torchvision import datasets, transforms
from functorch.dim import Tensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image


from Leo.CNN.Mobilenet_V3 import SqueezeExcitation

__all__=['ResNet','ResNeXt']



class ResBlock(nn.Module):
    def __init__(self, out_channel:int, downsample=False,bottleneck=False,firstlayer=False):
        super().__init__()
        layers=[]

        self.downsample=downsample
        self.out_channel=out_channel
        self.firstlayer=firstlayer
        #捷径下采样
        if downsample:self.ConvDown=nn.Conv2d(int(out_channel/2), out_channel, kernel_size=1, stride=2, padding=0)
        #主脉判断下采样
        if not downsample:
            if not bottleneck:
                layers=[nn.Conv2d(out_channel,out_channel, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_channel)]
            else:
                if not firstlayer:
                    layers=[nn.Conv2d(out_channel, int(out_channel/4), kernel_size=1, stride=1, padding=0),nn.BatchNorm2d(int(out_channel/4))]
                else:
                    layers=[nn.Conv2d(int(out_channel/4), int(out_channel/4), kernel_size=1, stride=1, padding=0),nn.BatchNorm2d(int(out_channel/4))]

        else:
            if not bottleneck:
                layers=[nn.Conv2d(int(out_channel/2), out_channel, kernel_size=3, stride=2, padding=1),nn.BatchNorm2d(out_channel)]
            else:
                layers=[nn.Conv2d(int(out_channel/2), int(out_channel/4), kernel_size=1, stride=2, padding=0),nn.BatchNorm2d(int(out_channel/4))]

        layers.append(nn.ReLU())

        if not bottleneck:
            layers.append(nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channel))
        else:
            layers.append(nn.Conv2d(int(out_channel/4), int(out_channel/4), kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(int(out_channel/4)))
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(int(out_channel/4), out_channel, kernel_size=1, stride=1, padding=0))
            layers.append(nn.BatchNorm2d(out_channel))


        self.block = nn.Sequential(*layers)
    def forward(self, x):
        out = self.block(x)
        if self.downsample:
            out += self.ConvDown(x)
        elif not self.firstlayer:
            out += x
        return nn.functional.relu(out)


class Bottleneck(nn.Module):
    def __init__(self, in_channel,out_channel, stride=1,expansion=4,groups=1,use_se=False):
        super(Bottleneck, self).__init__()

        middle = int(in_channel*stride/expansion)
        self.conv1 = nn.Conv2d(in_channel,middle , kernel_size=1,padding=0,bias=False)
        self.bn1 = nn.BatchNorm2d(middle)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(middle, middle, kernel_size=3,stride=stride,padding=1,groups=groups,bias=False)
        self.bn2 = nn.BatchNorm2d(middle)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(middle, out_channel, kernel_size=1,padding=0,bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)

        if stride==1 and in_channel==out_channel:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1,padding=0,stride=stride,bias=False),
                nn.BatchNorm2d(out_channel)
            )

        if use_se:
            self.se=SqueezeExcitation(middle)
        else:
            self.se = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        if self.se is not None:
            out = self.se(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            return out + self.downsample(x)
        else:
            return out + x

class ResLayer(nn.Module):
    def __init__(self,in_channel, out_channel,block_num, bottleneck=False,firstlayer=False,groups=1,use_se=False):
        super().__init__()
        if bottleneck:
            if firstlayer:
                self.firstlayer = Bottleneck(in_channel, out_channel, stride=1, expansion=1,groups=groups,use_se=use_se)
            else:
                self.firstlayer = Bottleneck(in_channel, out_channel, stride=2,groups=groups,use_se=use_se)

            blocks = []
            for i in range(block_num-1):
                blocks.append(Bottleneck(out_channel, out_channel,groups=groups,use_se=use_se))
            self.block = nn.Sequential(*blocks)


    def forward(self, x):
        out = x
        out = self.firstlayer(out)
        out = self.block(out)
        return out



class ResNet(nn.Module):
    def __init__(self,class_num,layers_structure,bottleneck=False,groups = 1,use_se=False):
        super().__init__()

        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        res=[]

        in_channel=64
        out_channel=256
        for i,block_num in enumerate(layers_structure):
            res.append(ResLayer(in_channel,out_channel,block_num,firstlayer= i==0,bottleneck=bottleneck,groups=groups,use_se=use_se))

            in_channel=out_channel
            out_channel*=2

        self.res = nn.Sequential(*res)
        self.avgpool = nn.AvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_channel*7*7,class_num)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x:Tensor):
        out=x
        out=self.conv1(out)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.maxpool(out)

        out=self.res(out)

        out=self.avgpool(out)
        out=self.flatten(out)
        out=self.fc(out)
        return out

class ResNeXt(ResNet):
    def __init__(self,class_num,layers_structure,bottleneck=False,use_se=False):
        super().__init__(class_num,layers_structure,bottleneck,32,use_se)
