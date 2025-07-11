import os

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torchvision import datasets, transforms
from functorch.dim import Tensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image


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

class ResLayer(nn.Module):
    def __init__(self, out_channel,block_num,downsample=False, bottleneck=False):
        super().__init__()
        self.downsample = ResBlock(out_channel,downsample=downsample,bottleneck=bottleneck,firstlayer=not downsample)


        blocks = []
        for i in range(block_num-1):
            blocks.append(ResBlock(out_channel,downsample=False,bottleneck=bottleneck))
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        out = x
        out = self.downsample(out)
        out = self.block(out)
        return out



class ResNet(nn.Module):
    def __init__(self,class_num,layers_structure,bottleneck=False):
        super().__init__()

        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        res=[]

        out_channel=0
        for i,block_num in enumerate(layers_structure):
            out_channel=64*2**i
            if bottleneck:out_channel*=4
            res.append(ResLayer(out_channel,block_num,i!=0,bottleneck=bottleneck))

        self.res = nn.Sequential(*res)
        self.avgpool = nn.AvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_channel*7*7,class_num)
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

