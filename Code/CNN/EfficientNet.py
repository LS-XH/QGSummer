import os

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torchvision import datasets, transforms
from functorch.dim import Tensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image

__all__ = ['MobileNetV2']

class MobileNetV2(nn.Module):
    def __init__(self,class_num,struct):
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
                convs.append(DWS(in_channel,out_channel,t=t,stride=s if i==0 else 1))
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


