import os

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torchvision import datasets, transforms
from functorch.dim import Tensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image



cfgs={
    'vgg11':[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'vgg13':[64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'vgg16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
    'vgg19':[64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M'],

}




class VGGnet(nn.Module):
    def __init__(self,class_num,struct):
        super().__init__()
#224*224*3

        convs=[]
        inc=3
        for v in cfgs[struct]:
            if v=='M':
                convs.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                convs.append(nn.Conv2d(in_channels=inc,out_channels=v,kernel_size=3, stride=1,padding=1))
                convs.append(nn.ReLU())
                inc = v

        self.conv=nn.Sequential(*convs)
        self.linear=nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512*7*7,out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096,out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096,out_features=class_num),
        )

    def forward(self,x:Tensor):
        out=x
        out=self.conv(out)
        out=self.linear(out)
        return out

