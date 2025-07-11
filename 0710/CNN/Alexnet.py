import os

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torchvision import datasets, transforms
from functorch.dim import Tensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=1),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=3,stride=3,ceil_mode=False),
            nn.Conv2d(in_channels=6,out_channels=12,kernel_size=3,stride=1,padding=1),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=3,stride=3,ceil_mode=False),
            nn.Flatten()

        )
        self.linear=nn.Sequential(
            nn.Linear(in_features=144,out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=2),
            nn.ReLU(),
        )

    def forward(self,x:Tensor):
        out=x
        out=self.lin1(out)
        out=self.lin2(out)
        return out

class AlexNet(nn.Module):
    def __init__(self,class_num):
        super().__init__()
#224*224*3
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=48,kernel_size=11,stride=4,padding=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=48,out_channels=128,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=128,out_channels=192,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=192,out_channels=192,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Flatten()

        )
        self.linear=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128*6*6,out_features=2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048,out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048,out_features=class_num),
        )

    def forward(self,x:Tensor):
        out=x
        out=self.conv(out)
        out=self.linear(out)
        return out


