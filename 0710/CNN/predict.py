import os

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torchvision import datasets, transforms
from functorch.dim import Tensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from CNN.Alexnet import AlexNet
from CNN.Mobilenet_V2 import MobileNetV2
from CNN.VGGnet import VGGnet
from CNN.Resnet import ResNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

datapath=R"E:\Program\python\test3\flower_data\train"
testpath=R"E:\Program\python\test3\flower_data\val"

data_train = datasets.ImageFolder(datapath,transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
]))

data_test = datasets.ImageFolder(testpath,transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
]))

dataloader = DataLoader(data_train, batch_size=128, shuffle=True,drop_last=False)
testloader = DataLoader(data_test, batch_size=1, shuffle=True,drop_last=False)



ckpts = [

]

for ckpt in ckpts:
    net = torch.load(ckpt)

    net.to("cpu")
    acc = 0
    total = 0
    for index, (img, label) in enumerate(testloader):

        out = net(img)

        if out.argmax(dim=1) == label:
            # 2500
            acc += 1
        total += 1
    print()
    print("ckpt:" + ckpt)
    print(str(acc) + "/" + str(total) + "=" + "acc:" + str(acc * 1.0 / total))