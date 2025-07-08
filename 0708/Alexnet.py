import os

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torchvision import datasets, transforms
from functorch.dim import Tensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



#datapath=R"E:\Program\$knowlage\python\dataset\hymenoptera_data\train"
#testpath=R"E:\Program\$knowlage\python\dataset\hymenoptera_data\val"

datapath=R"E:\Program\python\test3\flower_data\train"
testpath=R"E:\Program\python\test3\flower_data\val"

data_train = datasets.ImageFolder(datapath,transform=transforms.Compose([
    transforms.Resize((227,227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
]))

data_test = datasets.ImageFolder(testpath,transform=transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
]))

dataloader = DataLoader(data_train, batch_size=64, shuffle=True,drop_last=False)
testloader = DataLoader(data_test, batch_size=1, shuffle=True,drop_last=False)



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
            nn.Conv2d(in_channels=3,out_channels=48,kernel_size=11,stride=4,padding=0),
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


net=AlexNet(5)
net.to(device)

loss_fn=nn.CrossEntropyLoss()
optimizer=optim.Adam(params=net.parameters(),lr=0.0002)

for epoch in range(10):
    for img, label in dataloader:
        optimizer.zero_grad()
        loss = loss_fn(net(img.to(device)), label.to(device))
        loss.backward()
        optimizer.step()
    print(epoch)



net.to("cpu")
acc=0
total=0
for img, label in testloader:

    out=net(img)

    if out.argmax(dim=1) == label:
        print(out)
        print(label)
        acc+=1
    total += 1
print(acc*1.0/total)
