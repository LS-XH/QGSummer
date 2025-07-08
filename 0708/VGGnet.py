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

dataloader = DataLoader(data_train, batch_size=64, shuffle=True,drop_last=False)
testloader = DataLoader(data_test, batch_size=1, shuffle=True,drop_last=False)


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


net=VGGnet(5,'vgg11')
net.to(device)
#net.load_state_dict(torch.load('vgg11-vgg16.pth',weights_only=True))

loss_fn=nn.CrossEntropyLoss()
optimizer=optim.Adam(params=net.parameters(),lr=0.0002)

for epoch in range(10):
    for img, label in dataloader:
        optimizer.zero_grad()
        loss = loss_fn(net(img.to(device)), label.to(device))
        loss.backward()
        optimizer.step()
    print(epoch)

#torch.save(net.state_dict(),'vgg11-vgg16.pth')

net.to("cpu")
acc=0
total=0
for img, label in testloader:

    out=net(img)

    if out.argmax(dim=1) == label:
        #2500
        acc+=1
    total += 1
print(acc*1.0/total)
print(acc)
print(total)