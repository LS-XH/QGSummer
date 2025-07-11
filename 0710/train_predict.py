import os

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torchvision import datasets, transforms
from functorch.dim import Tensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from matplotlib import pyplot as plt


from CNN import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

flower_train = datasets.ImageFolder(R"E:\Program\python\test3\flower_data\train",transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
]))

flower_test = datasets.ImageFolder(R"E:\Program\python\test3\flower_data\val",transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
]))

dataloader = DataLoader(flower_train, batch_size=128, shuffle=True,drop_last=False)
testloader = DataLoader(flower_test, batch_size=1, shuffle=True,drop_last=False)


mn2=[
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]
cfgs={
    'vgg11':[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'vgg13':[64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'vgg16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
    'vgg19':[64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M'],

}
alexnet=AlexNet(5)
vggnet=VGGnet(5,'vgg11')
resnet=ResNet(5,[3,4,6,3],bottleneck=False)
mbv2net=MobileNetV2(5,mn2)
mbv3net=MobileNetV3(5)
net=vgg
net.to(device)
#net.load_state_dict(torch.load('vgg11-vgg16.pth',weights_only=True))

loss_col=[]

loss_fn=nn.CrossEntropyLoss()
optimizer=optim.Adam(params=net.parameters(),lr=0.0001)

for epoch in range(10):
    for index,(img, label) in enumerate(dataloader):
        net.train()
        optimizer.zero_grad()
        loss = loss_fn(net(img.to(device)), label.to(device))
        loss.backward()
        optimizer.step()
        net.eval()
        loss_col.append(loss.item())

        print("\repoch:%3d"%epoch+"%10s"%(str(index)+"/"+str(len(dataloader))),end='')


plt.plot(loss_col)
plt.show()



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
print(str(acc) + "/" + str(total) + "=" + "acc:"+str(acc * 1.0 / total))