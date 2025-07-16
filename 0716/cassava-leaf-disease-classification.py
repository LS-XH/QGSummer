import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torch.utils.data as data

from Leo.CNN import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def spliter(data_path:str,csv_path):
    label_name={0: "Cassava Bacterial Blight (CBB)",
           1: "Cassava Brown Streak Disease (CBSD)",
           2: "Cassava Green Mottle (CGM)",
           3: "Cassava Mosaic Disease (CMD)",
           4: "Healthy"
           }
    path = R"E:\Program\$knowlage\python\dataset\cassava-leaf-disease-classification\train_data"
    df = pd.read_csv(csv_path)
    img_ids = df['image_id'].tolist()
    label_ids = df['label'].tolist()

    for key in label_name:
        label_dir=os.path.join(path, label_name[key])
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)


    for i,(img_id,label_id) in enumerate(zip(img_ids,label_ids)):
        shutil.copy(os.path.join(data_path, img_id), os.path.join(path, label_name[label_id], img_id))
        print("\rstep:           "+str(i)+"/"+str(len(label_ids)))

#spliter(R"E:\Program\$knowlage\python\dataset\cassava-leaf-disease-classification\train_images",R"E:\Program\$knowlage\python\dataset\cassava-leaf-disease-classification\train.csv")

train_data = torchvision.datasets.ImageFolder(R"E:\Program\$knowlage\python\dataset\cassava-leaf-disease-classification\leaf_data\train",transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

]))

test_data = torchvision.datasets.ImageFolder(R"E:\Program\$knowlage\python\dataset\cassava-leaf-disease-classification\leaf_data\val",transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

]))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=True)

net = ResNet(5,[3,4,6,3],bottleneck=True)
#net = VGGnet(5,'vgg11')
net.to(device)
#net.load_state_dict(torch.load('vgg11-vgg16.pth',weights_only=True))

loss_col=[]

loss_fn=nn.CrossEntropyLoss()
optimizer=optim.Adam(params=net.parameters(),lr=0.0001)

for epoch in range(2):
    for index,(img, label) in enumerate(train_loader):
        net.train()
        optimizer.zero_grad()
        loss = loss_fn(net(img.to(device)), label.to(device))
        loss.backward()
        optimizer.step()
        net.eval()
        loss_col.append(loss.item())

        print("\repoch:%3d"%epoch+"%10s"%(str(index)+"/"+str(len(train_loader)))+ "     loss:" + str(loss.item()),end='')
    #torch.save(net.state_dict(), "res3-epoch"+str(epoch)+".pth")

#net.load_state_dict(torch.load("res3-epoch2.pth",weights_only=True))

plt.plot(loss_col)
plt.show()

print()
print()
print()

acc = 0
total = 0
for index, (img, label) in enumerate(test_loader):
    out = net(img.to(device))
    for N in range(out.shape[0]):
        if out[N,:].argmax() == label[0].to(device):
            acc += 1
        total += 1

    print("\rstep:%5d"%index + ("/" + str(len(test_loader))),end='')

print()
print(str(acc) + "/" + str(total) + "=" + "acc:"+str(acc * 1.0 / total))
