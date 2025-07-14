import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


from Leo.data.dataset import *
from Leo.RNN.transformer_net import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train_path=R"E:\Program\$knowlage\python\dataset\ag_news_csv\train.csv"
test_path=R"E:\Program\$knowlage\python\dataset\ag_news_csv\test.csv"


a,b=Text2TextDataset.file2data(train_path)
c,d=Text2TextDataset.file2data(test_path)
vocab=Vocab(a+b+c+d)


train_data = Text2TextDataset(train_path,origin_vocab=vocab)
train_loader = DataLoader(train_data, batch_size=512, shuffle=True,collate_fn=train_data.collate_fn)

test_data = Text2TextDataset(test_path,origin_vocab=vocab)
test_loader = DataLoader(train_data, batch_size=1, shuffle=True,collate_fn=test_data.collate_fn)




net = TransformerNet(vocab)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(net.parameters(), lr=0.0003)
net.to(device)
if not os.path.exists(os.path.join(os.path.abspath(os.curdir),"model.pth")):
    loss_col = []
    for epoch in range(5):
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            out = net(x.to(device), y.to(device)[:, 0:-1])
            label = y[:, 1:]

            out = out.reshape(out.size(0) * out.size(1), out.size(2)).to(device)
            label = label.reshape(label.size(0) * label.size(1)).to(device)

            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()

            loss_col.append(loss.item())
            print(
                "\repoch:%3d" % epoch + "%20s" % (str(i) + "/" + str(len(train_loader))) + "     loss:" + str(
                    loss.item()),
                end='')
    plt.plot(loss_col)
    plt.show()
    torch.save(net.state_dict(), "model.pth")
else:
    net.load_state_dict(torch.load("model.pth",weights_only=True))

for x,y in test_loader:
    x=x[0]
    y=y[0]
    print(vocab.index2str(x.tolist()))
    print(vocab.index2str(y.tolist()))
    out = net.predict(x.to(device))
    print(vocab.index2str(out.tolist()))

    print()
    print()
    print()

xx=[0]*50
yy=[0]*12
xm=0
ym=0
for x,y in test_loader:
    print(x.shape,y.shape)
    if x.shape[1]+1>50:
        xm+=1
    else:
        xx[x.shape[1]] += 1
    if y.shape[1]+1>12:
        ym+=1
    else:
        yy[y.shape[1]]+=1

plt.plot(xx)
plt.plot(yy)
plt.show()

print(xm,ym)