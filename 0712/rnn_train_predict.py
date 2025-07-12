import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

from Leo.RNN import *


from Leo.data import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



imdbvocab= {
    "<PAD>": 0,
    "<BOS>": 1,
    "<EOS>": 2,
    "<NUL>": 3,
}
for s in open(R"E:\Program\$knowlage\python\dataset\aclImdb\imdb.vocab",'r',encoding="utf-8").read().splitlines():
    imdbvocab[s]=len(imdbvocab)



text_data_train=TextDataset(R"E:\Program\$knowlage\python\dataset\aclImdb\train\neg",0,vocab_dic=imdbvocab,update_vocab=True)
text_data_train.append(R"E:\Program\$knowlage\python\dataset\aclImdb\train\pos",1)

text_data_test=TextDataset(R"E:\Program\$knowlage\python\dataset\aclImdb\test\neg",0,vocab_dic=imdbvocab,update_vocab=True)
text_data_test.append(R"E:\Program\$knowlage\python\dataset\aclImdb\test\pos",1)

train_loader = TextLoader(text_data_train,batch_size=512)
test_loader = TextLoader(text_data_test,batch_size=1)

net=LSTM(text_data_train.vocab_size)
net.to(device)


loss_fn=nn.CrossEntropyLoss()
optimizer=optim.Adam(params=net.parameters(),lr=0.0001)

loss_col=[]
for epoch in range(100):
    for index,(text_index, label) in enumerate(train_loader):
        net.train()
        optimizer.zero_grad()
        out=net(text_index.to(device))
        label=label.to(device).long()
        loss = loss_fn(out, label)
        loss.backward()
        optimizer.step()
        net.eval()
        loss_col.append(loss.item())
        print("\repoch:%3d"%epoch+"%20s"%(str(index)+"/"+str(len(train_loader)))+"     loss:"+str(loss.item()),end='')

plt.plot(loss_col)
plt.show()


print()
print()
print()
acc = 0
total = 0
for index, (text, label) in enumerate(test_loader):
    out = net(text.to(device)).to("cpu")

    for o,l in zip(out,label):
        if o.argmax().item() == l.item():
            # 2500
            acc += 1
        total += 1

    print("\rprocess: %20s" % (str(index) + "/" + str(len(test_loader))), end='')

print()
print("acc:  "+str(acc) + "/" + str(total) + "=" + "acc:"+str(acc * 1.0 / total))

while True:
    if net(text_data_train.str2tensor(input("说的道理：")).to(device)).to("cpu")[0].argmax().item() == 1:
        print("positive")
    else:
        print("negative")
