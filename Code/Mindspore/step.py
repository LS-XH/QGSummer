from mindspore import Model, LossMonitor
from mindspore import Tensor
from mindspore import nn
from mindspore import dataset
from mindspore import ops
import numpy as np
from mindspore.dataset import Dataset
from mindspore.dataset.core.validator_helpers import is_iterable

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1=nn.Linear(1, 10)
        self.lin2=nn.Linear(10, 10)
        self.lin3=nn.Linear(10, 1)

        self.relu = nn.ReLU()

    def construct(self, x):

        x=self.relu(self.lin1(x))
        x=self.relu(self.lin2(x))
        x=self.relu(self.lin3(x))
        return x



def gener():
    for i in range(5):
        yield np.array([i]).astype(np.float32),np.array([i*2+1]).astype(np.float32)

data=dataset.GeneratorDataset(gener(),["data","label"])
net = Net()
loss_fn=nn.MSELoss()
optim=nn.SGD(net.get_parameters(),learning_rate=0.001)

loss_net = nn.WithLossCell(net, loss_fn)
train_net = nn.TrainOneStepCell(loss_net, optim)

for i in range(100):
    for x,label in data:
        loss=train_net(x, label)
        print(loss)

print(net(Tensor([3]).float()))