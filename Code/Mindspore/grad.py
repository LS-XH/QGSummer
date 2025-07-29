from mindspore import Model, LossMonitor
from mindspore import Tensor
from mindspore import nn
from mindspore import dataset
from mindspore import ops
import numpy as np
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

class Itera:
    def __init__(self):
        self.i=-1
    def __iter__(self):
        self.i = -1
        return self
    def __next__(self):
        self.i+=1
        if self.i>5:
            raise StopIteration
        else:
            return np.array([self.i]).astype(np.float32),np.array([self.i*2+1]).astype(np.float32)

mynet = Net()

opti=nn.SGD(mynet.get_parameters(), learning_rate=0.001)

lossfunc=nn.MSELoss()
datas=dataset.GeneratorDataset(Itera(), ["data","label"])


model=Model(mynet,lossfunc,opti)
model.train(100,datas,callbacks=[LossMonitor()])


print(mynet(Tensor([3]).float()))
