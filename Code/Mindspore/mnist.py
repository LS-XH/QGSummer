import tkinter as tk
from PIL import ImageGrab, Image
import numpy as np
import matplotlib.pyplot as plt

from mindspore import Tensor, Model, LossMonitor
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore.dataset import vision
import mindspore.ops as ops

def preprocess(image):
    image = vision.Rescale(1.0 / 255.0, 0.0)(image)  # 归一化到 [0,1]
    return image
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()

        self.lin1=nn.Dense(784, 256)
        self.lin2=nn.Dense(256, 64)
        self.lin3=nn.Dense(64, 10)

        self.relu = nn.ReLU()


    def construct(self, x):
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.lin3(x)
        return x

    def getvalue(self,x):
        output = self(Tensor(x).reshape(1,784))
        maxi = 0
        for i in range(10):
            if output[0][i]>output[0][maxi]:
                maxi = i
        return maxi


batch_size = 256

dataset=ds.MnistDataset('./dataset/mnist')
dataset=dataset.map(preprocess).batch(batch_size,True)


net = Net()
loss = nn.MSELoss()
opti = nn.SGD(net.get_parameters(),learning_rate=0.01,momentum=0.9)

loss_net = nn.WithLossCell(net, loss_fn=loss)
train_net = nn.TrainOneStepCell(loss_net,opti)

loss_val=[]

for epoch in range(3):
    for index, [img, lab] in enumerate(dataset):
        img = Tensor(img).reshape(batch_size, 784).float()
        lab = ops.one_hot(lab.int(), 10).reshape(batch_size, 10).float()

        loss_val.append(train_net(img, lab))
        print(index)

plt.plot(loss_val)
plt.show()

testdata=ds.MnistDataset('./dataset/test')
testdata=testdata.map(preprocess)

'''
for index, [img, lab] in enumerate(testdata):
    img = Tensor(img).reshape(1, 784).float()
    print(lab)
    lab = ops.one_hot(lab.int(), 10).reshape(1, 10).float()


    print(net.getvalue(img))
    print()
    print()
    print()
    print()
'''

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写画布")

        # 创建画布
        self.canvas = tk.Canvas(root, width=280, height=280, bg='black')
        self.canvas.pack()

        # 绑定鼠标事件
        self.canvas.bind("<B1-Motion>", self.draw)

        # 添加按钮
        btn_frame = tk.Frame(root)
        btn_frame.pack()
        tk.Button(btn_frame, text="识别", command=self.predict).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="清空", command=self.clear).pack(side=tk.LEFT)

        # 初始化画笔
        self.last_x, self.last_y = None, None

    def draw(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=30, fill='white', capstyle=tk.ROUND, smooth=True
            )
        self.last_x, self.last_y = event.x, event.y

    def clear(self):
        self.canvas.delete("all")
        self.last_x, self.last_y = None, None

    def predict(self):
        # 将画布转为28x28灰度图像（适配MNIST）
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        xx = x + self.canvas.winfo_width()
        yy = y + self.canvas.winfo_height()
        img = ImageGrab.grab((x+2, y+2, xx-2, yy-2)).convert('L')
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        img.save("./dataset/input.png")
        # 转换为numpy数组并归一化
        img_array = np.array(img)
        img_array = img_array  # 反色（因MNIST是白底黑字）
        print(img_array)
        # 此处添加模型预测代码
        print("输入数据形状:", net.getvalue(preprocess(img_array)))  # 示例输出



root = tk.Tk()
app = DrawingApp(root)
root.mainloop()

