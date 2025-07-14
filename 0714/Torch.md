

# 环境配置

CUDA 11.8

```
conda install pytorch torchvision torchaudio torchtext pytorch-cuda=11.8 -c pytorch -c nvidia
```



# 文件架构

- ## torch
  
    - ## torch.utils
      
        - ## torch.utils.data   数据部分
          
            - ### *dataset.py  数据模块
              
                - Dataset
            - ### *sampler.py  采样器
              
                - Sampler
                - RandomSampler
            - ### *dataloader.py   数据加载器
        
    - ## torch.nn
      
        - ### *torch.nn.modules   主网络部分
          
            - ### *module.py
              
                - Module
                - Linear
                - Conv1d/2d/3d
                - MaxPool1d/2d/3d
            
        - ### ==functional.py==   函数部分
          
            ##### 非线性激活函数
            
            - relu
            - leaky_relu
            - elu
            - tanh
            - sigmoid
            
            ##### Convolution 函数（卷积）
            
            - conv1d
            - conv2d
            - conv3d
            
            ##### Pooling 函数（池化）
            
            - avg_pool1d/2d/3d
            - max_pool1d/2d/3d
            - max_unpool1d/2d/3d
            
            ##### Loss 函数
            
            - cross_entropy
            
            - mse_loss
            - l1_loss
            - nll_loss
        
    - ## torch.optim   优化器部分
      
        - ### ==*optimizer.py==
          
            - SGD
            - Adam
        
    
    

# 类型使用
## torch

- ### Tensor-Operate：

     - torch.tensor()

          ```py
          #创建tensor
          #dtype: 数据类型，可以是torch.long,torch.float32等
          def tensor(data: Any,
                     dtype: dtype | None = None,
                     device: str | device | int | None = None,
                     requires_grad: bool = False,
                     pin_memory: bool = False) -> Tensor
          
          ```

          

     - torch.stack()

          ```py
          #堆叠tensor
          def stack(tensors: tuple[Tensor, ...] | list[Tensor],dim: int = 0,*,out: Tensor | None = None) 
          ->Tensor
          ```

          

- ### Tensor (class)
  
    - #### 形状
    
        | Layer     | Tensor Shape                                                 |
        | --------- | ------------------------------------------------------------ |
        | Linear    | (batch_size, in_features)                                    |
        | Conv1d    | (batch_size, channels, sequence_length)                      |
        | Conv2d    | (batch_size, channels, height, width)                        |
        | Conv3d    | (batch_size, channels, depth, height, width)                 |
        | MaxPool2d | (batch_size, channels, height, width)                        |
        | AvgPool2d | (batch_size, channels, height, width)                        |
        | RNN       | (batch_size, sequence length（时刻）, vocab_encode_length（embedding后词向量长度）) |
    
        
    
    - #### 常用函数
    
        - ##### dim()->int：返回张量的维度
    
        - ##### ==reshape()==->Tensor：调整张量形状
    
             ```py
             #调整Tensor维度形状，-1表示自动推断当前维度的长度
             #example:
             
             # 1D → 2D (4元素 → 2×2矩阵)
             a = torch.tensor([0., 1., 2., 3.])
             b = a.reshape(2, 2)  # tensor([[0., 1.], [2., 3.]])
             
             # 自动维度推断 (-1)
             c = torch.arange(6)          # tensor([0, 1, 2, 3, 4, 5])
             d = c.reshape(2, -1)          # tensor([[1]][[2]])  (2×3)
             e = c.reshape(3, -1)          # tensor([[1]][[2]]) (3×2)
             
             
             ```
    
             
    
    - #### 反向传播
    
        每个张量（Tensor）在前向传播过程中会记录其依赖的操作（通过`grad_fn`属性）
    
        PyTorch会自动将梯度累积到张量的`.grad`属性中（需注意用`optimizer.zero_grad()`清空）。
    
        - ##### grad_fn：记录其依赖的操作
        - ##### grad：计算梯度时的累积梯度
        - 
    
    - #### 
    

## torch.nn

关于神经网络结构，训练的包，包括Module，层，激活函数，损失函数

### Module (class)

所有网络的基类。

使用方法一般是继承这个类。

```py
#example:

class MyModel(nn.Module):
    #初始化函数，用于初始化一些层
    def __init__(self):
        super(Model, self).__init__()

        #比如可以这样初始化卷积层
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

        #或者这样初始化线性层
        self.lin1 = nn.Linear(1, 100)
        self.lin2 = nn.Linear(100, 50)

    #需要用户定义向前传播的函数，X为输入张量
    def forward(self, x):

        #第一层线性输出，调用了Linear的__call__()
        x = F.relu(self.lin1(x))   

        #第一层激活输出
        return F.relu(self.conv2(x))   
```

### Sequential（class）

封装一组“层”，作为一个小的子网络

```py
#example:
#使用Sequential创建一个小模型。
model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )


#将Sequential与OrderedDict一起使用。
model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))


model(x)
```



### Layers

#### 线性层

- ### Linear (class)

    线性层，对输入数据做线性变换
    $$
    y=Ax+b
    $$

    ```py
    #in_features：输入节点数
    #out_features：输出节点数
    #bias：是否启用偏置
    class torch.nn.Linear(in_features, out_features, bias=True)
    ```
    ```py
    #一般在nn.Module.__init__()中使用
    self.lin1 = nn.Linear(1, 100)
    self.lin2 = nn.Linear(100, 50)
    ```

#### 卷积层

- ### Conv1d (class)

- ### Conv2d（Class）

     ```py
     #in/out_channels(int): 出入通道个数
     #kernel_size(int):
     #stride: 步长
     #padding: 变元填充
     #groups: 将输入，输出，按通道分组，每组仅与自己的卷积核计算，inchannels和outchannels都必须能被组整除（可用于实现Depthwise）
     #bias: 是否启用偏置
     class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
     ```

     

#### 池化层

- ### MaxPool2d

     ```py
     #2D最大池化
     #kernel_size
     #stride=None
     #padding：边缘填0的宽度
     #ceil_mode：多余部分的处理方式，如果为True则会为了多余部分插入0，如果为False则会舍去多余部分
     class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
     ```

- AvgPool2d

     ```py
     #2D平均池化
     #kernel_size
     #stride=None
     #padding：边缘填0的宽度
     class torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
     ```

     

- AdaptiveAvgPool2d

     ```py
     #2D自适应的平均池化
     #output_size: 用户期望的，输出tensor尺寸
     class torch.nn.AdaptiveAvgPool2d(output_size)
     ```

     

#### DropOut Layers

我们只希望在训练模型时使用Dropout层，而预测时不使用，所以要使用`net.train()`和`net.eval`来开启/关闭此方法

- ### Dropout

     ```py
     #p(float): 元素归零的概率。默认值：0.5
     #inplace(bool): 如果设置为True，将就地执行此操作。默认值：False
     class torch.nn.Dropout(p=0.5, inplace=False)
     
     ```

#### 归一化层

- ### BatchNorm2d

     ```py
     #num_features: 特征数，即输入通道数
     #eps: 分母中常常数epsilon
     
     class torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
     
     
     ```


#### 循环层

- #### RNN

     ```py
     #input_size: input的特征数量
     #hidden_size: hidden隐藏状态h中的特征数量
     #num_layers: RNN堆叠的层数，上一层的输出作为下一层的输入，即为RNN的复合
     #nonlinearity: 非线性，即激活函数，可为'tanh'或者'relu'
     #batch_first: 是否将Tensor的batch放在第一个维度
     class torch.nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dropout=0.0, bidirectional=False, device=None, dtype=None)
     
     #网络中传递时：
     #input: input（词向量组），hx（每个词向量的初始hidden值组）
     #output: output（输出向量组），ho（每个词向量传播后的hidden值组）
     ```

- #### LSTM

     ```py
     #input_size: input的特征数量
     #hidden_size: hidden隐藏状态h中的特征数量
     #num_layers: RNN堆叠的层数，上一层的输出作为下一层的输入，即为RNN的复合
     #batch_first: 是否将Tensor的batch放在第一个维度
     class torch.nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None)
     ```

- #### Embedding

     ```py
     #num_embeddings: 词的数量
     #embedding_dim: 词向量的长度
     #padding_idx: padding的序号，如果指定，此序号对应的词向量将一直全为0
     class torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None)
     ```


- #### Transformer

     ```py
     #d_model:模型维度，在NLP中即为序列中每个词向量的长度
     #nhead: Multi-Head中，头的个数
     #num_encoder_layers: 编码器部分重复次数
     #num_decoder_layers: 解码器部分重复次数
     #dim_feedforward: 前馈神经网络维度
     #batch_first: 是否将batch放在最高维度
     class torch.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation=<function relu>, custom_encoder=None, custom_decoder=None, layer_norm_eps=1e-05, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)
     ```

     



### Loss Function

- #### MSELoss

     均方误差损失
     $$
     Loss=\Sigma (\hat{y_i}-y)^2
     $$

     ```py
     loss = nn.MSELoss()
     
     input = torch.randn(3, 5, requires_grad=True)
     
     target = torch.randn(3, 5)
     output = loss(input, target)
     output.backward()
     ```

     

- #### CrossEntropyLoss

     交叉熵损失，**每个数据**的真实标记乘以其**真实标记**的**模型预测概率**
     $$
     Loss=\Sigma_i \Sigma_k ~~y^{(i)}_k*log(p^{(i)}_k)
     $$
     

#### 



## torch.nn.functional

定义了一些函数，这些函数其实在torch.nn里好像都有对应的类，这里只不过是提供了一个简便的使用方式

#### 非线性激活函数

- ### relu

     ```py
     torch.nn.functional.relu(input, inplace=False)
     ```

- ### leaky_relu

     ```py
     torch.nn.functional.leaky_relu(input, negative_slope=0.01, inplace=False)
     ```

- ### elu

     ```py
     torch.nn.functional.elu(input, alpha=1.0, inplace=False)
     ```

- ### tanh

     ```py
     torch.nn.functional.tanh(input)
     ```

- ### sigmoid

     ```py
     torch.nn.functional.sigmoid(input)
     ```

#### 损失函数

- ### mse_loss

     均方误差损失，多用于回归任务

     ```py
     #input(Tensor)：估计值
     #target(Tensor)：真实值
     #size_average(bool)
     #reduce(bool)
     #reduction(str)
     #weight(Tensor)
     torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='mean', weight=None)
     ```

- ### cross_entropy

     多分类交叉熵损失，多用于多分类任务

     ```py
     #input(Tensor)：估计值
     #target(Tensor)：真实值
     torch.nn.functional.cross_entropy(input, target, weight=None, size_average=True)
     ```
     
- ### binary_cross_entropy

     二元交叉熵损失，用于二分类任务（如医学图像诊断、垃圾邮件检测）。

     ```py
     #input(Tensor)：估计值
     #target(Tensor)：真实值
     torch.nn.functional.binary_cross_entropy(input, target, weight=None, size_average=None, reduce=None, reduction='mean')
     ```



#### 工具

- ### one_hot

     ```py
     #tensor: 要转换为独热编码的tensor
     #num_classes: 类别数
     torch.nn.functional.one_hot(tensor, num_classes=-1)
     ```

- 

## torch.utils.data

数据集，训练集，数据处理的包

### Dataset (class)

一般用于继承

```py
class torch.utils.data.Dataset
```


继承时，重写：

```py
class MyDataset(Dataset):
    #override
    def __len__(self):
    def __getitem__(self, index):
```

- #### +（operator）

     数据集之间支持相加（合并）

     ```
     ants_data = Mydata(ants)
     bees_data = Mydata(bees)
     
     data = ants_data+bees_data
     ```

- #### getitem

### DataLoader (class)

```py
#dataset(Dataset)：loader的数据集
#batch_size(int)：训练时一批的大小
#shuffle(bool)：是否打乱数据
#sampler(Sampler)定义从数据集中提取样本的策略
#drop_last(bool)：最后一batch不完整时，是否丢弃
class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False)


#example

```

## torch.utils.data.sampler

包含dataloader采样器的包
- ### Sampler (class)
  
    采样器的基类
    ```py
    class torch.utils.data.sampler.Sampler(data_source)
    ```
    ```py
    class MySampler(Sampler):
        #override
        def __len__(self):
        def __iter__(self, index):
    ```
- ### SequentialSampler (class)
  
    样本元素顺序排列，始终以相同的顺序。
    
    ```py
    class torch.utils.data.sampler.SequentialSampler(data_source)
    ```
- ### RandomSampler (class)
  
    样本元素随机，没有替换。
    ```py
    class torch.utils.data.sampler.RandomSampler(data_source)
    ```
- ### SubsetRandomSampler (class)
    样本元素从指定的索引列表中随机抽取，没有替换。
    ```py
    #indices (list)：索引的列表
    class torch.utils.data.sampler.SubsetRandomSampler(indices)
    ```

## torch.nn.utils

### torch.nn.utils.rnn

- pad_sequence

     ```py
     def pad_sequence(sequences: Tensor | list[Tensor],batch_first: bool = False,padding_value: float = 0.0,padding_side: str = "right") -> Tensor
     ```

     

## torchvision

用于除了图形数据的模块

### torchvision.transform

对PIL.Image进行变换

- #### Compose

     将多个`transform`组合起来使用，有点像Sequential

     ```py
     transforms.Compose([
          transforms.CenterCrop(10),
          transforms.ToTensor(),
      ])
     ```

- #### Pad

     ```py
     #将给定的PIL.Image的所有边用给定的pad value填充
     #padding：要填充多少像素
     #fill：用什么值
     class torchvision.transforms.Pad(padding, fill=0)
     
     transforms.Pad(padding=10, fill=0)
     ```

- #### Resize

     ```py
     #缩放图像
     
     #按照较短边等比例缩放
     transforms.Resize(256)
     
     #缩放
     transforms.Resize((256, 256))  # 注意是元组
     ```

     

- #### Normalize

     ```py
     #给定均值：(R,G,B)(mean) 方差：（R，G，B）(std)，将会把Tensor正则化。即：Normalized_image=(image-mean)/std。
     class torchvision.transforms.Normalize(mean, std)
     ```

- #### CenterCrop

     将给定的`PIL.Image`进行中心切割，得到给定的`size`，`size`可以是`tuple`，`(target_height, target_width)`。

- #### RandomHorizontalFlip

     ```py
     #水平随机翻转
     class torchvision.transforms.RandomHorizontalFlip
     
     transforms.RandomHorizontalFlip()
     ```

     

### torchvision.dataset

- #### ImageFolder（calss）

     一个通用的数据加载器，数据集中的数据以以下方式组织

     ```
     root/dog/xxx.png
     root/dog/xxy.png
     root/dog/xxz.png
     
     root/cat/123.png
     root/cat/nsdf3.png
     root/cat/asd932_.png
     ```

     ```py
     #example
     dataset = datasets.ImageFolder(
         root='F:/root',  # 数据集根目录
         transform=[transform]           # 应用预处理
     )
     ```

- #### class_to_idx（field）

     一次字典，数据label对应的在数据集中的编号

     ```py
     print(data.class_to_idx)
     
     #输出
     {'ants': 0, 'bees': 1}
     ```

     

### torchvision.utils

## torchtext

### torchtext.datasets

```py
#root:
#
class TextClassificationDataset(root,download=True,vectorize=False,text_field=TEXT,label_field=LABEL)
```



## torch.optim

包含模型训练优化器的包

优化器负责：

1. 梯度管理
2. 参数更新
3. 加速收敛
4. 自适应学习率
- ### Optimizer
  
    - ##### zero_grad()：清零所有优化过的梯度
    - ##### step(closure)：进行单次优化（更新训练参数）
    ```py
    for input, target in dataset:
        optimizer.zero_grad()#清零所有优化过的梯度
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()#计算梯度
        optimizer.step()#更新梯度
    ```
    

#### 基础梯度下降法

- ### SGD
  
    随机梯度下降（SDG），可以附带Nesterov（NAG）加速梯度
    ```py
    #params：待优化参数的iterable，一般是model.parameters()
    #lr(float)：学习率，一般设为0.01，回归任务时一般设为0.001
    #momentum(float)：动量因子，一般设为0.9
    #dampening(float)：动量的抑制因子
    #nesterov(bool)：是否启用NAG加速
    #weight_decay(float)：权重衰减
    class torch.optim.SGD(params, lr=, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    ```

#### 自适应学习率算法

- ### Adam
  
    ```py
    #params：待优化参数的iterable，一般是model.parameters()
    #lr(float)：学习率
    #betas(Tuple[float, float])：用于计算梯度以及梯度平方的运行平均值的系数
    #eps(float)：为了增加数值计算的稳定性而加到分母里的项
    #weight_decay(float)：权重衰减
    class torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    ```



# 功能分类

## 网络

不仅Module是用户的主题网络，**LossFunc和ActivationFunc也可以视作一个网络**（本质是对torch.nn.functional中简单函数变为前向传播网络的**封装**），==一个网络就是一个前向传播的过程==，相当于一个函数，所以最终功能的实现，是通过这一层层模块化的网络的叠加，相当于复合函数。

- ### nn.Module

- ### Layers层

- ### Activation Function

- ### Loss Function

## Tensor

网络中的**递质**，储存了各种信息的一个矩阵（？）（包括临时累加的梯度，被操作过的计算符），用户重点需要考虑其形状（跟MindSpore有差别感觉）

### nn中封装对Shape的要求

- ### Layers

- ### Activation Function

- ### Loss Function


### Tensor的转换

#### 分类问题

- one_hot
- argmax

## 反向传播

# Pandas

## 数据加载&处理

### CSV

- 读取

     ```py
     # 读取 CSV 文件
     df = pd.read_csv('data.csv')
     ```

     

- 访问

     ```py
     #按列访问，索引为纵向对齐的首行
     age_column = df['Age']
     
     #按列访问，索引为行数
     first_col = df.columns[0]
     
     #按行访问，索引为行数
     first_row = df.loc[0]
     
     
     
     #按坐标访问，行，列
     df.iloc[0:2, 0]
     
     
     #查看所有值
     print(df.values)
     
     #查看所有索引
     print(df.index)
     ```

     

- 写入

- 删除

     ```py
     #删除列
     del df['Age']
     
     # 删除第一列
     df = df.drop(df.columns[0], axis=1)
     
     # 删除索引为 1 的行
     df = df.drop(1, axis=0)
     ```

     

- 操作

     ```py
     
     
     # 查看前 5 行数据
     print(df.head())
     
     # 查看后 5 行数据
     print(df.tail())
     
     # 查看数据的基本信息
     print(df.info())
     
     # 查看数据的统计摘要
     print(df.describe())
     
     # 查看缺失值的数量
     print(df.isnull().sum())
     
     # 填充缺失值
     df.fillna(0, inplace=True)
     
     # 删除重复行
     df.drop_duplicates(inplace=True)
     ```

     



