# 文件架构

- ## mindspore

     - ## ==mindspore.nn==

          - ### *cell.py

          - ### *mindspore.nn.loss

               - #### *loss.py

                    - MSELoss

                    - CrossEntropyLoss
     
                    - BCELoss
     
                    - L1Loss
     
                    - NLLLoss
     
          - ### *mindspore.nn.optim
     
               - SGD
     
               - Adam
     
          - ### *mindspore.nn.layer
     
               - #### *activation.py
     
                    - ReLU
                    - LeakyReLU
                    - ELU
                    - Tanh
                    - Sigmoid
     
               - #### *basic.py
     
                    - Dense
                    - Flatten
                    - OneHot
                    - L1Regularizer
     
               - #### *conv.py
     
                    - Conv1d
                    - Conv2d
                    - Conv3d
     
          - ### *mindspore.nn.wrap
     
               - #### *cell_wrapper.py
     
                    - TrainOneStepCell
                    - WithLossCell
     
     - ## ==mindspore.dataset==
     
          - #### *mindspore.dataset.engine
     
               - #### *datasets.py
     
                    - Dataset
     
               - #### *datasets_user_defined.py
     
                    - GeneratorDataset
     
               - #### *datasets_audio.py
     
               - #### *datasets_text.py
     
               - #### *datasets_vision.py
               
          - #### ==mindspore.dataset.vision==
          
               - ##### *transforms.py
     
     
     - ## *mindspore.train
       
          - #### *serialization.py
               - save_checkpoint()
               
               - load_checkpoint()
               
               - load_param_into_net()
             
          - #### *mindspore.train.callback
          
               - #### *.py
               
                    - LossMonitor
                    - ModelCheckpoint
                    - TimeMonitor
                    - SummaryCollector
                    - LearningRateScheduler
             
          - #### *mindspore.train.train_thor
          
               - #### *model_thor.py
               
                    - Model
          
     - ## mindspore.common
     
          - #### initializer.py
     
               - Normal
               - Uniform
     
               - HeNormal
               - HeUniform
               - XavierNormal
               - XavierUniform
               - One
               - Zero
     
     
     - ## mindspore.ops   算子
     
          - ### mindspore.ops.function   神经网络函数



# 使用说明

## Tensor

太重要了，所以单独分一类

```py
#input_data(Union[Tensor, float, int, bool, tuple, list, numpy.ndarray])：被存储的数据，可以是其它Tensor，也可以是Python基本数据（如int，float，bool等），或是一个NumPy对象。
#dtype(mindspore.dtype)：用于定义该Tensor的数据类型，必须是 mindspore.dtype 中定义的类型。如果该参数为 None ，则数据类型与 input_data 一致
class mindspore.Tensor(input_data=None, dtype=None, shape=None, init=None, internal=False, const_arg=False)
```

### Tensor的函数

- #### \_\_getitem\_\_

     ```py
     img = Tensor(img)
     print(img[0][1][2])	#一定要加img[0]
     ```

- #### reshape

     重塑形状

     ```py
     img = Tensor(img).reshape(1, 784)
     ```

- #### float

     转为float32类型

     ```py
     img = Tensor(img).float()
     ```

     

     



### Tensor传递时的形状

| 层类型 | 输入形状 | 输出形状 |
| -------------- | -------------------------------- | -------------- |
| Dense（Linear） | (batch_size, in_features)        | (batch_size, out_features) |
| Conv1D | (batch_size, in_channels, sequence_length) | (batch_size, out_channels, new_sequence_length) |
| Conv2D | (batch_size, in_channels, height, width) | (batch_size, out_channels, new_height, new_width) |
| Conv3D | (batch_size, in_channels, depth, height, width) | (batch_size, out_channels, n_depth, n_height, n_width) |
| MaxPool1D/2D/3D | 与对应卷积层相同 | height，width，除以池化核 |
| RNN / LSTM / GRU | (batch_size, sequence_length, input_size) | (batch_size, sequence_length, hidden_size) |
| Transformer  | (batch_size, seq_len, embed_dim) | (batch_size, sequence_length, embedding_dim) |



## mindspore

### *mindspore.train

#### ==训练封装==（mindspore.train.train_thor）

- ### Model

     一个简便的用于训练模型的封装

     ```py
     #network(Cell)：用于训练或推理的神经网络。
     #loss_fn(Cell)：损失函数。
     #optimizer(Cell)：用于更新网络权重的优化器。
     #metrics(Union[dict, set])：用于模型评估的一组评价函数。例如：{‘accuracy’, ‘recall’}。
     
     class mindspore.Model(network, loss_fn=None, optimizer=None, metrics=None, eval_network=None, eval_indexes=None, amp_level='O0', boost_level='O0', **kwargs)
     ```

     - #### train()->

          用于训练模型，回调函数，详见[回调对象](# 训练时回调函数（mindspore.train.callback）)

          ```py
          #epoch(int)：训练步数
          #train_dataset(Dataset)：训练数据集
          #callbacks(Optional[list[Callback], Callback])：训练过程中回调对象
          '''
          	LossMonitor
          	ModelCheckpoint
          	TimeMonitor
          	SummaryCollector
          	LearningRateScheduler
          '''	
          #initial_epoch(int)：从哪个epoch开始训练，一般用于中断恢复训练场景。
          
          Model.train(epoch, train_dataset, callbacks=None, dataset_sink_mode=True, sink_size=- 1, initial_epoch=0)
          
          #example:
          
          data=dataset.GeneratorDataset(gener(), ["label","data"])	#使用方法生成数据集
          mynet = Net()												#实例化神经网络
          opti=nn.SGD(mynet.get_parameters(), learning_rate=0.001)	#初始化优化器
          lossfunc=nn.MSELoss()										#设置损失函数
          
          model = Model(mynet,lossfunc,opti,metrics={'accuracy'})		#设置model
          model.train(1000,data)										#训练model
          
          
          #ex1):*自定义回调对象
          
          
          ```

          



#### ==训练时回调函数==（mindspore.train.callback）

- ### Callback

     回调函数的基类

     ```py
     #ex1):
     model.train(100,MyData(),callbacks=[LossMonitor()])		#启用损失监测
     
     
     
     #用户自定义&实现
     class EarlyStopping(Callback):
         def __init__(self, patience=3, min_delta=0.01):
             super().__init__()
             self.patience = patience    # 允许连续验证损失不下降的轮次
             self.min_delta = min_delta  # 最小变化阈值
             self.best_loss = float('inf')
             self.counter = 0
             
         #epoch开始时调用
     	def epoch_begin(run_context)
         
         #epoch结束时调用
         def epoch_end(self, run_context):
             cb_params = run_context.original_args()
             current_loss = cb_params.net_outputs  # 假设网络输出为损失值
             if (self.best_loss - current_loss) > self.min_delta:
                 self.best_loss = current_loss
                 self.counter = 0
             else:
                 self.counter += 1
                 if self.counter >= self.patience:
                     run_context.request_stop()  # 停止训练
                     print(f"Early stopping triggered at epoch {cb_params.cur_epoch_num}")
                     
     ```
     
- ### LossMonitor

     损失监测

     ```py
     #per_print_times(int)：表示每隔多少个step打印一次loss。
     class mindspore.LossMonitor(per_print_times=1)
     ```

- ### ModelCheckpoint

     在训练过程中调用该方法可以保存网络参数

     ```py
     #prefix(str)：checkpoint文件的前缀名称。
     #directory(str)：保存checkpoint文件的文件夹路径。默认情况下，文件保存在当前目录下。
     class mindspore.ModelCheckpoint(prefix='CKP', directory=None, config=None)
     ```

- ### TimeMonitor

     监控训练或推理的时间

     ```py
     #data_size(int)：表示每隔多少个step打印一次信息。如果程序在训练期间获取到Model的 batch_num ，则将把 data_size 设为 batch_num ，否则将使用 data_size 。
     class mindspore.TimeMonitor(data_size=None)
     ```

- ### LearningRateScheduler

     用于在训练期间更改学习率

     ```py
     #learning_rate_function(Function)：在训练期间更改学习率的函数。
     class mindspore.LearningRateScheduler(learning_rate_function)
     
     #example：
     
     #lr：当前学习率
     #cur_step_num：当前步数
     def learning_rate_function(lr, cur_step_num):
         if cur_step_num%1000 == 0:
             lr = lr*0.1
         return lr
     ```

     

- ### EarlyStopping

     当监控的指标停止改进时停止训练。

#### ==模型检查点==

```py
#ex1)

mindspore.save_checkpoint(net,'test.ckpt')		#保存

mindspore.load_checkpoint('test.ckpt',net)		#加载


#ex2)

mindspore.save_checkpoint(net,'test.ckpt')		#保存

mindspore.load_param_into_net(net,mindspore.load_checkpoint('test.ckpt'))	#加载
```



- #### save_checkpoint()

     ```py
     #save_obj(Union[Cell, list, dict])：待保存的对象。数据类型可为 mindspore.nn.Cell 、list或dict。若为list，可以是 Cell.trainable_params() 的返回值，或元素为dict的列表（如[{"name": param_name, "data": param_data},…]，param_name 的类型必须是str，param_data 的类型必须是Parameter或者Tensor）；若为dict，可以是 mindspore.load_checkpoint() 的返回值。
     
     #ckpt_file_name(str)：checkpoint文件名称。如果文件已存在，将会覆盖原有文件。
     
     mindspore.save_checkpoint(save_obj, ckpt_file_name, integrated_save=True, async_save=False, append_dict=None, enc_key=None, enc_mode='AES-GCM', choice_func=None, crc_check=False, format='ckpt', **kwargs)
     ```

- #### load_checkpoint()

     ```py
     #ckpt_file_name(str)：checkpoint的文件名称。
     
     #net(Cell)：加载checkpoint参数的网络。
     
     #reutrn：字典，key是参数名称，value是Parameter类型。
     
     mindspore.load_checkpoint(ckpt_file_name, net=None, strict_load=False, filter_prefix=None, dec_key=None, dec_mode='AES-GCM', specify_prefix=None, choice_func=None, crc_check=False, remove_redundancy=False, format='ckpt')
     ```

     

- #### load_param_into_net()

     ```py
     #net (Cell) - 将要加载参数的网络。
     
     #parameter_dict (dict) - 加载checkpoint文件得到的字典。
     
     mindspore.load_param_into_net(net, parameter_dict, strict_load=False, remove_redundancy=False)
     ```

     



## mindspore.nn

### ==网络主体==（mindspore.nn）

- ### Cell

     一个神经网络的基本单位，用于继承

     ```py
     class Net(nn.Cell):
     	#override
         def __init__(self):
             super(Net, self).__init__()
         def construct(self, x):
             
             
             
     
     #example:
     #1) 构建前向传播的网络
     class Net(nn.Cell):
         def __init__(self):
             super(Net, self).__init__()
     
             self.lin1=nn.Linear(1, 10)
             self.lin2=nn.Linear(10, 10)
             self.lin3=nn.Linear(10, 1)
     
             self.relu = nn.ReLU()
     	#forward
         def construct(self, x):
     
             x=self.relu(self.lin1(x))
             x=self.relu(self.lin2(x))
             x=self.relu(self.lin3(x))
             return x
     ```

     - #### get_parameters()
     
          返回一个Parameter类型迭代器对象

### ==激活函数==（mindspore.nn.layer）

- ### Relu

### ==网络结构层==（mindspore.nn.layer）

- ### Dense

     ```py
     #in_channels(int)：Dense层输入Tensor的空间维度。
     #out_channels(int)：Dense层输出Tensor的空间维度。
     
     class mindspore.nn.Dense(in_channels, out_channels, weight_init=None, bias_init=None, has_bias=True, activation=None, dtype=mstype.float32)
     ```

     

- ### Conv2d

     ```py
     #in_channels(int)：Conv2d层输入Tensor的空间维度。
     
     #out_channels(int)：Conv2d层输出Tensor的空间维度。
     
     #kernel_size(Union[int, tuple[int]])：指定二维卷积核的高度和宽度。数据类型为整型或两个整型的tuple。一个整数表示卷积核的高度和宽度均为该值。两个整数的tuple分别表示卷积核的高度和宽度。
     
     #stride (Union[int, tuple[int]]，可选)：二维卷积核的移动步长。数据类型为整型或者长度为二或四的整型tuple。一个整数表示在高度和宽度方向的移动步长均为该值。两个整数的tuple分别表示在高度和宽度方向的移动步长。
     
     #weight_init(Union[Tensor, str, Initializer, numbers.Number]，可选)：权重参数的初始化方法。它可以是Tensor，str，Initializer或numbers.Number。当使用str时，可选 "TruncatedNormal" ， "Normal" ， "Uniform" ， "HeUniform" 和 "XavierUniform" 分布以及常量 "One" 和 "Zero" 分布的值，可接受别名 "xavier_uniform" ， "he_uniform" ， "ones" 和 "zeros" 。上述字符串大小写均可。更多细节请参考 Initializer, 的值。默认值： None ，权重使用 "HeUniform" 初始化。
     
     class mindspore.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, pad_mode='same', padding=0, dilation=1, group=1, has_bias=False, weight_init=None, bias_init=None, data_format='NCHW', dtype=mstype.float32)
     ```

     

### ==工具==（mindspore.nn.layer）

- ### OneHot（Cell）

     用于转换为独热编码

     ```py
     #axis(int)：指定第几阶为 depth 维one-hot向量，如果轴为-1，则 features * depth ，如果轴为0，则 depth * features 。默认值：-1。
     
     #depth(int)：定义one-hot向量的深度
     class mindspore.nn.OneHot(axis=- 1, depth=1, on_value=1.0, off_value=0.0, dtype=mstype.float32)
     
     #example：
     net = nn.OneHot(depth=4, axis=1)
     
     indices = Tensor([[1, 3], [0, 2]], dtype=mindspore.int32)
     
     output = net(indices)
     ```

     

### ==损失==（mindspore.nn.loss）

- ### LossBase（Cell）

     损失函数的基类

     ```py
     
     ```
     



- ### MESLoss

- ### CrossEntropyLoss

### ==优化器==（mindspore.nn.optim）

- ### SGD

### ==封装层==（mindspore.nn.wrap）

- #### WithLossCell

     输出损失的Cell

     封装 backbone 和 loss_fn 。

     loss_fn详见：[损失函数](损失（mindspore.nn.loss）)

     ```py
     #backbone(Cell)：要封装的骨干网络。
     #loss_fn(Cell)：用于计算损失函数。
     class mindspore.nn.WithLossCell(backbone, loss_fn)
     
     
     #example:
     
     net = Net()													#实例化网络对象
     loss_fn = nn.MSELoss()										#实例化损失函数对象
     
     loss_net = nn.WithLossCell(net, loss_fn)					#实例化loss_cell
     
     
     
     #封装实现
     class WithLossCell(nn.Cell):
         def __init__(self, backbone, loss_fn):
             super().__init__()
             self.backbone = backbone  # 用户定义的神经网络
             self.loss_fn = loss_fn     # 损失函数
     	
         #x(Tensor)：输入数据
         #label(Tensor)：输出标签
         def construct(self, x, label):
             output = self.backbone(x)          # 前向传播
             loss = self.loss_fn(output, label)  # 计算损失
             return loss
     ```

- #### TrainOneStepCell

     训练网络封装类。

     封装 network 和 optimizer 。

     ```py
     #network(Cell)：训练网络。即输出损失的Cell。
     #optimizer(Union[Cell])：用于更新网络参数的优化器。
     class mindspore.nn.TrainOneStepCell(network, optimizer, sens=1.0)
     
     
     #example
     
     data = dataset.GeneratorDataset(gener(),["data","label"])	#创建数据集
     net = Net()													#实例化网络对象
     loss_fn = nn.MSELoss()										#实例化损失函数对象
     optim = nn.SGD(net.get_parameters(),learning_rate=0.001)	#实例化优化器对象
     
     loss_net = nn.WithLossCell(net, loss_fn)					#创建损失Cell
     train_net = nn.TrainOneStepCell(loss_net, optim)			#创建训练Cell
     
     for i in range(100):
         for x,label in data:									#遍历数据集
             loss = train_net(x, label)							#训练一个step
             print(loss)
             
     
             
             
     #封装实现
     class TrainOneStepCell(nn.Cell):
         def __init__(self, network, optimizer):
             super().__init__()
             self.network = network  							#通常是 WithLossCell 实例
             self.optimizer = optimizer							#设置优化器
             self.grad = ops.GradOperation(get_by_list=True)  	#自动微分算子
             
     	#x(Tensor)：输入数据
         #label(Tensor)：输出标签
         def construct(self, x, label):
             loss = self.network(x, label)						#使用losscell前向计算损失
             # 反向传播计算梯度
             grads = self.grad(self.network, self.optimizer.parameters)(x, label)
             
             self.optimizer(grads)								#优化器更新参数
             return loss
     ```

     

## mindspore.common.initializer

### ==权重初始化==（mindspore.common.initializer）

- #### Initializer

     初始化器的抽象基类

     ```py
     #ex1)
     class Net(nn.Cell):
         def __init__(self):
             super(Net, self).__init__()
             self.lin1=nn.Dense(784, 256, HeUniform())
             self.relu = nn.ReLU()
     
         def construct(self, x):
             x = self.relu(self.lin1(x))
             return x
         
     #ex2)
     self.lin1=nn.Dense(784, 256, initializer(HeUniform(),[256,768]))
     
     
     #用户自定义&实现
     
     class Two(Initializer):
         #传入arr参数，为numpy.array
         def _initialize(self, arr):
             arr.fill(2)
     ```

     

- #### initializer（func）

     用Initializer创建并初始化一个Tensor

     ```py
     #init(Union[Tensor, str, Initializer, numbers.Number])：初始化方式。
     '''
     	 str - init 是继承自 Initializer 的类的别名，实际使用时会调用相应的类。 init 的值可以是 "normal" 、 "ones" 	   或 "zeros" 等。
          Initializer - init 是继承自 Initializer ，用于初始化Tensor的类。
          numbers.Number - 用于初始化Tensor的常量。
          Tensor - 用于初始化Tensor的Tensor。
     '''
     #shape(Union[tuple, list, int])：被初始化的Tensor的shape，注意，行对应输出神经元，列对应输入特征
     #dtype(mindspore.dtype)：被初始化的Tensor的数据类型
     
     mindspore.common.initializer.initializer(init, shape=None, dtype=mstype.float32)
     ```

- #### Uniform

     平均分布

     ```py
     #scale(float)：均匀分布的边界
     class mindspore.common.initializer.Uniform(scale=0.07)
     ```

- #### Normal

     正态分布

     ```py
     #sigma(float)：正态分布的标准差
     #mean(float)：正态分布的均值
     class mindspore.common.initializer.Normal(sigma=0.01, mean=0.0)
     ```

- #### HeUniform/HeNormal

     服从HeKaiming均匀/正态分布

     ```py
     #negative_slope(int, float)：本层激活函数的负数区间斜率（仅适用于非线性激活函数 'leaky_relu'）
     #mode(str)：可选 'fan_in' 或 'fan_out' ， 'fan_in' 会保留前向传递中权重方差的量级，'fan_out' 会保留反向传递的量级
     #nonlinearity(str)：非线性激活函数，推荐使用 'relu' 或 'leaky_relu'
     
     class mindspore.common.initializer.HeUniform(negative_slope=0, mode='fan_in', nonlinearity='leaky_relu')
     
     class mindspore.common.initializer.HeNormal(negative_slope=0, mode='fan_in', nonlinearity='leaky_relu')
     ```

     

- #### XavierUniform/XavierNormal

     服从Xarvier均匀/正态分布

     ```py
     #gain(float)：可选的缩放因子
     
     class mindspore.common.initializer.XavierUniform(gain=1)
     
     class mindspore.common.initializer.XavierNormal(gain=1)
     ```

     







## mindspore.dataset

### 基类&方法

- ### Dataset

     - #### map

          对数据集中的每个元素应用指定的操作

          ```py
          # 要应用的操作（单个函数或函数列表）
          
          dataset.map(operations,           
              input_columns=None,   # 输入列名（默认对所有列操作）
              output_columns=None,  # 输出列名（默认覆盖输入列）
              num_parallel_workers=1 # 并行线程数（加速处理）
          )
          
          ```


### 用户自定义数据集

- ### GeneratorDataset

     ```py
     #source：可以是可迭代或可随机访问的Python对象，要求有__next__()和__iter__()或__getitem__和__len__，或直接使用yield
     #
     class mindspore.dataset.GeneratorDataset(source, column_names=None, column_types=None, schema=None, num_samples=None, num_parallel_workers=1, shuffle=None, sampler=None, num_shards=None, shard_id=None, python_multiprocessing=True, max_rowsize=6)
     ```

     构建数据集，三种方法

     ```py
     #1)	用函数生成可迭代对象
     def gener():
         for i in range(5):
             yield np.array([i]).astype(np.float32),np.array([i*2+1]).astype(np.float32)
     
     datas=dataset.GeneratorDataset(gener(), ["label","data"])
     
     #2) 自定义迭代器类型
     class MyIter:
         #初始化
         def __init__(self):
             self.i=-1
         
         #每次使用迭代器的起点
         def __iter__(self):
             self.i = -1
             return self
         
         #next
         def __next__(self):
             self.i+=1
             if self.i>5:
                 raise StopIteration
             else:
                 return np.array([self.i]).astype(np.float32),np.array([self.i*2+1]).astype(np.float32)
         
         #长度
         def __len__(self):
             return 6
         
     dataset = ds.GeneratorDataset(source=MyIter(), column_names=["data", "label"])
     
     #3) 自定义随机可访问数据
     class MyAccessible:
         #初始化
         def __init__(self):
             #数据（也就是输入，即函数的x）
             self._data = np.random.sample((5, 2))
     		#标签（也就是输出，即函数的y） 
             self._label = np.random.sample((5, 1))
     
     	#获取元素
         def __getitem__(self, index):
             return self._data[index], self._label[index]
     
     	#长度
         def __len__(self):
             return len(self._data)
     	
     dataset = ds.GeneratorDataset(source=MyAccessible(), column_names=["data", "label"])
     ```
     
     

## mindspore.dataset.vision

### 转换（mindspore.dataset.vision.transforms）

- #### HWC2CHW





# 功能

## 环境安装

```cmd
```



## CUDA







