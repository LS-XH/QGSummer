import math
import random

import torch
import torch.nn.utils as utils

from Leo.data.dataset import TextDataset

__all__=['TextLoader']


class TextLoader:
    def __init__(self, dataset:TextDataset,batch_size:int=1,shuffle:bool=True):

        """

        :param dataset: 数据集，会按照序列长度从小到大排序为self.iterator
        :param batch_size: 打包的长度，每个包会自动填充所有序列至相同长度
        :param shuffle: 是否打乱
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.iterator = []
        self.batch_iter = []
        self.i = -1

        #根据length排序
        self.iterator=sorted(zip(dataset.vocab_index_seqs,dataset.label_index),key=lambda x:len(x[0]))

        self.text,self.label=zip(*self.iterator)


        for i in range(len(self)):
            # batch为1时
            if batch_size ==1:
                self.batch_iter.append([torch.tensor([self.text[i]]),[torch.tensor(self.label[i])]])
            #遍历i~i+bs将每个sequence转为tensor，之后将本batch填充到相同长度
            elif i != len(self)-1:
                self.batch_iter.append([utils.rnn.pad_sequence(
                    [torch.tensor(t) for t in list(self.text[i * self.batch_size:(i + 1) * self.batch_size])],
                    batch_first=True, padding_value=0).long(),
                    torch.tensor(list(self.label[i * self.batch_size:(i + 1) * self.batch_size])).long()])
            #末位batch
            elif batch_size!=1:
                self.batch_iter.append([utils.rnn.pad_sequence(
                    [torch.tensor(t) for t in list(self.text[i * self.batch_size:-1])],
                    batch_first=True,padding_value=0).long(),
                    torch.tensor(list(self.label[i * self.batch_size:-1])).long()])

    def __next__(self):
        self.i+=1
        if self.i>=len(self):
            raise StopIteration

        #padding+return
        return self.batch_iter[self.i]

    def __iter__(self):
        # 打乱
        self.batch_iter = random.sample(self.batch_iter, len(self.batch_iter))

        self.i=-1
        return self

    def __len__(self):
        return math.ceil(len(self.iterator)/self.batch_size)

    def __getitem__(self, index):
        return torch.tensor(self.iterator[index:index+self.batch_size]).long()

