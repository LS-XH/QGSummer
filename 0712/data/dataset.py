import os

import torch
import torch.nn.utils as utils
from torch.utils.data import Dataset




class TextDataset(Dataset):
    def __init__(self, data_path:str, label:int, vocab_dic:dict=None, update_vocab:bool=False):
        """

        :param data_path: 数据集路径，此路径应全为txt并且label相同，不同
        :param label: 本数据集的标签，不同的标签应用“append”合并
        :param vocab_dic: 本数据集的预制词表
        :param update_vocab: 遇见新词是，是选择添加还是使用<NUL>填充
        """
        self.update_vocab=update_vocab

        if vocab_dic is None:
            self.vocab_dic = {
                "<PAD>": 0,
                "<BOS>": 1,
                "<EOS>": 2,
                "<NUL>": 3,
            }
        else:
            self.vocab_dic = vocab_dic
        self.vocab_size = len(self.vocab_dic)

        self.vocab_index_seqs = []
        self.label_index = []

        self.a=1

        #遍历全部序列
        for train in os.listdir(data_path):
            vocab_index_seq=[]

            #一句话的位置
            path = os.path.join(data_path, train)
            #读取内容
            text=open(path, "r", encoding="utf-8").read()

            #BOS
            vocab_index_seq.append(1)

            if update_vocab:
                vocab_index_seq.extend(self.str2index(text))
            #非测试集不仅需要转换词，还需要检测并添加新词
            else:
                #分词
                tokens = self.tokenization(text)
                for token in tokens:
                    # 词表中是否加入新词
                    if token not in self.vocab_dic:
                        # 添加元素
                        self.vocab_dic[token] = len(self.vocab_dic)
                        # 转换词为序号
                        vocab_index_seq.append(self.vocab_dic[token])
                    else:
                        vocab_index_seq.append(self.vocab_dic[token])

            #EOS
            vocab_index_seq.append(2)

            self.vocab_index_seqs.append(vocab_index_seq)
            self.label_index.append(label)

    def tokenization(self,text):
        # 分词
        filters = ['!', '#', '"', ',', '.', ':', ';', '/', '\\', '\t', '\n']
        text = text.lower()
        # text=re.sub("<br />"," ",text)
        # text=re.sub("|".join(filters)," ",content)
        return text.split(" ")


    def str2index(self,text):
        tokens = self.tokenization(text)
        vocab_index_seq=[]
        for token in tokens:
            if token not in self.vocab_dic:
                vocab_index_seq.append(self.vocab_dic["<NUL>"])
            else:
                vocab_index_seq.append(self.vocab_dic[token])

        return vocab_index_seq

    def str2tensor(self,text):
        return torch.tensor([self.str2index(text)])


    def append(self,data_path,label):
        new_ele = TextDataset(data_path, label, vocab_dic=self.vocab_dic, update_vocab=self.update_vocab)

        #统一词表
        self.vocab_dic=new_ele.vocab_dic
        #更新词表长度
        self.vocab_size=new_ele.vocab_size

        #合并序列
        self.vocab_index_seqs.extend(new_ele.vocab_index_seqs)
        #合并标签
        self.label_index.extend(new_ele.label_index)



    def __len__(self):
        return len(self.vocab_index_seqs)

    def __getitem__(self, index):
        return torch.tensor(self.vocab_index_seqs[index]).long(), torch.tensor(self.label_index[index]).float()

    @staticmethod
    def collate_fn(batch):
        text,label=zip(*batch)
        #填充
        text = utils.rnn.pad_sequence(text, batch_first=True)
        return text, torch.stack(label)

