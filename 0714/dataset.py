import os

import torch
import torch.nn.utils as utils
from torch.utils.data import Dataset

import pandas as pd

__all__=['TextDataset','Text2TextDataset','Vocab']




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


class Vocab:
    def __init__(self, text_sequence, vocab_dic: dict = None,vocab_min_frequency=8,vocab_max_frequency=None):
        if vocab_dic is None:
            self.vocab_dic = {
                "<PAD>": 0,
                "<BOS>": 1,
                "<EOS>": 2,
                "<NUL>": 3,
            }
        else:
            self.vocab_dic = vocab_dic

        vocab_fqcy = {

        }

        for text in text_sequence:
            # 分词
            tokens = self.tokenization(text)

            for token in tokens:



                # 原词表中是否已经存在
                if token not in self.vocab_dic:
                    # 添加元素
                    self.vocab_dic[token] = len(self.vocab_dic)
                    #统计词频
                    vocab_fqcy[token] = 1
                else:
                    vocab_fqcy[token] += 1

        #词过滤
        for vocab in vocab_fqcy:
            if vocab_fqcy[vocab] < vocab_min_frequency:
                del self.vocab_dic[vocab]
            elif vocab_max_frequency is not None and vocab_fqcy[vocab] > vocab_max_frequency:
                del self.vocab_dic[vocab]

        #缩去空的序号
        self.vocab_dic = {k:index for index,k in enumerate(self.vocab_dic.keys())}

    def save(self,path):
        os.path.exists(path) and os.makedirs(path)
        file=open(path,"w")
        for vocab in self.vocab_dic:
            file.writelines(vocab+"\n")

        file.close()
    @staticmethod
    def load(path):
        vocab_dic={}
        file=open(path,"r")
        line = file.readline()
        while line:
            vocab_dic[line[0:-1]]=len(vocab_dic)
            line = file.readline()
        return Vocab([],vocab_dic = vocab_dic)


    def append(self,text_sequence):
        """
        新增词表序列
        :param text_sequence:
        :return:
        """
        return Vocab(text_sequence, vocab_dic=self.vocab_dic)

    def tokenization(self, text:str):
        content = text.replace(',', '').replace('.','').replace('!','')
        content = content.split(' ')

        filter = ['#', '-', '&', '=', '/']
        for token in content:
            remove = False
            for f in filter:
                if f in token:
                    remove = True
                    break
            if remove:
                content.remove(token)

        return content

    def str2index(self, text):
        tokens = self.tokenization(text)
        vocab_index_seq = []
        vocab_index_seq.append(self.vocab_dic["<BOS>"])
        for token in tokens:
            if token not in self.vocab_dic:
                vocab_index_seq.append(self.vocab_dic["<NUL>"])
            else:
                vocab_index_seq.append(self.vocab_dic[token])
        vocab_index_seq.append(self.vocab_dic["<EOS>"])
        return vocab_index_seq

    def index2str(self, index:list)->str:
        res = ""
        reversed_dic = dict(zip(self.vocab_dic.values(), self.vocab_dic.keys()))
        for i in index:
            res+=reversed_dic[i]+" "
        return res

class Text2TextDataset(Dataset):
    def __init__(self, data_path,origin_vocab:Vocab=None, cut_input=50,cut_output=12):
            self.cut_input = cut_input
            self.cut_output = cut_output
            self.input,self.output=Text2TextDataset.file2data(data_path)

            #读取存档
            if os.path.exists(os.path.splitext(data_path)[0]+".vcb"):
                self.vocab=Vocab.load(os.path.splitext(data_path)[0]+".vcb")
            else:
                if origin_vocab is None:
                    self.vocab = Vocab(self.input + self.output)
                    self.vocab.save(os.path.splitext(data_path)[0]+".vcb")
                else:
                    self.vocab = origin_vocab
                    self.vocab.save(os.path.splitext(data_path)[0] + ".vcb")


    def __getitem__(self, item):
        return self.vocab.str2index(self.input[item]), self.vocab.str2index(self.output[item])

    def __len__(self):
        return len(self.input)

    @staticmethod
    def file2data(path):
        """
        将各种文件转变为输入输出两个个列表的策略，\n
        cvs:input,output
        :param path: 文件的路径
        :return:
        """
        _, extension = os.path.splitext(path)
        if extension == ".csv":
            df = pd.read_csv(path, encoding="utf-8")
            df = df.drop(df.columns[0],axis=1)
            value = df.values.tolist()
            outp,inp=zip(*value)
            return list(inp),list(outp)

    def collate_fn(self, batch):
        input_origin,output_origin=zip(*batch)

        input_b=list(input_origin)
        output_b=list(output_origin)
        #截断
        for i in range(len(batch)):
            if len(input_origin[i])>self.cut_input:
                input_b[i] = input_origin[i][:self.cut_input]
            if len(output_origin[i])>self.cut_output:
                output_b[i] = output_origin[i][:self.cut_output]

        #填充
        return utils.rnn.pad_sequence([torch.tensor(t) for t in input_b],padding_value=0, batch_first=True),utils.rnn.pad_sequence([torch.tensor(t) for t in output_b],padding_value=0, batch_first=True)