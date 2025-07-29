import torch
import torch.nn as nn
import math
from Leo.data.dataset import *
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from Leo.data import *

__all__=['TransformerNet']

class TransformerNet(nn.Module):
    def __init__(self,vocab:Vocab,vector_dim:int=50):
        super(TransformerNet, self).__init__()
        self.vocab = vocab

        self.embedding = nn.Embedding(num_embeddings=len(vocab.vocab_dic), embedding_dim=vector_dim)
        self.pe= PositionalEncoding(vector_dim)
        self.transformer = nn.Transformer(batch_first=True,d_model=vector_dim,nhead=10,num_encoder_layers=64,num_decoder_layers=64)

        self.fc = nn.Linear(in_features=vector_dim, out_features=len(vocab.vocab_dic))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, src,tgt):
        tgt_seq_len=0
        if tgt.dim()==2:
            tgt_seq_len = tgt.size(1)
        else:
            tgt_seq_len=tgt.size(0)

        src = self.embedding(src)
        src = self.pe(src)

        tgt = self.embedding(tgt)
        tgt = self.pe(tgt)

        out = self.transformer(src, tgt,tgt_mask=torch.triu(torch.ones(tgt_seq_len, tgt_seq_len).to(src.device) == 1).transpose(0, 1))
        out = self.fc(out)
        out = self.softmax(out)
        return out

    def predict(self,src:torch.Tensor):
        res=[]
        index=self.vocab.vocab_dic['<BOS>']
        odd_index = None
        while index!=self.vocab.vocab_dic['<EOS>'] and len(res)!=128:
            out = self(src,torch.tensor([index]).to(src.device))
            index = out.argmax(dim=-1)[0]
            if  odd_index is not None and odd_index == index:
                _,index = torch.topk(out, k=2)
                index = index[0][1]
            res.append(index)
            odd_index=index
        return torch.stack(res)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置用cos
        self.register_buffer('pe', pe)  # 不参与训练

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            x: (seq_len, d_model)
        Returns:
            添加位置编码后的张量
        """
        if x.dim() == 3:
            x = x + self.pe[:x.shape[1]]  # 自动广播到batch维度
            return x
        else:
            return x + self.pe[:x.shape[0]]