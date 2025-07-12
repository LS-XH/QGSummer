import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np
import matplotlib.pyplot as plt


def generate_binary_parity_data(num_samples:int, seq_len:int):
    """
    生成二进制奇偶判断数据集。

    参数:
        num_samples: 要生成的样本数量
        seq_len: 每个二进制序列的长度

    返回:
        X: 形状为 (num_samples, seq_len) 的 numpy 数组，包含0和1
        y: 形状为 (num_samples,) 的 numpy 数组，包含0(偶数)或1(奇数)
    """
    # 随机生成二进制序列 (0 和 1)
    X = np.random.randint(0, 2, size=(num_samples, seq_len))

    # 计算每个序列中1的个数
    sum_ones = np.sum(X, axis=1)

    # 判断1的个数是奇数(1)还是偶数(0): 使用模2运算 (sum_ones % 2)
    y = sum_ones % 2

    return torch.tensor(X,dtype=torch.float32).reshape(num_samples,seq_len,1), torch.tensor(y,dtype=torch.long).reshape(num_samples)

class SimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=2, embedding_dim=10)
        self.rnn = nn.RNN(1, 100, 4, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(100, 10)
        #relu
        self.fc2 = nn.Linear(10, 2)
        #relu

    def forward(self, x):
        # x形状: [batch_size, seq_length]
        # RNN处理
        output, hidden = self.rnn(x)

        # 使用最后一个时间步的输出
        last_output = output[:, -1, :]  # [batch_size, hidden_dim]

        # 全连接层
        return F.relu(self.fc2(F.relu(self.fc(self.dropout(last_output)))))

class LSTM(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=200, padding_idx=0)
        self.lstm = nn.LSTM(input_size= 200, hidden_size=64, num_layers=2,bidirectional=False, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        embedding = self.embedding(x)
        output, (h_n,c_n) = self.lstm(embedding)
        #last_output = output[:, -1, :]
        last_output=h_n[-1,:]
        out=self.classifier(last_output)
        return out

class SentimentAnalyzer(nn.Module):
    def __init__(self,vocab_size, hidden_dim=200, layers=2, dropout=0.5, bidirectional=True):
        super().__init__()

        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=200, padding_idx=0)
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim, num_layers=layers,
                            dropout=dropout, batch_first=True,
                            bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(2 * hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Get the word embeddings of the batch
        embedded = self.embedding(x)
        # Propagate the input through LSTM layer/s
        _, (hidden, _) = self.lstm(embedded)

        # Extract output of the last time step
        # Extract forward and backward hidden states of the
        # last time step
        out = torch.cat([hidden[-2, :, :], hidden[-1, :, :]], dim=1)

        out = self.dropout(out)
        out = self.fc1(out)
        out = self.sigmoid(out)

        return out