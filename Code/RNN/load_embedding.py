import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from Leo.data.dataset import Vocab







class LoadEmbedding(nn.Module):
    def __init__(self,data_path,vocab:Vocab=None):
        super(LoadEmbedding, self).__init__()

        if vocab is None:
            file = open(data_path,'r',encoding='utf-8')

            line = file.readline()[0:-1].split(' ')

            words = []
            vecs = []
            while line is not None:
                words.append(line[0])
                vecs.append([float(w) for w in line[1:]])

            self.embedding = nn.Embedding(100, 100)

            self.embedding.weight.data.copy_(torch.from_numpy(np.random.rand(100, 100)))
        else:


