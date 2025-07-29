import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




class BottlenectX(nn.Module):
    def __init__(self):
        super(BottlenectX, self).__init__()
