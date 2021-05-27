import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


# gpu-used
CUDA_DEVICES = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def test():
    Inputs = torch.randn(64, 24, 24)
    T = []
    max_seq_len = 0
    for i in range(len(Inputs)):
        max_seq_len = max(max_seq_len, len(Inputs[i][:, 0]))
        T.append(len(Inputs[i][:, 0]))
    model = PositionalEncoding(
        d_model=24,
        dropout=0.1,
        max_len=max_seq_len
    )
    model.train()
    outputs = model(Inputs)
    print("[Self attention utils.py] model: {}".format(model))
    print("[Self attention utils.py] inputs: {}".format(Inputs.shape))
    print("[Self attention utils.py] outputs: {}".format(outputs.shape))


if __name__ == '__main__':
    test()
