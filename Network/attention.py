import torch
from torch.autograd import Variable
import torch.nn as nn
# import numpy as np
import configparser


config = configparser.ConfigParser()
config.read('../Configure.ini', encoding="utf-8")

CUDA_DEVICES = torch.device("cuda:"+config.get('default', 'cuda_device_number') if torch.cuda.is_available() else "cpu")

class BiLSTM_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiLSTM_Attention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, bidirectional=True)
        self.out = nn.Linear(self.hidden_dim * 2, self.output_dim)

    # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.hidden_dim * 2, 1)   # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, self.hidden_dim).squeeze(2) # attn_weights : [batch_size, n_step]
        soft_attn_weights = nn.Softmax(attn_weights, 1)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy() # context : [batch_size, n_hidden * num_directions(=2)]

    def forward(self, X, H):
        # X : [batch_size, len_seq, embedding_dim]
        input = X.permute(1, 0, 2) # input : [len_seq, batch_size, embedding_dim]

        hidden_state = Variable(torch.zeros(1*2, len(X), self.hidden_dim)).to(CUDA_DEVICES) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = Variable(torch.zeros(1*2, len(X), self.hidden_dim)).to(CUDA_DEVICES) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2) # output : [batch_size, len_seq, n_hidden]
        attn_output, attention = self.attention_net(output, final_hidden_state)
        return self.out(attn_output), attention # model : [batch_size, num_classes], attention : [batch_size, n_step]




# def batch_matmul(seq, weight, nonlinearity=''):
#     s = None
#     for i in range(seq.size(0)):
#         _s = torch.mm(seq[i], weight)
#         if(nonlinearity=='tanh'):
#             _s = torch.tanh(_s)
#         _s = _s.unsqueeze(0)
#         if(s is None):
#             s = _s
#         else:
#             s = torch.cat((s,_s),0)
#     return s.squeeze()

# class AttentionLayer(nn.Module):
#     """Implements an Attention Layer"""

#     def __init__(self, cuda, nhid):
#         super(AttentionLayer, self).__init__()
#         self.nhid = nhid
#         self.weight_W = nn.Parameter(torch.Tensor(nhid,nhid))
#         self.weight_proj = nn.Parameter(torch.Tensor(nhid, 1))
#         self.softmax = nn.Softmax()
#         self.weight_W.data.uniform_(-0.1, 0.1)
#         self.weight_proj.data.uniform_(-0.1,0.1)
#         self.cuda = cuda

#     def forward(self, inputs, attention_width=3):
#         results = None
#         for i in range(inputs.size(0)):
#             if(i<attention_width):
#                 output = inputs[i]
#                 output = output.unsqueeze(0)

#             else:
#                 lb = i - attention_width
#                 if(lb<0):
#                     lb = 0
#                 selector = torch.from_numpy(np.array(np.arange(lb, i)))
#                 if self.cuda:
#                     selector = Variable(selector).to(self.cuda)
#                 else:
#                     selector = Variable(selector)
#                 vec = torch.index_select(inputs, 0, selector)
#                 u = batch_matmul(vec, self.weight_W, nonlinearity='tanh')
#                 a = batch_matmul(u, self.weight_proj)
#                 a = self.softmax(a)
#                 output = None
#                 for i in range(vec.size(0)):
#                     h_i = vec[i]
#                     a_i = a[i].unsqueeze(1).expand_as(h_i)
#                     h_i = a_i * h_i
#                     h_i = h_i.unsqueeze(0)
#                     if(output is None):
#                         output = h_i
#                     else:
#                         output = torch.cat((output,h_i),0)
#                 output = torch.sum(output,0)
#                 output = output.unsqueeze(0)

#             if(results is None):
#                 results = output

#             else:
#                 results = torch.cat((results,output),0)

#         return results