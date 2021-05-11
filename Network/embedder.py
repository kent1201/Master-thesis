import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
# for others
from Network.tcn import TemporalConvNet
# from Network.Self_Attention.layers import EncoderLayer
# for inner testing
# from tcn import TemporalConvNet
# from Self_Attention.layers import EncoderLayer
# from attention import BiLSTM_Attention


def reparameterization(mu, logvar, output_dim):
    std = torch.exp(logvar / 2)
    sampled_z = torch.randn_like(std)
    z = sampled_z * std + mu
    return z


class Embedder(nn.Module):
    def __init__(self, module='gru', mode='data', time_stamp=82, input_size=27, hidden_dim=17,
                 output_dim=24, num_layers=10, activate_function=nn.Tanh(), padding_value=-999.0, max_seq_len=100):
        super(Embedder, self).__init__()
        self.module = module
        self.input_size = input_size
        self.time_stamp = time_stamp
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.hidden_dim_layers = []
        self.output_dim = output_dim
        self.padding_value = padding_value
        self.max_seq_len = max_seq_len
        self.mode = mode      

        if self.module == 'gru':
            self.r_cell = nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True
            )
        elif self.module == 'tcn':
            for i in range(num_layers):
                self.hidden_dim_layers.append(self.hidden_dim)
            self.r_cell = TemporalConvNet(
                num_inputs=self.hidden_dim, 
                num_channels=self.hidden_dim_layers, 
                kernel_size=4,
                dropout=0.2
            )


        self.activate = activate_function
        self.fc1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self.input_size, self.hidden_dim)),
            nn.LayerNorm([self.time_stamp, self.hidden_dim]),
            self.activate
        )
        self.fc2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self.hidden_dim, self.output_dim)),
        )
        self.logvar = nn.Linear(self.output_dim, self.output_dim)
        self.mu = nn.Linear(self.output_dim, self.output_dim)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.r_cell.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.fc1.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)
            for name, param in self.fc2.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, X, T):
        
        fc1_out = self.fc1(X)
        
        if self.module == 'tcn':
            fc1_out = torch.transpose(fc1_out, 1, 2)
            output = self.r_cell(fc1_out)
            output = torch.transpose(output, 1, 2)
        elif self.module == 'gru':
            X_packed = torch.nn.utils.rnn.pack_padded_sequence(
                input=fc1_out,
                lengths=T.cpu(),
                batch_first=True,
                enforce_sorted=True
            )
            H_o, H_t = self.r_cell(X_packed)
            # Pad RNN output back to sequence length
            output, T = torch.nn.utils.rnn.pad_packed_sequence(
                sequence=H_o,
                batch_first=True,
                padding_value=self.padding_value,
                total_length=self.max_seq_len
            )
        
        H = self.fc2(output)
        if self.mode == 'data':
            H = self.activate(H)
        elif self.mode == 'noise':
            H = self.leaky_relu(H)
            H = torch.clamp(H, -1, 1)
            mu = self.mu(H)
            logvar = self.logvar(H)
            H = reparameterization(mu, logvar, self.output_dim)

        return H


def test():

    labels = torch.randn(64, 24, 6)
    inputs = torch.randn(64, 24, 24)
    T = []
    max_seq_len = 0
    for i in range(len(inputs)):
        max_seq_len = max(max_seq_len, len(inputs[i][:, 0]))
        T.append(len(inputs[i][:, 0]))

    model = Embedder(
        module='tcn',
        mode='noise',
        time_stamp=24,
        input_size=24,
        hidden_dim=24,
        output_dim=6,
        num_layers=20,
        activate_function=nn.tanh(),
        max_seq_len = max_seq_len
    )

    model.train()

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    optimizer.zero_grad()

    outputs = model(inputs, T)
    print("[embedder.py] outputs shape: {}".format(outputs.shape))

    loss = criterion(outputs, labels)

    loss.backward()

    optimizer.step()

    # for name, param in model.named_parameters():
    #   if param.requires_grad:
    #     print(name, param.data)

    model.eval()
    print("[embedder.py] model: {}".format(model))


if __name__ == '__main__':
    test()
