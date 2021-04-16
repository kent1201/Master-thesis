import torch.nn as nn
import torch
import torch.nn.functional as F
# for others
from Network.tcn import TemporalConvNet
from Network.Self_Attention.layers import EncoderLayer
# for inner testing
# from tcn import TemporalConvNet
# from Self_Attention.layers import EncoderLayer
# from attention import BiLSTM_Attention


class Embedder(nn.Module):
    def __init__(self, module='gru', time_stamp=82, input_size=27, hidden_dim=17,
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

        if self.module == 'gru':
            self.r_cell = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True
            )
        elif self.module == 'lstm':
            self.r_cell = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True
            )
        elif self.module == 'bi-lstm':
            self.r_cell = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=True
            )

        elif self.module == 'tcn':
            for i in range(num_layers):
                self.hidden_dim_layers.append(self.hidden_dim)
            self.r_cell = TemporalConvNet(num_inputs=self.input_size,
                                          num_channels=self.hidden_dim_layers,
                                          kernel_size=2,
                                          dropout=0.2
                                          )

        elif self.module == 'self-attn':
            # d_k, d_v =  d_model / n_head
            d_model, d_inner = self.input_size, self.input_size
            n_head = d_model // 3
            d_k, d_v = d_model // n_head, d_model // n_head
            self.r_cell = nn.ModuleList([EncoderLayer(
                d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=0.1) for _ in range(6)])

        if self.module == 'bi-lstm':
            self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)
        elif self.module == 'self-attn':
            self.fc = nn.Linear(self.input_size, self.output_dim)
        else:
            self.fc = nn.Linear(self.hidden_dim, self.output_dim)

        self.activate = activate_function

        self.conv1 = nn.Conv2d(1, 1, kernel_size=(3, 1), stride=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(3, 1), stride=1)

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
            for name, param in self.fc.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, X, T):
        if self.module == 'tcn':
            # Input X shape: (batch_size, seq_len, input_dim)
            X = torch.transpose(X, 1, 2)
            # Input X shape: (batch_size, input_dim, seq_len)
            output = self.r_cell(X)
            # H shape: (batch_size, input_dim, seq_len)
            # H_transpose: (batch_size, seq_len, input_dim)
            output = torch.transpose(output, 1, 2)
        elif self.module == 'self-attn':
            X = torch.unsqueeze(X, 1)
            X = self.activate(self.conv2(self.conv1(X)))
            # print("X2: {}".format(X.shape))
            X = torch.squeeze(X, 1)
            # print("X3: {}".format(X.shape))
            # Input X shape: (batch_size, seq_len, input_dim)
            enc_output = torch.transpose(X, 0, 1)
            # Input X shape: (seq_len, batch_size, input_dim)
            for enc_layer in self.r_cell:
                enc_output, enc_slf_attn = enc_layer(enc_output)
            # Input X shape: (seq_len, batch_size, input_dim)
            output = torch.transpose(enc_output, 0, 1)
            # Input X shape: (batch_size, seq_len, input_dim)
        else:

            X_packed = torch.nn.utils.rnn.pack_padded_sequence(
                input=X,
                lengths=T,
                batch_first=True,
                enforce_sorted=False
            )
            H_o, H_t = self.r_cell(X_packed)
            # Pad RNN output back to sequence length
            output, T = torch.nn.utils.rnn.pad_packed_sequence(
                sequence=H_o,
                batch_first=True,
                padding_value=self.padding_value,
                total_length=self.max_seq_len
            )
            # Inpu t X shape: (seq_len, batch_size, input_dim)
            # output, _ = self.r_cell(X, T)
            # Outputs shape: (seq_len, batch_size, input_dim)

        output_H = self.activate(self.fc(output))

        return output_H


def test():

    model = Embedder(
        module='attn-bilstm',
        time_stamp=82,
        input_size=27,
        hidden_dim=100,
        output_dim=27,
        num_layers=5,
        activate_function=nn.Tanh()
    )

    model.train()

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    optimizer.zero_grad()

    labels = torch.randn(32, 78, 27)
    inputs = torch.randn(32, 82, 27)

    inputs = inputs.to(CUDA_DEVICES)
    labels = labels.to(CUDA_DEVICES)

    # for name, param in model.named_parameters():
    #   if param.requires_grad:
    #     print(name, param.data)

    outputs = model(inputs, None)
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
