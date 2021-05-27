import torch.nn as nn
import torch
import torch.nn.functional as F
# for others
from Network.tcn import TemporalConvNet
from Network.Self_Attention.layers import EncoderLayer
from Network.Self_Attention.utils import PositionalEncoding
from Network.Self_Attention.sublayers import MultiHeadAttention
# for inner testing
# from tcn import TemporalConvNet
# from Self_Attention.layers import EncoderLayer
# from Self_Attention.utils import PositionalEncoding
# from Self_Attention.sublayers import MultiHeadAttention


class Discriminator(nn.Module):
    def __init__(self, module='gru', time_stamp=82, input_size=24, hidden_dim=17,
    output_dim=1, num_layers=10, activate_function=None, padding_value=-999.0, max_seq_len = 100):
        super(Discriminator, self).__init__()

        self.module = module
        self.input_size = input_size
        self.time_stamp = time_stamp
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.hidden_dim_layers = []
        self.output_dim = output_dim
        self.padding_value = padding_value
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(p=0.1)

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
        elif self.module == 'self-attn':
            self.position = PositionalEncoding(self.hidden_dim, dropout=0.1, max_len=self.max_seq_len)
            # self.r_cell = nn.ModuleList([
            #     EncoderLayer(
            #         d_model=self.hidden_dim,
            #         d_inner=self.hidden_dim,
            #         n_head=(self.hidden_dim // 3),
            #         d_k=(self.hidden_dim // (self.hidden_dim // 3)),
            #         d_v=(self.hidden_dim // (self.hidden_dim // 3))
            #     ) for _ in range(self.num_layers)])
            self.r_cell = MultiHeadAttention(
                n_head=(self.hidden_dim // 3),
                d_model=self.hidden_dim,
                d_k=(self.hidden_dim // (self.hidden_dim // 3)),
                d_v=(self.hidden_dim // (self.hidden_dim // 3))
            )

        self.activate = activate_function

        self.fc1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self.input_size, self.hidden_dim)),
            nn.LayerNorm([self.time_stamp, self.hidden_dim]),
            self.dropout,
            self.activate
        )
        self.fc2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim)),
            self.activate,
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        # self.fc1_1 = nn.Sequential(
        #     nn.utils.spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim)),
        #     nn.LayerNorm([self.hidden_dim, self.hidden_dim]),
        #     self.activate
        # )
        # self.fc1_2 = nn.Sequential(
        #     nn.utils.spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim)),
        #     nn.LayerNorm([self.hidden_dim, self.hidden_dim]),
        #     self.activate
        # )
        # self.fc1_3 = nn.Sequential(
        #     nn.utils.spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim)),
        #     nn.LayerNorm([self.hidden_dim, self.hidden_dim]),
        #     self.activate
        # )

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
        elif self.module == 'self-attn':
            fc1_out = self.position(fc1_out)
            fc1_out = self.fc1_1(fc1_out)
            fc1_out = self.fc1_2(fc1_out)
            enc_output = torch.transpose(fc1_out, 0, 1)
            ## transformer encoder layer
            # for enc_layer in self.r_cell:
            #     enc_output, enc_slf_attn = enc_layer(enc_output)
            enc_output, enc_slf_attn = self.r_cell(enc_output, enc_output, enc_output)
            output = torch.transpose(enc_output, 0, 1)
            output = self.fc1_3(output)
        elif self.module == 'gru':        
            X_packed = torch.nn.utils.rnn.pack_padded_sequence(
                input=fc1_out, 
                lengths=T.cpu(), 
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
        
        Y = self.fc2(output).squeeze()
        return Y


# gpu-used
CUDA_DEVICES = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test():
    Inputs = torch.randn(64, 24, 24)
    T = []
    max_seq_len = 0
    for i in range(len(Inputs)):
        max_seq_len = max(max_seq_len, len(Inputs[i][:, 0]))
        T.append(len(Inputs[i][:, 0]))
    model = Discriminator(
        module='self-attn',
        time_stamp=24,
        input_size=24,
        hidden_dim=24,
        output_dim=1,
        num_layers=18,
        activate_function=nn.Sigmoid(),
        max_seq_len=max_seq_len
    )
    model.train()
    outputs = model(Inputs, T)
    print("[discriminator.py] model: {}".format(model))
    print("[discriminator.py] inputs: {}".format(Inputs.shape))
    print("[discriminator.py] outputs: {}".format(outputs.shape))


if __name__ == '__main__':
    test()
