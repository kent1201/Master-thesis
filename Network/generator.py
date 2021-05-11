import torch.nn as nn
import torch
import torch.nn.functional as F
# for others
from Network.tcn import TemporalConvNet
# from Network.Self_Attention.layers import EncoderLayer
# for inner testing
# from tcn import TemporalConvNet
# from Attention.layers import EncoderLayer


class Generator(nn.Module):
    def __init__(self, module='gru', time_stamp=82, input_size=27, hidden_dim=100, 
    output_dim=100, num_layers=10, activate_function=nn.Tanh(), padding_value=-999.0, max_seq_len = 100):
        super(Generator, self).__init__()

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

        H = self.fc2(output)

        return self.activate(H)


# gpu-used
CUDA_DEVICES = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test():
    Inputs = torch.randn(64, 24, 6)
    T = []
    max_seq_len = 0
    for i in range(len(Inputs)):
        max_seq_len = max(max_seq_len, len(Inputs[i][:, 0]))
        T.append(len(Inputs[i][:, 0]))
    model = Generator(
        module='tcn',
        time_stamp=24,
        input_size=6,
        hidden_dim=24,
        output_dim=24,
        num_layers=18,
        activate_function=nn.Sigmoid(),
        max_seq_len=max_seq_len
    )
    model.train()
    outputs = model(Inputs, T)
    print("[generator.py] model: {}".format(model))
    print("[generator.py] inputs: {}".format(Inputs.shape))
    print("[generator.py] outputs: {}".format(outputs.shape))


if __name__ == '__main__':
    test()
