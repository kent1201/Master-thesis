import torch.nn as nn
import torch
import torch.nn.functional as F

class Simple_Discriminator(nn.Module):
  def __init__(self, time_stamp=82, input_size=27, hidden_dim=120, output_dim=1,
   num_layers=1, padding_value=-999.0, max_seq_len = 100):
    super(Simple_Discriminator, self).__init__()
    self.input_size = input_size
    self.time_stamp = time_stamp
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.output_dim = output_dim
    self.r_cell = nn.GRU(
        input_size = self.input_size,
        hidden_size = self.hidden_dim,
        num_layers = self.num_layers,
        batch_first = True
    )
    self.activate = nn.Sigmoid()
    self.fc = nn.Linear(self.hidden_dim, self.output_dim)
    self.padding_value = padding_value
    self.max_seq_len = max_seq_len

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
    # Input X shape: (seq_len, batch_size, input_dim)
    # outputs, _ = self.r_cell(X, T)
    X_packed = torch.nn.utils.rnn.pack_padded_sequence(
                input=X, 
                lengths=T, 
                batch_first=True, 
                enforce_sorted=False
    )
    H_o, H_t = self.r_cell(X_packed)
    # Pad RNN output back to sequence length
    temp_outputs, T = torch.nn.utils.rnn.pad_packed_sequence(
        sequence=H_o, 
        batch_first=True,
        padding_value=self.padding_value,
        total_length=self.max_seq_len
    )
    outputs = self.activate(self.fc(temp_outputs))
    # Outputs shape: (seq_len, batch_size, input_dim)
    return outputs


# gpu-used
CUDA_DEVICES = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test():
  model = Simple_Discriminator(
    time_stamp = 81,
    input_size = 33,
    hidden_dim = 17,
    output_dim = 1,
    num_layers = 2
  )
  model = model.to(CUDA_DEVICES)
  model.train()
  inputs = torch.randn(32, 82, 33)
  inputs = inputs.to(CUDA_DEVICES)
  outputs = model(inputs, None)
  print("[discriminate.py main] model: {}".format(model))
  print("[discriminate.py main] inputs: {}".format(inputs.shape))
  print("[discriminate.py main] outputs: {}".format(outputs[0].data))


if __name__ == '__main__':
  test()







