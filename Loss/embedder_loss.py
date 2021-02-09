import math
import torch
import torch.nn as nn
from Loss.soft_dtw import SoftDTW

class EmbedderLoss(nn.Module):

  def __init__(self, dis_func):
    super(EmbedderLoss, self).__init__()
    self.dis_func = dis_func
    self.MSELoss = nn.MSELoss()
    self.sdtw_cuda = SoftDTW(gamma=1.0, normalize=True)

  def forward(self, outputs, targets):

    loss_only = 0
    if self.dis_func == "MSE":
      loss_only = self.MSELoss(outputs, targets)
    elif self.dis_func == "Soft_DTW":
      loss_only = self.sdtw_cuda(outputs, targets).mean()

    loss = torch.mul(10.0, torch.sqrt(torch.add(loss_only, 1e-10)))

    return loss_only, loss


if __name__ == '__main__':

  outputs = torch.randn(32, 82, 24)
  targets = torch.randn(32, 82, 24)

  criterion = EmbedderLoss()
  loss_only, loss = criterion(outputs, targets)
  print(loss)




