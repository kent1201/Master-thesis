import math
import torch
import torch.nn as nn
# from Loss.soft_dtw import SoftDTW

class JointDloss(nn.Module):

  def __init__(self, mode):
    super(JointDloss, self).__init__()
    self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
    self.gamma = 1
    self.mode = mode
    self.relu = nn.ReLU()

  def forward(self, Y_real, Y_fake, Y_fake_e):

    loss_real, loss_fake, loss_fake_e = 0.0, 0.0, 0.0

    if self.mode == "default":
      loss_real = self.BCEWithLogitsLoss(Y_real, torch.ones_like(Y_real))
      loss_fake = self.BCEWithLogitsLoss(Y_fake, torch.zeros_like(Y_fake))
      loss_fake_e = self.BCEWithLogitsLoss(Y_fake_e, torch.zeros_like(Y_fake_e))
    elif self.mode == "wgan":
      loss_real = -(Y_real.mean())
      loss_fake = Y_fake.mean()
      loss_fake_e = Y_fake_e.mean()
    elif self.mode == "hinge":
      loss_real = torch.mean(self.relu(torch.sub(torch.ones_like(Y_real), Y_real)))
      loss_fake = torch.mean(self.relu(torch.add(torch.ones_like(Y_fake), Y_fake)))
      loss_fake_e = torch.mean(self.relu(torch.add(torch.ones_like(Y_fake_e), Y_fake_e)))

    loss = loss_real.add(loss_fake).add(torch.mul(loss_fake_e, self.gamma))


    return loss


if __name__ == '__main__':

  Y_real = torch.randn(32, 82, 1)
  Y_fake = torch.randn(32, 82, 1)
  Y_fake_e = torch.randn(32, 82, 1)

  criterion = JointDloss()
  loss = criterion(Y_real, Y_fake, Y_fake_e)
  print(loss)