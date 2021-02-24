import math
import torch
import torch.nn as nn
from Loss.soft_dtw import SoftDTW

class JointGloss(nn.Module):

  def __init__(self, dis_func, mode):
    super(JointGloss, self).__init__()
    self.dis_func = dis_func
    # Adversarial loss
    # equivalent as sigmoid_cross_entropy_with_logits
    self.G_loss_U = nn.BCEWithLogitsLoss()
    # self.G_loss_U = nn.BCELoss()
    self.gamma = 1
    self.mode=mode
    # Supervised loss
    if self.dis_func == "MSE":
      self.G_loss_S = nn.MSELoss()
    elif self.dis_func == "Soft_DTW":
      self.G_loss_S = SoftDTW(gamma=1.0, normalize=True)
    # Two Momments (計算合成 data 與原始 data 的 loss)


  def forward(self, Y_fake, Y_fake_e, H, H_hat_supervise, X, X_hat):

    """
      Y_fake, Y_fake_e: [batch_size, seq_len, 1]
      H, H_hat_supervise: [batch_size, seq_len-1, n_features(hidden)]
      X, X_hat: [batch_size, seq_len, n_features]
    """
    loss_U, loss_U_e = 0.0, 0.0

    loss_V1 = torch.mean(torch.abs(torch.sub(torch.sqrt(torch.add(torch.var(X_hat, dim=0, keepdim=True, unbiased=True), 1e-7)), torch.sqrt(torch.add(torch.var(X, dim=0, keepdim=True, unbiased=True), 1e-7)))))
    loss_V2 = torch.mean(torch.abs(torch.sub(torch.mean(X_hat, dim=0, keepdim=True), torch.mean(X, dim=0, keepdim=True))))
    loss_V = loss_V1.add(loss_V2)

    if self.mode == "default":
      loss_U = self.G_loss_U(Y_fake, torch.ones_like(Y_fake))
      loss_U_e = self.G_loss_U(Y_fake_e, torch.ones_like(Y_fake_e))
    elif self.mode == "wgan" or self.mode == "hinge":
      loss_U = -(Y_fake.mean())
      loss_U_e = -(Y_fake_e.mean())

    loss_U = torch.add(loss_U.add(torch.mul(self.gamma, loss_U_e)), 1e-5)
    loss_S = torch.mul(torch.sqrt(torch.add(self.G_loss_S(H_hat_supervise, H).mean(), 1e-7)), 100)
    loss = loss_U.add(loss_S).add(torch.mul(loss_V, 100))
    loss = torch.add(loss, 1e-7)

    return loss, loss_U, loss_S, loss_V


if __name__ == '__main__':

  Y_fake = torch.randn(32, 82, 1)
  Y_fake_e = torch.randn(32, 82, 1)
  H = torch.randn(32, 81, 24)
  H_hat_supervise = torch.randn(32, 81, 24)
  X = torch.randn(32, 82, 34)
  X_hat = torch.randn(32, 82, 34)

  criterion = JointGloss()
  loss = criterion(Y_fake, Y_fake_e, H, H_hat_supervise, X, X_hat)
  print(loss)
