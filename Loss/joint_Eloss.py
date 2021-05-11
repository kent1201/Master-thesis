import math
import torch
import torch.nn as nn
from Loss.soft_dtw import SoftDTW


class JointEloss(nn.Module):

    def __init__(self, dis_func):
        super(JointEloss, self).__init__()
        self.dis_func = dis_func
        self.MSELoss = nn.MSELoss()
        self.soft_dtw = SoftDTW(gamma=0.001, normalize=True)

    def forward(self, X_tilde, X, H, H_hat_supervise):

        loss_T0, G_loss_S = 0.0, 0.0

        if self.dis_func == "MSE":
            loss_T0 = self.MSELoss(X_tilde, X)
            G_loss_S = torch.mul(self.MSELoss(H_hat_supervise, H), 0.1)
        elif self.dis_func == "Soft_DTW":
            loss_T0 = self.MSELoss(X_tilde, X).mean()
            G_loss_S = torch.mul(torch.add(self.soft_dtw(
                H_hat_supervise, H), 1e-7).mean(), 0.1)
        loss_0 = torch.mul(torch.sqrt(torch.add(loss_T0, 1e-7)), 10)
        loss = torch.add(loss_0, G_loss_S)
        loss = torch.add(loss, 1e-7)

        return loss, loss_T0


if __name__ == '__main__':

    X_tilde = torch.randn(32, 82, 34)
    X = torch.randn(32, 82, 34)
    H = torch.randn(32, 82, 24)
    H_hat_supervise = torch.randn(32, 82, 24)

    criterion = JointEloss()
    loss = criterion(X_tilde, X, H, H_hat_supervise)
    print(loss)
