import math
import torch
import torch.nn as nn
from Loss.soft_dtw import SoftDTW


class SupervisedLoss(nn.Module):

    def __init__(self, dis_func):
        super(SupervisedLoss, self).__init__()
        self.dis_func = dis_func
        self.MSELoss = nn.MSELoss()
        self.sdtw_cuda = SoftDTW(gamma=0.001, normalize=True)

    def forward(self, outputs, targets):

        loss = 0.0
        if self.dis_func == "MSE":
            E_loss_T0 = self.MSELoss(outputs, targets)
        elif self.dis_func == "Soft_DTW":
            E_loss_T0 = self.sdtw_cuda(outputs, targets).mean()

        E_loss0 = torch.mul(10.0, torch.sqrt(torch.add(E_loss_T0, 1e-10)))

        return E_loss_T0, E_loss0


if __name__ == '__main__':

    outputs = torch.randn(32, 82, 24)
    targets = torch.randn(32, 82, 24)

    criterion = SupervisedLoss()
    loss = criterion(outputs, targets)
    print(loss)
