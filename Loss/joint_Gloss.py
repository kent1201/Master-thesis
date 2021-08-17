import math
import torch
import torch.nn as nn
from Loss.soft_dtw import SoftDTW


class JointGloss(nn.Module):

    def __init__(self, uloss_func):
        super(JointGloss, self).__init__()
        self.uloss_func = uloss_func
        # Adversarial loss
        # equivalent as sigmoid_cross_entropy_with_logits
        # self.G_loss_U = nn.BCELoss()
        self.gamma = 1

    def forward(self, Y_fake, Y_fake_e):
        """
          Y_fake, Y_fake_e: [batch_size, seq_len, 1]
          H, H_hat_supervise: [batch_size, seq_len-1, n_features(hidden)]
          X, X_hat: [batch_size, seq_len, n_features]
        """
        if self.uloss_func == 'wgan' or self.uloss_func == 'hinge':
            lossG = torch.add(-torch.mean(Y_fake), torch.mul(0.1, -torch.mean(Y_fake_e)))
        else: 
            loss_g = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake, torch.ones_like(Y_fake)) 
            loss_g_e = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e, torch.ones_like(Y_fake_e)) 
            lossG = loss_g + 0.1 * loss_g_e

        return lossG


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
