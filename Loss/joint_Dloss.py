import math
import torch
import torch.nn as nn
# from Loss.soft_dtw import SoftDTW


class JointDloss(nn.Module):

    def __init__(self, uloss_func):
        super(JointDloss, self).__init__()
        self.gamma = 1
        self.uloss_func = uloss_func

    def forward(self, Y_real, Y_fake, Y_fake_e):

        real_loss, fake_loss, fake_loss_e = 0.0, 0.0, 0.0
        lossD = 0.0

        if self.uloss_func == 'wgan':
            real_loss = Y_real.mean()
            fake_loss = Y_fake.mean()
            fake_loss_e = Y_fake_e.mean()
            lossD = 0.5 * (fake_loss + fake_loss_e) - real_loss
        elif self.uloss_func == 'hinge':
            # label smoothing : 1.0 -> 0.9 (to avoid discriminator become overconfident)
            d_loss_real = torch.nn.ReLU()(1.0 - Y_real).mean()
            d_loss_fake = torch.nn.ReLU()(1.0 + Y_fake).mean()
            d_loss_fake_e = torch.nn.ReLU()(1.0 + Y_fake_e).mean()
            lossD = d_loss_real + d_loss_fake + 0.1 * d_loss_fake_e
        else:
            d_loss_real = torch.nn.functional.binary_cross_entropy_with_logits(Y_real, torch.ones_like(Y_real))
            d_loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake, torch.zeros_like(Y_fake))
            d_loss_fake_e = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e, torch.zeros_like(Y_fake_e))
            lossD = d_loss_real + d_loss_fake + 0.1 * d_loss_fake_e

        return lossD


if __name__ == '__main__':

    Y_real = torch.randn(32, 82, 1)
    Y_fake = torch.randn(32, 82, 1)
    Y_fake_e = torch.randn(32, 82, 1)

    criterion = JointDloss()
    loss = criterion(Y_real, Y_fake, Y_fake_e)
    print(loss)
