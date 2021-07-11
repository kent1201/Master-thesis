import math
import configparser
import os
from datetime import date
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import itertools
import random
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from Network.c_rnn_gan import Generator
from Network.c_rnn_gan import Discriminator
# from dataset import WaferDataset
from Timedataset import TimeSeriesDataset
from utils import random_generator


config = configparser.ConfigParser()
config.read('Configure.ini', encoding="utf-8")

# torch.backends.cudnn.deterministic = True
# 固定演算法進行加速
# torch.backends.cudnn.benchmark = True
# 防止 specified launch error: 強行統一至同一GPU
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# gpu-used
CUDA_DEVICES = torch.device("cuda:"+config.get('default',
                                               'cuda_device_number') if torch.cuda.is_available() else "cpu")
dataset_dir = config.get('train', 'Dataset_path')
stage1_epochs = config.getint('train', 'stage1_epochs')
stage2_epochs = config.getint('train', 'stage2_epochs')
stage3_epochs = config.getint('train', 'stage3_epochs')
stage4_epochs = config.getint('train', 'stage4_epochs')
stage5_epochs = config.getint('train', 'stage5_epochs')
batch_size = config.getint('train', 'batch_size')
seq_len = config.getint('train', 'seq_len')
n_features = config.getint('train', 'n_features')
hidden_size = config.getint('train', 'hidden_size')
num_layers = config.getint('train', 'num_layers')
learning_rate1 = config.getfloat('train', 'learning_rate1')
learning_rate2 = config.getfloat('train', 'learning_rate2')
learning_rate3 = config.getfloat('train', 'learning_rate3')
learning_rate4 = config.getfloat('train', 'learning_rate4')
learning_rate5 = config.getfloat('train', 'learning_rate5')
dis_func = config.get('train', 'dis_func')
uloss_func = config.get('train', 'uloss_func')
generator_name = config.get('default', 'generator_name')
discriminator_name = config.get('default', 'discriminator_name')
module_name = config.get('default', 'module_name')
PADDING_VALUE = config.getfloat('default', 'padding_value')


# save model path
today = date.today()
save_time = today.strftime("%d_%m_%Y")
output_dir = config.get('train', 'model_path') + '/' + save_time + \
    '/' + config.get('train', 'classification_dir') + '/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


class GLoss(nn.Module):
    ''' C-RNN-GAN generator loss
    '''
    def __init__(self):
        super(GLoss, self).__init__()

    def forward(self, logits_gen):
        logits_gen = torch.clamp(logits_gen, 1e-40, 1.0)
        batch_loss = -torch.log(logits_gen)

        return torch.mean(batch_loss)


class DLoss(nn.Module):
    ''' C-RNN-GAN discriminator loss
    '''
    def __init__(self, label_smoothing=False):
        super(DLoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits_real, logits_gen):
        ''' Discriminator loss
        logits_real: logits from D, when input is real
        logits_gen: logits from D, when input is from Generator
        loss = -(ylog(p) + (1-y)log(1-p))
        '''
        logits_real = torch.clamp(logits_real, 1e-40, 1.0)
        d_loss_real = -torch.log(logits_real)

        if self.label_smoothing:
            p_fake = torch.clamp((1 - logits_real), 1e-40, 1.0)
            d_loss_fake = -torch.log(p_fake)
            d_loss_real = 0.9*d_loss_real + 0.1*d_loss_fake

        logits_gen = torch.clamp((1 - logits_gen), 1e-40, 1.0)
        d_loss_gen = -torch.log(logits_gen)

        batch_loss = d_loss_real + d_loss_gen
        return torch.mean(batch_loss)

# Start Training
def train_stage(data_loader, generator, discriminator):

    print('Start Training')

    # loss
    Gloss_criterion = GLoss()
    Dloss_criterion = DLoss(label_smoothing=True)

    # model
    discriminator.train()
    generator.train()

    # Optimizer
    optimizerG = torch.optim.Adam(
        generator.parameters(),
        lr=learning_rate5
    )
    optimizerD = torch.optim.Adam(
        params=discriminator.parameters(), lr=(learning_rate5*3.0))

    # learning rate scheduler
    idx = np.round(np.linspace(0, stage5_epochs-1, 10)).astype(int)
    idx = idx[1:-1]
    schedulerD = MultiStepLR(optimizerD, milestones=idx, gamma=0.8)
    schedulerG = MultiStepLR(optimizerG, milestones=idx, gamma=0.8)

    # automatic mixed precision (AMP) 節省空間
    scalerD = torch.cuda.amp.GradScaler()
    scalerG = torch.cuda.amp.GradScaler()

    training_loss_D = []
    training_loss_G = []

    logger = trange(
        stage5_epochs,  desc=f"Data G: 0, Data D: 0")
    
    for epoch in logger:

        for X_mb, T_mb in data_loader:

            X = X_mb.to(CUDA_DEVICES)
            T = T_mb.to(CUDA_DEVICES)
            z_batch_size, z_seq_len, z_dim = X.shape
            Z = random_generator(z_batch_size, z_seq_len, z_dim, T_mb)
            Z = Z.to(CUDA_DEVICES)

            # inital state
            d_state = discriminator.init_hidden(z_batch_size)
            g_state = generator.init_hidden(z_batch_size)

            
            ## Train generator
            optimizerG.zero_grad()
            for p in discriminator.parameters():  # reset requires_grad
                p.requires_grad = False
            with torch.cuda.amp.autocast():
                X_hat, _ = generator(Z, g_state)
                Y_fake, _, _ = discriminator(X_hat, d_state)
                lossG =  Gloss_criterion(Y_fake)
            
            scalerG.scale(lossG).backward() # lossG.backward()
            scalerG.step(optimizerG) # optimizerG.step()
            scalerG.update()

            ## Discriminator training

            for p in discriminator.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update
            optimizerD.zero_grad()

            with torch.cuda.amp.autocast():
                Y_real, _, _ = discriminator(X, d_state)
                Y_fake, _, _ = discriminator(X_hat.detach(), d_state)    # Output of supervisor
                lossD = Dloss_criterion(Y_real, Y_fake)
            # Train discriminator (only when the discriminator does not work well)
            if lossD > 0.15:
                scalerD.scale(lossD).backward() # lossD.backward()
                scalerD.step(optimizerD) # optimizerD.step()
                scalerD.update()

        training_loss_D.append(lossD.item())
        training_loss_G.append(lossG.item())
        logger.set_description(
            f"Epoch: {epoch}, Data G: {lossG.item():.4f}, Data D: {lossD.item():.4f}"
        )

        schedulerD.step()
        schedulerG.step()

        # Save multiple checkpoints
        if epoch % 100 == 0:
            torch.save(generator.state_dict(), f'{output_dir+str(epoch)+"_"+generator_name}')
            torch.save(discriminator.state_dict(), f'{output_dir+str(epoch)+"_"+discriminator_name}')


    plt.plot(training_loss_D, color='red', label="Data D")
    plt.plot(training_loss_G, color='green', label="Data G")
    plt.title("C_RNN_GAN Training loss")
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('./Loss_curve/training_loss_curve.png', bbox_inches='tight')
    plt.close()

    print('Finish Joint Training')


if __name__ == '__main__':

    # # save model path
    # today = date.today()
    # save_time = today.strftime("%d_%m_%Y")
    # output_dir = config.get('train', 'model_path') + '/' + save_time + \
    #     '/' + config.get('train', 'classification_dir') + '/'
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # Parameters
    print("CUDA DEVICE: {}".format(CUDA_DEVICES))
    print("[train] module: {}".format(module_name))
    print("[train] action: {}".format(config.get(
        'train', 'classification_dir').split('_')[0]))
    print("[train] seq_len: {}".format(seq_len))
    print("[train] n_features: {}".format(n_features))
    print("[train] hidden size: {}".format(hidden_size))
    print("[train] num_layers: {}".format(num_layers))
    print("[train] num_epochs: {}".format(stage5_epochs))
    print("[train] batch_size: {}".format(batch_size))
    print("[train] distance function: {}".format(dis_func))
    print("[train] adversarial loss function: {}".format(uloss_func))

    # Dataset
    Data_set = TimeSeriesDataset(
        root_dir=dataset_dir, seq_len=seq_len, transform=None)
    Data_loader = DataLoader(
        dataset=Data_set, batch_size=batch_size, shuffle=False, num_workers=0)

    Max_Seq_len = Data_set.max_seq_len

    # models
    generator = Generator(n_features, CUDA_DEVICES, hidden_units=hidden_size)
    discriminator = Discriminator(n_features, CUDA_DEVICES, hidden_units=hidden_size)

    # discriminator.load_state_dict(torch.load('/home/kent1201/Documents/Master-thesis/models/26_05_2021/action1_gru_MSE_hinge_6000_64_82_27_108_5/1000_discriminator.pth'))
    # generator.load_state_dict(torch.load('/home/kent1201/Documents/Master-thesis/models/26_05_2021/action1_gru_MSE_hinge_6000_64_82_27_108_5/1000_discriminator.pth'))

    discriminator = discriminator.to(CUDA_DEVICES)
    generator = generator.to(CUDA_DEVICES)

    train_stage(Data_loader, generator, discriminator)

    torch.save(discriminator.state_dict(), f'{output_dir+discriminator_name}')
    torch.save(generator.state_dict(), f'{output_dir+generator_name}')

    print('Finish Saving Models.')
