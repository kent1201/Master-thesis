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
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from Network.embedder import Embedder
from Network.recovery import Recovery
from Network.supervisor import Supervisor
from Network.generator import Generator
from Network.discriminator import Discriminator
# from dataset import SensorSignalDataset
from Timedataset import TimeSeriesDataset
from Loss.embedder_loss import EmbedderLoss
from Loss.supervised_loss import SupervisedLoss
from Loss.joint_Gloss import JointGloss
from Loss.joint_Eloss import JointEloss
from Loss.joint_Dloss import JointDloss
from utils import random_generator, _gradient_penalty


config = configparser.ConfigParser()
config.read('Configure.ini', encoding="utf-8")

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True

# gpu-used
CUDA_DEVICES = torch.device("cuda:"+config.get('default',
                                               'cuda_device_number') if torch.cuda.is_available() else "cpu")
dataset_dir = config.get('train', 'Dataset_path')
stage1_epochs = config.getint('train', 'stage1_epochs')
stage2_epochs = config.getint('train', 'stage2_epochs')
stage3_epochs = config.getint('train', 'stage3_epochs')
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
embedder_name = config.get('default', 'embedder_name')
recovery_name = config.get('default', 'recovery_name')
generator_name = config.get('default', 'generator_name')
supervisor_name = config.get('default', 'supervisor_name')
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


# 1. Embedding network training


def train_stage1(data_loader, embedder, recovery):

    # Loss
    criterion = EmbedderLoss(dis_func="MSE")

    # model
    embedder.train()
    recovery.train()

    # Optimizer
    optimizer = torch.optim.Adam(
        [{'params': embedder.parameters()},
         {'params': recovery.parameters()}],
        lr=learning_rate1
    )

    idx = np.round(np.linspace(0, stage1_epochs, 5)).astype(int)
    idx = idx[1:-1]
    # idx = np.insert(idx, 0, 15)
    scheduler = MultiStepLR(optimizer, milestones=idx, gamma=0.9)

    print('Start Embedding Network Training')

    # for epoch in range(num_epochs):
    logger = trange(stage1_epochs, desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:

        training_loss = 0.0

        for _, (X_mb, T_mb) in enumerate(data_loader):

            X = X_mb.to(CUDA_DEVICES)
            T = T_mb.to(CUDA_DEVICES)

            optimizer.zero_grad()

            H = embedder(X, T)
            # For attention
            if module_name == "self-attn":
                decoder_inputs = torch.zeros_like(X)
                decoder_inputs = decoder_inputs.to(CUDA_DEVICES)
                outputs = recovery(H, decoder_inputs)
            # For GRU
            else:
                outputs = recovery(H, T)

            E_loss_T0, E_loss0 = criterion(outputs, X)
            E_loss0.backward()
            optimizer.step()
            training_loss = np.sqrt(E_loss_T0.item())

        logger.set_description(f"Epoch: {epoch}, Loss: {training_loss:.4f}")
        scheduler.step()

        # if epoch % (np.round(num_epochs / 5))  == 0:
        #   print('epoch: '+ str(epoch) + '/' + str(num_epochs) + ', e_loss: ' + str(np.round(training_loss,4)))

    print('Finish Embedding Network Training')


# 2. Training only with supervised loss
def train_stage2(data_loader, embedder, supervisor, generator):

    # Loss
    criterion = SupervisedLoss(dis_func=dis_func)

    # model
    embedder.train()
    supervisor.train()
    generator.train()

    # # Optimizer
    optimizer = torch.optim.Adam(
        [{'params': generator.parameters()},
         {'params': supervisor.parameters()}],
        lr=learning_rate2
    )

    idx = np.round(np.linspace(0, stage2_epochs-1, 5)).astype(int)
    idx = idx[1:-1]
    scheduler = MultiStepLR(optimizer, milestones=idx, gamma=0.8)

    print('Start Training with Supervised Loss Only')

    logger = trange(stage2_epochs, desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:

        training_loss = 0.0

        for _, (X_mb, T_mb) in enumerate(data_loader):

            X = X_mb.to(CUDA_DEVICES)
            T = T_mb.to(CUDA_DEVICES)

            optimizer.zero_grad()

            H = embedder(X, T)

            H_hat_supervise = supervisor(H, T)

            # Teacher forcing next output
            loss = criterion(H_hat_supervise[:, :-1, :], H[:, 1:, :])
            loss.backward()
            optimizer.step()

            training_loss = np.sqrt(loss.item())

        logger.set_description(f"Epoch: {epoch}, Loss: {training_loss:.4f}")
        scheduler.step()

    print('Finish Training with Supervised Loss Only')


# 3. Joint Training
def train_stage3(data_loader, embedder, recovery, generator, supervisor, discriminator):

    print('Start Joint Training')

    # generator loss
    Gloss_criterion = JointGloss(dis_func=dis_func, mode=uloss_func)
    Eloss_criterion = JointEloss(dis_func=dis_func)
    Dloss_criterion = JointDloss(mode=uloss_func)

    # recorded wasserstein distance curve
    wasserstein_dis_list = []
    embedding_loss_list = []
    generator_loss_list = []
    discriminator_loss_list = []
    
    # model
    embedder.train()
    recovery.train()
    generator.train()
    supervisor.train()
    discriminator.train()

    # Optimizer
    optimizerG = torch.optim.Adam(
        [{'params': generator.parameters()},
         {'params': supervisor.parameters()}],
        lr=learning_rate3
    )
    optimizerD = torch.optim.Adam(
        params=discriminator.parameters(), lr=learning_rate5)

    optimizerE = torch.optim.Adam(
        [{'params': embedder.parameters()},
         {'params': recovery.parameters()}],
        lr=learning_rate4
    )

    idx = np.round(np.linspace(0, stage3_epochs-1, 5)).astype(int)
    idx = idx[1:-1]
    schedulerE = MultiStepLR(optimizerE, milestones=idx, gamma=0.8)
    schedulerD = MultiStepLR(optimizerD, milestones=idx, gamma=0.8)
    schedulerG = MultiStepLR(optimizerG, milestones=idx, gamma=0.8)

    if uloss_func == "wgan":
        logger = trange(
            stage3_epochs,  desc=f"Epoch: 0, E_loss: 0, wasserstein_loss: 0")
    else:
        logger = trange(
            stage3_epochs,  desc=f"Epoch: 0, E_loss: 0, G_loss: 0, D_loss: 0")
    for epoch in logger:

        training_loss_G = 0.0
        # training_loss_U = 0.0
        # training_loss_S = 0.0
        # training_loss_V = 0.0
        training_loss_E0 = 0.0
        training_loss_D = 0.0
        wasserstein_loss = 0.0

        for X_mb, T_mb in data_loader:

            # Generator training (twice more than discriminator training)

            for p in discriminator.parameters():
                p.requires_grad = False  # to avoid computation

            for _ in range(2):

                X = X_mb.to(CUDA_DEVICES)
                T = T_mb.to(CUDA_DEVICES)

                optimizerG.zero_grad()

                # Train generator
                # Generate random data
                z_batch_size, z_seq_len, z_dim = X.shape
                Z = random_generator(z_batch_size, z_seq_len, z_dim, T_mb)
                Z = Z.to(CUDA_DEVICES)

                # Supervised Forward Pass
                H = embedder(X, T)
                H_hat_supervise = supervisor(H, T)
                # For attention
                if module_name == "self-attn":
                    decoder_inputs = torch.zeros_like(X)
                    decoder_inputs = decoder_inputs.to(CUDA_DEVICES)
                    X_tilde = recovery(H, decoder_inputs)
                # For GRU
                else:
                    X_tilde = recovery(H, T)

                # Generator Forward Pass
                E_hat = generator(Z, T)
                H_hat = supervisor(E_hat, T)

                # Synthetic data generated
                # For attention
                if module_name == "self-attn":
                    decoder_inputs = torch.zeros_like(X)
                    decoder_inputs = decoder_inputs.to(CUDA_DEVICES)
                    X_hat = recovery(H_hat, decoder_inputs)
                # For GRU
                else:
                    X_hat = recovery(H_hat, T)

                # Adversarial loss
                Y_fake = discriminator(H_hat, T)
                Y_fake_e = discriminator(E_hat, T)

                lossG, loss_U, loss_S, loss_V = Gloss_criterion(
                    Y_fake, Y_fake_e, H[:, 1:, :], H_hat_supervise[:, :-1, :], X, X_hat)

                lossG.backward()

                optimizerG.step()

                # Train autoencoder
                optimizerE.zero_grad()

                H = embedder(X, T)
                # For attention
                if module_name == "self-attn":
                    decoder_inputs = torch.zeros_like(X)
                    decoder_inputs = decoder_inputs.to(CUDA_DEVICES)
                    X_tilde = recovery(H, decoder_inputs)
                # For GRU
                else:
                    X_tilde = recovery(H, T)

                H_hat_supervise = supervisor(H, T)

                lossE, lossE_0 = Eloss_criterion(
                    X_tilde, X, H[:, 1:, :], H_hat_supervise[:, :-1, :])

                lossE.backward()
                optimizerE.step()

            training_loss_E0 = np.sqrt(lossE_0.item())
            training_loss_G = np.sqrt(lossG.item())
            generator_loss_list.append(training_loss_G)
            embedding_loss_list.append(training_loss_E0)
            # training_loss_U = np.sqrt(loss_U.item())
            # training_loss_S = np.sqrt(loss_S.item())
            # training_loss_V = np.sqrt(loss_V.item())

            ## Discriminator training

            for p in discriminator.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            optimizerD.zero_grad()

            # Train discriminator
            # Generate random data
            z_batch_size, z_seq_len, z_dim = X.shape
            Z = random_generator(z_batch_size, z_seq_len, z_dim, T_mb)
            Z = Z.to(CUDA_DEVICES)

            # latent code forward
            H = embedder(X, T).detach()
            H_hat = supervisor(E_hat, T).detach()
            E_hat = generator(Z, T).detach()

            # Forward Pass
            # Encoded original data
            Y_real = discriminator(H, T)
            # Output of supervisor
            Y_fake = discriminator(H_hat, T)
            Y_fake_e = discriminator(
                E_hat, T)   # Output of generator

            if uloss_func == "wgan":
                # Adversarial loss
                lossD_real, lossD_fake, lossD_fake_e = Dloss_criterion(
                    Y_real, Y_fake, Y_fake_e)
                wasserstein_dis = lossD_real - \
                    0.5 * (lossD_fake + lossD_fake_e)
                with torch.backends.cudnn.flags(enabled=False):
                    lossD_gp = _gradient_penalty(
                        CUDA_DEVICES, discriminator, H, H_hat, T)
                lossD = lossD_fake + lossD_fake_e - lossD_real + lossD_gp
                lossD.backward()
                optimizerD.step()
            else:
                # Adversarial loss
                lossD = Dloss_criterion(Y_real, Y_fake, Y_fake_e)
                # Train discriminator (only when the discriminator does not work well)
                if lossD > 0.15:
                    lossD.backward()
                    optimizerD.step()

        if uloss_func == "wgan":
            wasserstein_loss = wasserstein_dis.item()
            wasserstein_dis_list.append(wasserstein_loss)
        else:
            training_loss_D = lossD.item()
            discriminator_loss_list.append(training_loss_D)

        if uloss_func == "wgan":
            logger.set_description(
                f"Epoch: {epoch}, E: {training_loss_E0:.4f}, wasserstein_loss: {wasserstein_loss:.4f}"
            )
        else:
            logger.set_description(
                f"Epoch: {epoch}, E: {training_loss_E0:.4f}, G: {training_loss_G:.4f}, D: {training_loss_D:.4f}"
            )

        schedulerE.step()
        schedulerD.step()
        schedulerG.step()

        # Save multiple checkpoints
        # if epoch % (np.round(int(stage3_epochs) // 5)) == 0:
        if epoch % 10 == 0:
            torch.save(embedder, f'{output_dir+str(epoch)+"_"+embedder_name}')
            torch.save(recovery, f'{output_dir+str(epoch)+"_"+recovery_name}')
            torch.save(generator, f'{output_dir+str(epoch)+"_"+generator_name}')
            torch.save(supervisor, f'{output_dir+str(epoch)+"_"+supervisor_name}')
            torch.save(discriminator, f'{output_dir+str(epoch)+"_"+discriminator_name}')
    
    if uloss_func == "wgan":
        plt.plot(wasserstein_dis_list, color='brown', label="wasserstein")
    else:
        plt.plot(discriminator_loss_list, color='red', label="discriminative")
        plt.plot(generator_loss_list, color='green', label="generative")
    
    plt.plot(embedding_loss_list, color='blue', label="embedding")
    plt.title("Training loss")
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
    print("[train] num_epochs: {}".format(stage3_epochs))
    print("[train] batch_size: {}".format(batch_size))
    print("[train] distance function: {}".format(dis_func))
    print("[train] adversarial loss function: {}".format(uloss_func))

    # Dataset
    Data_set = TimeSeriesDataset(
        root_dir=dataset_dir, seq_len=seq_len, transform=None)
    Data_loader = DataLoader(
        dataset=Data_set, batch_size=batch_size, shuffle=False, num_workers=1)

    Max_Seq_len = Data_set.max_seq_len

    # models
    embedder = Embedder(
        module=module_name,
        time_stamp=seq_len,
        input_size=n_features,
        hidden_dim=hidden_size,
        output_dim=hidden_size,
        num_layers=num_layers,
        activate_function=nn.Sigmoid(),
        padding_value=PADDING_VALUE,
        max_seq_len=Max_Seq_len
    )

    recovery = Recovery(
        module=module_name,
        time_stamp=seq_len,
        input_size=hidden_size,
        hidden_dim=hidden_size,
        output_dim=n_features,
        num_layers=num_layers,
        # Only for attention
        activate_function=nn.Sigmoid(),
        padding_value=PADDING_VALUE,
        max_seq_len=Max_Seq_len
    )

    generator = Generator(
        module=module_name,
        time_stamp=seq_len,
        input_size=n_features,
        hidden_dim=hidden_size,
        output_dim=hidden_size,
        num_layers=num_layers,
        activate_function=nn.Sigmoid(),
        padding_value=PADDING_VALUE,
        max_seq_len=Max_Seq_len
    )

    supervisor = Supervisor(
        module=module_name,
        time_stamp=seq_len,
        input_size=hidden_size,
        hidden_dim=hidden_size,
        output_dim=hidden_size,
        # [Supervisor] num_layers must less(-1) than other component, embedder
        num_layers=num_layers - 1,
        activate_function=nn.Sigmoid(),
        padding_value=PADDING_VALUE,
        max_seq_len=Max_Seq_len
    )

    discriminator = Discriminator(
        module=module_name,
        time_stamp=seq_len,
        input_size=hidden_size,
        hidden_dim=hidden_size,
        output_dim=1,
        num_layers=num_layers,
        # activate_function=nn.Sigmoid(),
        padding_value=PADDING_VALUE,
        max_seq_len=Max_Seq_len
    )

    embedder = embedder.to(CUDA_DEVICES)
    recovery = recovery.to(CUDA_DEVICES)
    generator = generator.to(CUDA_DEVICES)
    supervisor = supervisor.to(CUDA_DEVICES)
    discriminator = discriminator.to(CUDA_DEVICES)

    train_stage1(Data_loader, embedder, recovery)
    train_stage2(Data_loader, embedder, supervisor, generator)
    train_stage3(Data_loader, embedder, recovery,
                 generator, supervisor, discriminator)

    torch.save(embedder, f'{output_dir+embedder_name}')
    torch.save(recovery, f'{output_dir+recovery_name}')
    torch.save(generator, f'{output_dir+generator_name}')
    torch.save(supervisor, f'{output_dir+supervisor_name}')
    torch.save(discriminator, f'{output_dir+discriminator_name}')

    print('Finish Saving Models.')
