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
from Network.embedder import Embedder
from Network.recovery import Recovery
from Network.supervisor import Supervisor
from Network.generator import Generator
from Network.discriminator import Discriminator
from Timedataset import TimeSeriesDataset
from Loss.supervised_loss import SupervisedLoss
from Loss.joint_Gloss import JointGloss
from Loss.joint_Dloss import JointDloss
from utils import random_generator, _gradient_penalty, add_noise

# 取得設定檔
config = configparser.ConfigParser()
config.read('Configure.ini', encoding="utf-8")

# torch.backends.cudnn.deterministic = True
# 固定演算法進行加速
# torch.backends.cudnn.benchmark = True
# 防止 specified launch error: 強行統一至同一GPU
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# 取得設定檔參數
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


# 1. Embedding network training: 預先訓練 data autoencoder
def train_stage1(data_loader, embedder, recovery):

    # Loss
    criterion = SupervisedLoss(dis_func="MSE")

    # model
    embedder.train()
    recovery.train()

    # Optimizer
    optimizer = torch.optim.Adam(
        [{'params': embedder.parameters()},
         {'params': recovery.parameters()}],
        lr=learning_rate1
    )

    # 設定 learning rate 隨 epoch 衰減率
    idx = np.round(np.linspace(0, stage1_epochs, 10)).astype(int)
    idx = idx[1:-1]
    # idx = np.insert(idx, 0, 15)
    scheduler = MultiStepLR(optimizer, milestones=idx, gamma=0.9)

    print('Start Embedding Network Training')

    # trange: 以進度條顯示訓練進度
    logger = trange(stage1_epochs, desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:

        training_loss = 0.0

        for _, (X_mb, T_mb) in enumerate(data_loader):

            X = X_mb.to(CUDA_DEVICES)
            T = T_mb.to(CUDA_DEVICES)

            optimizer.zero_grad()

            H = embedder(X, T)
            outputs = recovery(H, T)

            E_loss_T0, E_loss0 = criterion(outputs, X)
            E_loss0.backward()
            optimizer.step()
            training_loss = E_loss_T0.item()

        logger.set_description(f"Epoch: {epoch}, Loss: {training_loss:.4f}")
        scheduler.step()

    print('Finish Embedding Network Training')

# 2. Training with supervised loss only
def train_stage2(data_loader, embedder,  supervisor, z_recovery):

    # Loss
    criterion = SupervisedLoss(dis_func=dis_func)

    # model
    embedder.train()
    supervisor.train()
    z_recovery.train()

    # Optimizer
    optimizer = torch.optim.Adam(
        [{'params': supervisor.parameters()},
         {'params': z_recovery.parameters()}],
        lr=learning_rate2
    )
    # 設定 learning rate 隨 epoch 衰減率
    idx = np.round(np.linspace(0, stage2_epochs-1, 5)).astype(int)
    idx = idx[1:-1]
    scheduler = MultiStepLR(optimizer, milestones=idx, gamma=0.8)

    print('Start Training with Supervised Loss Only')
    # trange: 以進度條顯示訓練進度
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
            loss, _ = criterion(H_hat_supervise[:, :-1, :], H[:, 1:, :])
            loss.backward()
            optimizer.step()

            training_loss = np.sqrt(loss.item())

        logger.set_description(f"Epoch: {epoch}, Loss: {training_loss:.4f}")
        scheduler.step()

    print('Finish Training with Supervised Loss Only')


# 3. Joint Training
def train_stage3(data_loader, embedder, recovery, z_embedder, z_recovery, supervisor, discriminator, z_discriminator):

    print('Start Joint Training')

    # loss
    # data autoencoder loss
    E_loss_criterion = SupervisedLoss(dis_func=dis_func)
    # noise autoencoder loss
    GZ_loss_crition = SupervisedLoss(dis_func=dis_func)
    # Supervised loss
    GS_loss_criterion = SupervisedLoss(dis_func=dis_func)
    # data discriminator loss
    DD_loss_criterion = JointDloss(uloss_func=uloss_func)
    # data generator loss
    DG_loss_criterion = JointGloss(uloss_func=uloss_func)
    
    # model
    embedder.train()
    recovery.train()
    z_embedder.train()
    z_recovery.train()
    supervisor.train()
    discriminator.train()
    z_discriminator.train()

    # Optimizer
    optimizerE =  torch.optim.Adam(
        [{'params': embedder.parameters()},
         {'params': recovery.parameters()}],
        lr=learning_rate3
    )
    optimizerZE = torch.optim.Adam(
        [{'params': z_embedder.parameters()},
         {'params': z_recovery.parameters()}],
        lr=learning_rate3
    )
    optimizerZG = torch.optim.Adam(
        params=z_embedder.parameters(),
        lr=learning_rate3
    )
    optimizerZD = torch.optim.Adam(
        params= z_discriminator.parameters(),
        lr=(learning_rate3*3.0)
    )
    optimizerGS = torch.optim.Adam(
        [{'params': supervisor.parameters()},
         {'params': z_recovery.parameters()}],
        lr=learning_rate3
    )
    optimizerG = torch.optim.Adam(
        [{'params': supervisor.parameters()},
         {'params': z_recovery.parameters()},
         {'params': recovery.parameters()}],
        lr=learning_rate3
    )
    optimizerD = torch.optim.Adam(
        params=discriminator.parameters(), lr=(learning_rate3*3.0))

    # 設定 learning rate 隨 epoch 衰減率
    # learning rate scheduler
    idx = np.round(np.linspace(0, stage3_epochs-1, 10)).astype(int)
    idx = idx[1:-1]
    schedulerE = MultiStepLR(optimizerE, milestones=idx, gamma=0.8)
    schedulerZE = MultiStepLR(optimizerZE, milestones=idx, gamma=0.8)
    schedulerZG = MultiStepLR(optimizerZG, milestones=idx, gamma=0.6)
    schedulerZD = MultiStepLR(optimizerZD, milestones=idx, gamma=0.6)
    schedulerGS = MultiStepLR(optimizerGS, milestones=idx, gamma=0.6)
    schedulerD = MultiStepLR(optimizerD, milestones=idx, gamma=0.6)
    schedulerG = MultiStepLR(optimizerG, milestones=idx, gamma=0.6)

    # automatic mixed precision (AMP) 節省空間
    scalerE = torch.cuda.amp.GradScaler()
    scalerZE = torch.cuda.amp.GradScaler()
    scalerZG= torch.cuda.amp.GradScaler()
    scalerZD = torch.cuda.amp.GradScaler()
    scalerGS = torch.cuda.amp.GradScaler()
    scalerD = torch.cuda.amp.GradScaler()
    scalerG = torch.cuda.amp.GradScaler()

    Noise_wasserstein_dis = []
    Data_wasserstein_dis = []
    training_loss_D = []
    training_loss_G = []
    training_loss_GZ = []
    training_loss_DZ = []
    
    # trange: 以進度條顯示訓練進度
    if uloss_func == 'wgan':
        logger = trange(
            stage3_epochs,  desc=f"Epoch: 0, Noise wasserstein dis: 0, Data wasserstein dis: 0")
    else:
        logger = trange(
            stage3_epochs,  desc=f"Epoch: 0, Noise G: 0, Noise D: 0, Data G: 0, Data D: 0")
    
    for epoch in logger:

        for X_mb, T_mb in data_loader:
            
            # Get data x
            X = X_mb.to(CUDA_DEVICES)
            T = T_mb.to(CUDA_DEVICES)
            # Generate noise z
            z_batch_size, z_seq_len, z_dim = X.shape
            Z = random_generator(z_batch_size, z_seq_len, z_dim, T_mb)
            Z = Z.to(CUDA_DEVICES)

            ## Train data autoencoder
            optimizerE.zero_grad()

            with torch.cuda.amp.autocast():
                H = embedder(X, T)
                X_tilde = recovery(H, T)
                E_loss_T0, E_loss0 = E_loss_criterion(X_tilde, X)
            
            scalerE.scale(E_loss0).backward() # E_loss0.backward()
            scalerE.step(optimizerE) # optimizerE.step()
            scalerE.update()

            ## Train AAE
            for p in z_discriminator.parameters():
                p.requires_grad = False

            optimizerZE.zero_grad()
            ## Train noise autoencoder
            with torch.cuda.amp.autocast():
                H = embedder(X, T)
                Z_hat = z_embedder(H, T)
                H_hat = z_recovery(Z_hat, T)
                _, ze_loss = GZ_loss_crition(H_hat, H)

            scalerZE.scale(ze_loss).backward() # ze_loss.backward()
            scalerZE.step(optimizerZE) # optimizerZE.step()
            scalerZE.update()

            ## Train noise discriminator
            optimizerZD.zero_grad()
            for p in z_discriminator.parameters():  # reset requires_grad
                p.requires_grad = True

            with torch.cuda.amp.autocast():
                d_out_real = z_discriminator(Z, T)
                Z_hat = z_embedder(H.detach(), T)
                d_out_fake = z_discriminator(Z_hat, T)
                if uloss_func == 'wgan':
                    real_loss = torch.mean(d_out_real)
                    fake_loss = torch.mean(d_out_fake)
                    noise_wasserstein_dis = real_loss - fake_loss
                    with torch.backends.cudnn.flags(enabled=False):
                        lossD_gp = _gradient_penalty(CUDA_DEVICES, z_discriminator, Z, Z_hat, T)
                    d_loss = fake_loss - real_loss + 2 * lossD_gp
                elif uloss_func == 'hinge':
                    # label smoothing : 1.0 -> 0.9 (to avoid discriminator become overconfident)
                    d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
                    d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
                    d_loss = d_loss_real + d_loss_fake
                else:
                    d_loss_real = torch.nn.functional.binary_cross_entropy_with_logits(d_out_real, torch.ones_like(d_out_real))
                    d_loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(d_out_fake, torch.zeros_like(d_out_fake))
                    d_loss = d_loss_real + d_loss_fake
            if d_loss > 0.15:
                scalerZD.scale(d_loss).backward() # d_loss.backward()
                scalerZD.step(optimizerZD) # optimizerZD.step()
                scalerZD.update()

            ## Train noise generator
            optimizerZG.zero_grad()
            for p in z_discriminator.parameters():  # reset requires_grad
                p.requires_grad = False
            
            with torch.cuda.amp.autocast():
                Z_hat = z_embedder(H.detach(), T)
                d_out_fake = z_discriminator(Z_hat, T)
                if uloss_func == 'wgan' or uloss_func == 'hinge':
                    g_loss = -torch.mean(d_out_fake)
                else:
                    g_loss = torch.nn.functional.binary_cross_entropy_with_logits(d_out_fake, torch.ones_like(d_out_fake))
            
            scalerZG.scale(g_loss).backward() # g_loss.backward()
            scalerZG.step(optimizerZG) # optimizerZG.step()
            scalerZG.update()

            ## Train supervised loss
            optimizerGS.zero_grad()
            with torch.cuda.amp.autocast():
                H = embedder(X, T)
                H_hat_supervise = supervisor(H, T)
                # Teacher forcing next output
                _, GS_loss = GS_loss_criterion(H_hat_supervise[:, :-1, :], H[:, 1:, :])
                
            # 可選擇現在 updata or 在下一階段 update
            scalerGS.scale(GS_loss).backward() # GS_loss.backward()
            scalerGS.step(optimizerGS) # optimizerGS.step()
            scalerGS.update()

            ## Train data generator
            optimizerG.zero_grad()
            for p in discriminator.parameters():  # reset requires_grad
                p.requires_grad = False
            with torch.cuda.amp.autocast():
                H = embedder(X, T)
                H_hat_supervise = supervisor(H, T)
                _, GS_loss = GS_loss_criterion(H_hat_supervise[:, :-1, :], H[:, 1:, :])
                E_hat = z_recovery(Z, T)
                H_hat = supervisor(E_hat, T)
                # Synthetic data generated
                X_hat = recovery(H_hat, T)
                X_hat_e = recovery(E_hat, T)
               
                # Adversarial loss
                Y_fake = discriminator(X_hat, T)
                Y_fake_e = discriminator(X_hat_e, T)
                lossG = DG_loss_criterion(Y_fake, Y_fake_e)
                # Add supervised loss
                lossG = lossG + GS_loss
            
            scalerG.scale(lossG).backward() # lossG.backward()
            scalerG.step(optimizerG) # optimizerG.step()
            scalerG.update()

            ## Data Discriminator training
            for p in discriminator.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update
            optimizerD.zero_grad()

            with torch.cuda.amp.autocast():
                H = embedder(X, T)
                H_hat_supervise = supervisor(H, T)
                E_hat = z_recovery(Z, T)
                H_hat = supervisor(E_hat, T)
                X_hat = recovery(H_hat, T)
                X_hat_e = recovery(E_hat, T)
                Y_real = discriminator(X, T)
                Y_fake = discriminator(X_hat, T)    # Output of supervisor
                Y_fake_e = discriminator(X_hat_e, T)   # Output of generator

                lossD = 0.0
                _, GS_loss = GS_loss_criterion(H_hat_supervise[:, :-1, :], H[:, 1:, :])
                lossD = DD_loss_criterion(Y_real, Y_fake, Y_fake_e)
                if uloss_func == 'wgan':
                    data_wasserstein_dis = Y_real.mean() - 0.5 * (Y_fake.mean() + Y_fake_e.mean())
                    with torch.backends.cudnn.flags(enabled=False):
                        lossD_gp = _gradient_penalty(CUDA_DEVICES, discriminator, X, X_hat, T)
                    lossD = lossD + 10 * lossD_gp
                # Add supervised loss
                lossD = lossD + GS_loss
            # only when the generator does not work well
            if lossD > 0.15:
                scalerD.scale(lossD).backward() # lossD.backward()
                scalerD.step(optimizerD) # optimizerD.step()
                scalerD.update()

        if uloss_func == 'wgan':
            Noise_wasserstein_dis.append(noise_wasserstein_dis.item())
            Data_wasserstein_dis.append(data_wasserstein_dis.item())
            logger.set_description(
                f"Epoch: {epoch}, Noise wasserstein dis: {noise_wasserstein_dis.item():.4f}, Data wasserstein dis: {data_wasserstein_dis.item():.4f}"
            )
        else:
            training_loss_D.append(lossD.item())
            training_loss_G.append(lossG.item())
            training_loss_GZ.append(g_loss.item())
            training_loss_DZ.append(d_loss.item())
            logger.set_description(
                f"Epoch: {epoch}, Noise G: {g_loss.item():.4f}, Noise D: {d_loss.item():.4f}, Data G: {lossG.item():.4f}, Data D: {lossD.item():.4f}"
            )

        schedulerE.step()
        schedulerZE.step()
        schedulerZG.step()
        schedulerZD.step()
        schedulerGS.step()
        schedulerD.step()
        schedulerG.step()

        # Save multiple checkpoints
        if epoch % 100 == 0:
            torch.save(recovery.state_dict(), f'{output_dir+str(epoch)+"_"+recovery_name}')
            torch.save(z_recovery.state_dict(), f'{output_dir+str(epoch)+"_"+generator_name}')
            torch.save(supervisor.state_dict(), f'{output_dir+str(epoch)+"_"+supervisor_name}')
            torch.save(discriminator.state_dict(), f'{output_dir+str(epoch)+"_discriminator.pth"}')
            torch.save(z_discriminator.state_dict(), f'{output_dir+str(epoch)+"_Zdiscriminator.pth"}')
            torch.save(embedder.state_dict(), f'{output_dir+str(epoch)+"_embedder.pth"}')
            torch.save(z_embedder.state_dict(), f'{output_dir+str(epoch)+"_Zembedder.pth"}')

    # 畫圖驗證 loss 
    if uloss_func == 'wgan':
        Noise_wasserstein_dis
        plt.plot(Noise_wasserstein_dis, color='blue', label="Noise W dis")
        plt.plot(Data_wasserstein_dis, color='green', label="Data W dis")
        plt.title("WGAN Training loss")
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig('./Loss_curve/training_loss_curve.png', bbox_inches='tight')
        plt.close()
    else:
        plt.plot(training_loss_D, color='red', label="Data D")
        plt.plot(training_loss_G, color='green', label="Data G")
        plt.plot(training_loss_GZ, color='blue', label="Noise G")
        plt.plot(training_loss_DZ, color='orange', label="Noise D")
        plt.title("Hinge Training loss")
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig('./Loss_curve/training_loss_curve.png', bbox_inches='tight')
        plt.close()

    print('Finish Joint Training')


if __name__ == '__main__':

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
        dataset=Data_set, batch_size=batch_size, shuffle=False, num_workers=0)

    Max_Seq_len = Data_set.max_seq_len

    # models
    embedder = Embedder(
        module=module_name,
        mode='data',
        time_stamp=seq_len,
        input_size=n_features,
        hidden_dim=hidden_size,
        output_dim=hidden_size,
        num_layers=num_layers,
        activate_function=nn.Tanh(),
        padding_value=PADDING_VALUE,
        max_seq_len=Max_Seq_len
    )

    recovery = Recovery(
        module=module_name,
        mode='data',
        time_stamp=seq_len,
        input_size=hidden_size,
        hidden_dim=hidden_size,
        output_dim=n_features,
        num_layers=num_layers,
        activate_function=nn.Tanh(),
        padding_value=PADDING_VALUE,
        max_seq_len=Max_Seq_len
    )

    z_embedder = Embedder(
        module=module_name,
        mode='noise',
        time_stamp=seq_len,
        input_size=hidden_size,
        hidden_dim=hidden_size,
        output_dim=n_features,
        num_layers=num_layers,
        activate_function=nn.Tanh(),
        padding_value=PADDING_VALUE,
        max_seq_len=Max_Seq_len
    )

    z_recovery = Recovery(
        module=module_name,
        mode='noise',
        time_stamp=seq_len,
        input_size=n_features,
        hidden_dim=hidden_size,
        output_dim=hidden_size,
        num_layers=num_layers,
        activate_function=nn.Tanh(),
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
        activate_function=nn.Tanh(),
        padding_value=PADDING_VALUE,
        max_seq_len=Max_Seq_len
    )

    discriminator = Discriminator(
        module=module_name,
        time_stamp=seq_len,
        input_size=n_features,
        hidden_dim=hidden_size,
        output_dim=1,
        num_layers=num_layers,
        activate_function=nn.Tanh(),
        padding_value=PADDING_VALUE,
        max_seq_len=Max_Seq_len
    )

    z_discriminator = Discriminator(
        module=module_name,
        time_stamp=seq_len,
        input_size=n_features,
        hidden_dim=hidden_size,
        output_dim=1,
        num_layers=num_layers,
        activate_function=nn.Tanh(),
        padding_value=PADDING_VALUE,
        max_seq_len=Max_Seq_len
    )

    # embedder.load_state_dict(torch.load('/home/kent1201/Documents/Master-thesis/models/26_05_2021/action1_gru_MSE_hinge_6000_64_82_27_108_5/1000_embedder.pth'))
    # recovery.load_state_dict(torch.load('/home/kent1201/Documents/Master-thesis/models/26_05_2021/action1_gru_MSE_hinge_6000_64_82_27_108_5/1000_recovery.pth'))
    # z_embedder.load_state_dict(torch.load('/home/kent1201/Documents/Master-thesis/models/26_05_2021/action1_gru_MSE_hinge_6000_64_82_27_108_5/1000_Zembedder.pth'))
    # z_recovery.load_state_dict(torch.load('/home/kent1201/Documents/Master-thesis/models/26_05_2021/action1_gru_MSE_hinge_6000_64_82_27_108_5/1000_generator.pth'))
    # supervisor.load_state_dict(torch.load('/home/kent1201/Documents/Master-thesis/models/26_05_2021/action1_gru_MSE_hinge_6000_64_82_27_108_5/1000_supervisor.pth'))
    # discriminator.load_state_dict(torch.load('/home/kent1201/Documents/Master-thesis/models/26_05_2021/action1_gru_MSE_hinge_6000_64_82_27_108_5/1000_discriminator.pth'))
    # z_discriminator.load_state_dict(torch.load('/home/kent1201/Documents/Master-thesis/models/26_05_2021/action1_gru_MSE_hinge_6000_64_82_27_108_5/1000_Zdiscriminator.pth'))


    embedder = embedder.to(CUDA_DEVICES)
    recovery = recovery.to(CUDA_DEVICES)
    z_embedder = z_embedder.to(CUDA_DEVICES)
    z_recovery = z_recovery.to(CUDA_DEVICES)
    supervisor = supervisor.to(CUDA_DEVICES)
    discriminator = discriminator.to(CUDA_DEVICES)
    z_discriminator = z_discriminator.to(CUDA_DEVICES)


    train_stage1(Data_loader, embedder, recovery)
    train_stage2(Data_loader, embedder, supervisor, z_recovery)
    train_stage3(Data_loader, embedder, recovery, z_embedder, z_recovery, supervisor, discriminator, z_discriminator)

    torch.save(recovery.state_dict(), f'{output_dir+recovery_name}')
    torch.save(z_recovery.state_dict(), f'{output_dir+generator_name}')
    torch.save(supervisor.state_dict(), f'{output_dir+supervisor_name}')

    print('Finish Saving Models.')
