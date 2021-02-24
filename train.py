import math
import configparser
import os
from datetime import date
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import itertools
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
CUDA_DEVICES = torch.device("cuda:"+config.get('default', 'cuda_device_number') if torch.cuda.is_available() else "cpu")
dataset_dir = config.get('train', 'Dataset_path')
num_epochs = config.getint('train', 'num_epochs')
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

# 1. Embedding network training
def train_stage1(data_set, data_loader, embedder, recovery):

  # Loss
  criterion = EmbedderLoss(dis_func = "MSE")

  # model
  embedder.train()
  recovery.train()

  # Optimizer
  # models_param = [embedder.parameters(), recovery.parameters()]
  # optimizer = torch.optim.Adam(params=itertools.chain(*models_param), lr=learning_rate)
  optimizer = torch.optim.Adam(
    [{'params': embedder.parameters()},
     {'params': recovery.parameters()}],
    lr=learning_rate1
  )

  print('Start Embedding Network Training')

  for epoch in range(num_epochs):

    training_loss = 0.0

    for _, inputs in enumerate(data_loader):

      X = inputs[0].to(CUDA_DEVICES)

      optimizer.zero_grad()

      H = embedder(X, None)
      # For attention
      # decoder_inputs = torch.zeros_like(X)
      # decoder_inputs = decoder_inputs.to(CUDA_DEVICES)
      # outputs = recovery(H, decoder_inputs)
      # For GRU
      outputs = recovery(H, None)

      E_loss_T0, E_loss0 = criterion(outputs, X)
      E_loss0.backward()
      optimizer.step()
      training_loss = np.sqrt(E_loss_T0.item())

    if epoch % (np.round(num_epochs / 5))  == 0:
      print('epoch: '+ str(epoch) + '/' + str(num_epochs) + ', e_loss: ' + str(np.round(training_loss,4)))

  print('Finish Embedding Network Training')

# 2. Training only with supervised loss
def train_stage2(data_set, data_loader, embedder, supervisor, generator):

  # Loss
  criterion = SupervisedLoss(dis_func=dis_func)

  # model
  embedder.train()
  supervisor.train()
  generator.train()

  # Optimizer
  # models_param = [generator.parameters(), supervisor.parameters()]
  # optimizer = torch.optim.Adam(params=itertools.chain(*models_param), lr=learning_rate)

  optimizer = torch.optim.Adam(
    [{'params': generator.parameters()},
     {'params': supervisor.parameters()}],
    lr=learning_rate2
  )

  # optimizer = torch.optim.RMSprop(
  #   [{'params': generator.parameters()},
  #    {'params': supervisor.parameters()}],
  #    lr=learning_rate2, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

  print('Start Training with Supervised Loss Only')

  for epoch in range(num_epochs):

    training_loss = 0.0

    for _, inputs in enumerate(data_loader):

      X = inputs[0].to(CUDA_DEVICES)


      optimizer.zero_grad()

      H = embedder(X, None)

      H_hat_supervise = supervisor(H, None)

      loss = criterion(H_hat_supervise[:,:-1,:], H[:,1:,:])
      loss.backward()
      optimizer.step()

      training_loss = np.sqrt(loss.item())

    if epoch % (np.round(num_epochs / 5)) == 0:
      print('epoch: '+ str(epoch) + '/' + str(num_epochs) + ', s_loss: ' + str(np.round(training_loss, 4)))


  print('Finish Training with Supervised Loss Only')

# 3. Joint Training
def train_stage3(data_set, data_loader, embedder, recovery, generator, supervisor, discriminator):

  print('Start Joint Training')

  # generator loss
  Gloss_criterion = JointGloss(dis_func=dis_func, mode=uloss_func)
  Eloss_criterion = JointEloss(dis_func=dis_func)
  Dloss_criterion = JointDloss(mode=uloss_func)


  # model
  embedder.train()
  recovery.train()
  generator.train()
  supervisor.train()
  discriminator.train()

  # optimizer
  # models_paramG = [generator.parameters(), supervisor.parameters()]
  # optimizerG = torch.optim.Adam(params=itertools.chain(*models_paramG), lr=learning_rate)

  optimizerG = torch.optim.Adam(
    [{'params': generator.parameters()},
     {'params': supervisor.parameters()}],
    lr=learning_rate3
  )

  # optimizerG = torch.optim.RMSprop(
  #   [{'params': generator.parameters()},
  #    {'params': supervisor.parameters()}],
  #    lr=learning_rate3, alpha=0.99, eps=1e-7, weight_decay=0, momentum=0, centered=False)

  # models_paramE = [embedder.parameters(), recovery.parameters()]
  # optimizerE = torch.optim.Adam(params=itertools.chain(*models_paramE), lr=learning_rate)

  optimizerE = torch.optim.Adam(
    [{'params': embedder.parameters()},
     {'params': recovery.parameters()}],
    lr=learning_rate4
  )

  optimizerD = torch.optim.Adam(params=discriminator.parameters(), lr=learning_rate5)

  # optimizerD = torch.optim.RMSprop(discriminator.parameters(), lr=learning_rate5, alpha=0.99, eps=1e-7, weight_decay=0, momentum=0, centered=False)


  for epoch in range(num_epochs):

    training_loss_G = 0.0
    training_loss_U = 0.0
    training_loss_S = 0.0
    training_loss_V = 0.0
    training_loss_E0 = 0.0
    training_loss_D = 0.0

    
    for inputs in data_loader:
      ## Generator training (twice more than discriminator training)
      for _ in range(2):
        X = inputs[0].to(CUDA_DEVICES)

        optimizerG.zero_grad()

        # Train generator
        # Generate random data
        z_batch_size, z_seq_len, z_dim = X.shape
        Z = random_generator(z_batch_size, z_seq_len, z_dim)
        Z = Z.to(CUDA_DEVICES)

        # Supervised Forward Pass
        H = embedder(X, None)
        # For attention
        # decoder_inputs = torch.zeros_like(X)
        # decoder_inputs = decoder_inputs.to(CUDA_DEVICES)
        # X_tilde = recovery(H, decoder_inputs)
        # For GRU
        H_hat_supervise = supervisor(H, None)
        X_tilde = recovery(H, None)

        # Generator Forward Pass
        E_hat = generator(Z, None)
        H_hat = supervisor(E_hat, None)
        
        # Synthetic data generated
        # For attention
        # decoder_inputs = torch.zeros_like(X)
        # decoder_inputs = decoder_inputs.to(CUDA_DEVICES)
        # X_hat = recovery(H_hat, decoder_inputs)
        # For GRU
        X_hat = recovery(H_hat, None)

        # Adversarial loss
        Y_fake = discriminator(H_hat, None)
        Y_fake_e = discriminator(E_hat, None)

        lossG, loss_U, loss_S, loss_V = Gloss_criterion(Y_fake, Y_fake_e, H[:,1:,:], H_hat_supervise[:,:-1,:], X, X_hat)

        lossG.backward()
        optimizerG.step()


        training_loss_G = np.sqrt(lossG.item())
        training_loss_U = np.sqrt(loss_U.item())
        training_loss_S = np.sqrt(loss_S.item())
        training_loss_V = np.sqrt(loss_V.item())

        # Train autoencoder
        optimizerE.zero_grad()

        H = embedder(X, None)
        # For attention
        # decoder_inputs = torch.zeros_like(X)
        # decoder_inputs = decoder_inputs.to(CUDA_DEVICES)
        # X_tilde = recovery(H, decoder_inputs)
        # For GRU
        X_tilde = recovery(H, None)

        H_hat_supervise = supervisor(H, None)

        lossE, lossE_0 = Eloss_criterion(X_tilde, X, H[:,1:,:], H_hat_supervise[:,:-1,:])

        lossE.backward()
        optimizerE.step()

        training_loss_E0 = np.sqrt(lossE_0.item())


    ## Discriminator training
    # with torch.backends.cudnn.flags(enabled=False):
      optimizerD.zero_grad()

      # Train discriminator

      # Generate random data
      z_batch_size, z_seq_len, z_dim = X.shape
      Z = random_generator(z_batch_size, z_seq_len, z_dim)
      Z = Z.to(CUDA_DEVICES)

      # latent code forward
      H = embedder(X, None).detach()
      H_hat = supervisor(E_hat, None).detach()
      E_hat = generator(Z, None).detach()
      
      # Forward Pass
      Y_real = discriminator(H, None)         # Encoded original data
      Y_fake = discriminator(H_hat, None)     # Output of supervisor
      Y_fake_e = discriminator(E_hat, None)   # Output of generator
      
      # Adversarial loss
      lossD = Dloss_criterion(Y_real, Y_fake, Y_fake_e)
      # lossD_gp = _gradient_penalty(CUDA_DEVICES, discriminator, H, H_hat)
      # lossD = lossD.add(lossD_gp)

      # Train discriminator (only when the discriminator does not work well)
      if lossD > 0.15:
        lossD.backward()
        optimizerD.step()
      
      training_loss_D = lossD.item()

    # Print multiple checkpoints
    if epoch % (np.round(num_epochs / 5)) == 0:
      print('step: '+ str(epoch) + '/' + str(num_epochs) +
            ', d_loss: ' + str(np.round(training_loss_D, 4)) +
            ', g_loss_u: ' + str(np.round(training_loss_U, 4)) +
            ', g_loss_s: ' + str(np.round(training_loss_S, 4)) +
            ', g_loss_v: ' + str(np.round(training_loss_V, 4)) +
            ', e_loss_t0: ' + str(np.round(training_loss_E0, 4)))

  print('Finish Joint Training')


if __name__ == '__main__':


  # save model path
  today = date.today()
  save_time = today.strftime("%d_%m_%Y")
  output_dir = config.get('train', 'model_path') + '/' + save_time + '/' + config.get('train', 'classification_dir') + '/'
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # Parameters
  print("CUDA DEVICE: {}".format(CUDA_DEVICES))
  print("[train] module: {}".format(module_name))
  print("[train] action: {}".format(config.get('train', 'classification_dir').split('_')[0]))
  print("[train] seq_len: {}".format(seq_len))
  print("[train] n_features: {}".format(n_features))
  print("[train] hidden size: {}".format(hidden_size))
  print("[train] num_layers: {}".format(num_layers))
  print("[train] num_epochs: {}".format(num_epochs))
  print("[train] batch_size: {}".format(batch_size))
  print("[train] distance function: {}".format(dis_func))
  print("[train] adversarial loss function: {}".format(uloss_func))

  # models
  embedder = Embedder(
    module = module_name,
    time_stamp = seq_len,
    input_size = n_features,
    hidden_dim = hidden_size,
    output_dim = hidden_size,
    num_layers = num_layers,
    activate_function=nn.Tanh()
  )

  recovery = Recovery(
    module = module_name,
    time_stamp = seq_len,
    input_size = hidden_size,
    hidden_dim = hidden_size,
    output_dim = n_features,
    num_layers = num_layers,
    activate_function=nn.Tanh()
  )

  generator = Generator(
    module = module_name,
    time_stamp = seq_len,
    input_size = n_features,
    hidden_dim = hidden_size,
    output_dim = hidden_size,
    num_layers = num_layers,
    activate_function=nn.Tanh()
  )

  supervisor = Supervisor(
    module = module_name,
    time_stamp = seq_len,
    input_size = hidden_size,
    hidden_dim = hidden_size,
    output_dim = hidden_size,
    # [Supervisor] num_layers must less(-1) than other component, embedder
    num_layers = num_layers - 1,
    activate_function=nn.Tanh()
  )

  discriminator = Discriminator(
    module = module_name,
    time_stamp = seq_len,
    input_size = hidden_size,
    hidden_dim = hidden_size,
    output_dim = 1,
    num_layers = num_layers,
  )

  embedder = embedder.to(CUDA_DEVICES)
  recovery = recovery.to(CUDA_DEVICES)
  generator = generator.to(CUDA_DEVICES)
  supervisor = supervisor.to(CUDA_DEVICES)
  discriminator = discriminator.to(CUDA_DEVICES)

  # Dataset
  Data_set = TimeSeriesDataset(root_dir=dataset_dir, transform=None, seq_len=seq_len)
  Data_loader = DataLoader(dataset=Data_set, batch_size=batch_size, shuffle=True, num_workers=1)

  train_stage1(Data_set, Data_loader, embedder, recovery)
  train_stage2(Data_set, Data_loader, embedder, supervisor, generator)
  train_stage3(Data_set, Data_loader, embedder, recovery, generator, supervisor, discriminator)

  torch.save(embedder, f'{output_dir+embedder_name}')
  torch.save(recovery, f'{output_dir+recovery_name}')
  torch.save(generator, f'{output_dir+generator_name}')
  torch.save(supervisor, f'{output_dir+supervisor_name}')
  torch.save(discriminator, f'{output_dir+discriminator_name}')
