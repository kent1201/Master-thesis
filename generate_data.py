import math
import torch
import torch.nn as nn
import configparser
import os
import pandas as pd
from datetime import date
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from dataset import WaferDataset
from utils import random_generator
from dataset_preprocess import ReMinMaxScaler1

config = configparser.ConfigParser()
config.read('Configure.ini', encoding="utf-8")

# gpu-used
CUDA_DEVICES = torch.device("cuda:"+config.get('default', 'cuda_device_number') if torch.cuda.is_available() else "cpu")

# 模型參數路徑
PATH_TO_WEIGHTS = config.get('GenTstVis', 'model_path')

OUTPUT_DIR = config.get('GenTstVis', 'syntheticDataset_path')

dataset_dir = config.get('GenTstVis', 'Dataset_path')

classification_dir = config.get('GenTstVis', 'classification_dir')

date_dir = config.get('GenTstVis', 'date_dir')

generator_name = config.get('generate_data','generator_name')
supervisor_name = config.get('generate_data','supervisor_name')
recovery_name = config.get('generate_data','recovery_name')

iterations = config.get('generate_data', 'iteration')


def Save_Data(data ,data_names):

  output_dir_real = OUTPUT_DIR + '/' + date_dir + '/' + classification_dir
  # print("saved_data_dir_: {}".format(output_dir_real))

  if not os.path.exists(output_dir_real):
    os.makedirs(output_dir_real)

  for i in range(0, data.size(0)):
    temp_data = data[i]
    # print("temp_data: {}".format(temp_data.shape))
    temp_data = temp_data.data.cpu().numpy()
    final_data = temp_data[::-1]
    # print(final_data.shape)
    # Save data to csv
    file_name = str(data_names) + '.csv'
    output_path = os.path.join(output_dir_real, file_name)
    temp_df = pd.DataFrame(final_data)
    temp_df.to_csv(output_path, index=False, header=False)
    data_names+=1

  return data_names




def Generate_data():

  data_set = WaferDataset(root_dir=dataset_dir, mode='test', transform=None)
  data_loader = DataLoader(dataset=data_set, batch_size=config.getint('generate_data', 'batch_size'), shuffle=True, num_workers=1)

  model_path = PATH_TO_WEIGHTS + '/' + date_dir + '/' + classification_dir

  generator = torch.load(model_path + '/' + generator_name)
  supervisor = torch.load(model_path +'/' + supervisor_name)
  recovery = torch.load(model_path + '/' + recovery_name)

  generator = generator.cuda(CUDA_DEVICES)
  supervisor = supervisor.cuda(CUDA_DEVICES)
  recovery = recovery.cuda(CUDA_DEVICES)

  generator.eval()
  supervisor.eval()
  recovery.eval()

  data_names = 1

  for iteration in range(0, iterations):

    data_loader = DataLoader(dataset=data_set, batch_size=config.getint('generate_data', 'batch_size'), shuffle=True, num_workers=1)

    for i, inputs in enumerate(data_loader):

      X, T, min_val, max_val = inputs[0], inputs[1], inputs[2], inputs[3]

      z_batch_size, z_seq_len, z_dim = X.shape
      Z = random_generator(z_batch_size, z_seq_len, z_dim)
      Z = Z.to(CUDA_DEVICES)
      X = X.to(CUDA_DEVICES)

      min_val = min_val.to(CUDA_DEVICES)
      max_val = max_val.to(CUDA_DEVICES)

      E_hat = generator(Z, T)
      H_hat = supervisor(E_hat, T)
      X_hat = recovery(H_hat, T)

      X_hat = ReMinMaxScaler1(X_hat, min_val, max_val)

      data_names = Save_Data(X_hat, data_names)

      # print("i: {}, X_hat: {}".format(i, X_hat[0].data))



if __name__ == '__main__':
  Generate_data()