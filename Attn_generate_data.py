import math
import torch
import torch.nn as nn
import configparser
import os
import gc
import pandas as pd
from datetime import date
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from Timedataset import TimeSeriesDataset
from Network.recovery import Recovery
from Network.supervisor import Supervisor
from utils import random_generator
from dataset_preprocess import MinMaxScaler1, batch_generation, extract_time, MinMaxScaler2, ReMinMaxScaler2, data_postprocess

config = configparser.ConfigParser()
config.read('Configure.ini', encoding="utf-8")

# gpu-used
CUDA_DEVICES = torch.device("cuda:"+config.get('default',
                                               'cuda_device_number') if torch.cuda.is_available() else "cpu")

# 模型參數路徑
PATH_TO_WEIGHTS = config.get('GenTstVis', 'model_path')

OUTPUT_DIR = config.get('GenTstVis', 'syntheticDataset_path')

dataset_dir = config.get('GenTstVis', 'Dataset_path')

classification_dir = config.get('GenTstVis', 'classification_dir')

date_dir = config.get('GenTstVis', 'date_dir')
seq_len = config.getint('GenTstVis', 'seq_len')
n_features = config.getint('GenTstVis', 'n_features')
hidden_size = config.getint('GenTstVis', 'hidden_size')
num_layers = config.getint('GenTstVis', 'num_layers')
PADDING_VALUE = config.getfloat('default', 'padding_value')

generator_name = config.get('generate_data', 'generator_name')
supervisor_name = config.get('generate_data', 'supervisor_name')
recovery_name = config.get('generate_data', 'recovery_name')

syntheitc_data_name = config.get('GenTstVis', 'synthetic_data_name')
module_name = config.get('default', 'module_name')

times_iteration = config.getint('generate_data', 'iteration')


def concat_data(data, data_list):
    # concat each batch data into a alist
    for i in range(0, len(data)):
        if len(data_list):
            data_list = np.concatenate((data_list, data[i]), axis=0)
        else:
            data_list = data[i]
    return data_list


def Save_Data(data, save_path, data_names):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_name = data_names
    output_path = os.path.join(save_path, file_name)
    temp_df = pd.DataFrame(data)
    temp_df.to_csv(output_path, index=False, header=False)

    return data_names


def Generate_data():

    # get synthetic data directory
    save_dir_path = OUTPUT_DIR + '/' + date_dir + '/' + classification_dir

    # get models' path
    model_path = PATH_TO_WEIGHTS + '/' + date_dir + '/' + classification_dir

    # load real data
    real_data = np.loadtxt(dataset_dir, delimiter=",", skiprows=0)

    # get real data min(max) value
    real_data, min_val1, max_val1 = MinMaxScaler1(real_data)
    batch_real_data = batch_generation(real_data, seq_len, 1)
    ori_time, _ = extract_time(batch_real_data)
    _, min_val2, max_val2 = MinMaxScaler2(batch_real_data)

    # To get same amount of data
    data_seq_len, dim = np.asarray(real_data).shape

    no = len(batch_real_data)
    
    # release variable memory
    del _
    del real_data
    del batch_real_data
    gc.collect()
    # _, real_data, batch_real_data = None, None, None

    # load model
    generator = Recovery(
        module=module_name,
        mode='noise',
        time_stamp=seq_len,
        input_size=n_features,
        hidden_dim=hidden_size,
        output_dim=hidden_size,
        num_layers=num_layers,
        activate_function=nn.Tanh(),
        padding_value=PADDING_VALUE,
        max_seq_len=seq_len
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
        max_seq_len=seq_len
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
        max_seq_len=seq_len
    )
    generator.load_state_dict(torch.load(model_path + '/' + generator_name))
    supervisor.load_state_dict(torch.load(model_path + '/' + supervisor_name))
    recovery.load_state_dict(torch.load(model_path + '/' + recovery_name))
    generator.eval()
    supervisor.eval()
    recovery.eval()

    # move to GPU
    generator = generator.to(CUDA_DEVICES)
    supervisor = supervisor.to(CUDA_DEVICES)
    recovery = recovery.to(CUDA_DEVICES)


    data_names = 1
    generated_data = list()
    generated_data_list = list()

    # times_iteration: How many times of the data's amount we want
    for _ in range(0, times_iteration):

        for i in range(no):

            with torch.no_grad():
                
                with torch.cuda.amp.autocast():
                    # generate noize
                    Z = random_generator(1, seq_len, dim, ori_time[i])
                    Z = Z.to(CUDA_DEVICES)
                    T = torch.as_tensor(ori_time[i], dtype=torch.int).to(CUDA_DEVICES)

                    # generate synthetic data
                    E_hat = generator(Z, T)
                    H_hat = supervisor(E_hat, T)
                    X_hat = recovery(H_hat, T)

            # X_hat = X_hat.cpu().detach().numpy()
            temp = X_hat.cpu().detach().numpy().squeeze()
            generated_data.append(temp)

    generated_data = generated_data * max_val2
    generated_data = generated_data + min_val2

    # Make all batch data into a list
    generated_data_list = concat_data(generated_data, generated_data_list)

    # release variable memory
    del generated_data
    gc.collect()

    # Renormalized the synthetic normalized data
    generated_data_list = data_postprocess(
        generated_data_list, min_val1, max_val1)

    # Save the data
    data_names = Save_Data(
        generated_data_list, save_dir_path, syntheitc_data_name)


if __name__ == '__main__':
    Generate_data()
