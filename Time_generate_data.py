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
from Timedataset import TimeSeriesDataset
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
seq_len = config.getint('train', 'seq_len')

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

    # load model
    generator = torch.load(model_path + '/' + generator_name)
    supervisor = torch.load(model_path + '/' + supervisor_name)
    recovery = torch.load(model_path + '/' + recovery_name)

    # move to GPU
    generator = generator.cuda(CUDA_DEVICES)
    supervisor = supervisor.cuda(CUDA_DEVICES)
    recovery = recovery.cuda(CUDA_DEVICES)

    generator.eval()
    supervisor.eval()
    recovery.eval()

    data_names = 1
    generated_data = list()
    generated_data_list = list()

    # times_iteration: How many times of the data's amount we want
    for _ in range(0, times_iteration):

        with torch.no_grad():

            # generate noize
            Z = random_generator(no, seq_len, dim, ori_time)
            Z = Z.to(CUDA_DEVICES)

            # generate synthetic data
            E_hat = generator(Z, ori_time)
            H_hat = supervisor(E_hat, ori_time)
            # For attention
            if module_name == "self-attn":
                decoder_inputs = torch.zeros_like(Z)
                decoder_inputs = decoder_inputs.to(CUDA_DEVICES)
                X_hat = recovery(H_hat, decoder_inputs)
            # For GRU
            else:
                X_hat = recovery(H_hat, ori_time)

        X_hat = X_hat.cpu().detach().numpy()

        for i in range(no):
            temp = X_hat[i, :ori_time[i], :]
            generated_data.append(temp)
    
    generated_data = generated_data * max_val2
    generated_data = generated_data + min_val2

    # Make all batch data into a list
    generated_data_list = concat_data(generated_data, generated_data_list)

    # Renormalized the synthetic normalized data
    generated_data_list = data_postprocess(
        generated_data_list, min_val1, max_val1)

    # Save the data
    data_names = Save_Data(
        generated_data_list, save_dir_path, syntheitc_data_name)


if __name__ == '__main__':
    Generate_data()
