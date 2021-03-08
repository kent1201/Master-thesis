import os
import numpy as np
import math
import configparser
import torch

config = configparser.ConfigParser()
config.read('Configure.ini', encoding="utf-8")


def StandardScaler(data):
    """Min Max normalizer.
    do for each column
    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    m = data.mean(0, keepdim=True)
    s = data.std(0, unbiased=False, keepdim=True)
    data = data - m
    # epsilon = 1e-7 to avoid loss=nan
    norm_data = data / (s + 1e-7)
    return norm_data


def MinMaxScaler1(data):
    """Min Max normalizer.
    do for each column
    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    min_val = np.min(data, 0)
    max_val = np.max(data, 0)
    numerator = data - min_val
    denominator = max_val - min_val
    norm_data = numerator / (denominator + 1e-7)
    # rescale to (-1, 1)
    # norm_data = 2 * norm_data - 1
    return norm_data, min_val, max_val


def MinMaxScaler2(data):
    """Min-Max Normalizer.

    Args:
      - data: raw data

    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val

    max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)
    # [test] min-max to (-1, 1)
    # norm_data = 2 * norm_data - 1

    return norm_data, min_val, max_val


def ReMinMaxScaler2(data, min_val, max_val):
    """Min-Max Normalizer.

    Args:
      - data: raw data

    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """
    # print("[generate_data.py] min_val2: {}, max_val2: {}".format(min_val.shape, max_val.shape))
    # Pytorch version
    # min_val = min_val.unsqueeze(1).unsqueeze(2)
    # max_val = max_val.unsqueeze(1).unsqueeze(2)
    # data = torch.mul(torch.add(data, 1), 0.5)
    # data = torch.mul(data, torch.add(max_val, 1e-7))
    # re_data = torch.add(data, min_val)

    # Numpy version
    # temp_data = (data+1) / 2
    temp_data = data * (max_val + 1e-7)
    re_data = temp_data + min_val

    return re_data


def ReMinMaxScaler1(data, min_val, max_val):
    """Min Max normalizer.
      do for each column
      Args:
      -  data: original data
      Returns:
      - norm_data: normalized data
    """
    # Pytorch version
    # min_val = min_val.unsqueeze(1)
    # max_val = max_val.unsqueeze(1)
    # data = torch.mul(torch.add(data, 1), 0.5)
    # data = torch.mul(data, torch.sub(max_val, min_val))
    # re_data = torch.add(data, min_val)

    # Numpy version
    # temp_data = (data+1) / 2
    temp_data = data * (max_val - min_val + 1e-7)
    re_data = temp_data + min_val

    return re_data


def path_preprocess(path):

    files_path = []
    file_names = []
    files = os.listdir(path)
    for f in files:
        file_name = f.rstrip('.csv')
        file_names.append(file_name)

    rndseq = np.random.RandomState(config.getint(
        'default', 'random_state')).permutation(file_names)

    if len(files) != 0:
        for j in range(len(files)):
            file_path = path + '/' + str(rndseq[j]) + '.csv'
            files_path.append(file_path)

    return files_path


def data_preprocess(data, seq_len):

    temp_data = data[::-1]
    temp_data, min_val1, max_val1 = MinMaxScaler1(temp_data)
    # Convert the normalized data into [batches, seq_len, dimension] with window slicing
    batch_temp_data = batch_generation(temp_data, seq_len)
    output_data, min_val2, max_val2 = MinMaxScaler2(batch_temp_data)

    return output_data, min_val1, max_val1, min_val2, max_val2


def data_postprocess(data, min_val1, max_val1):

    temp_data = ReMinMaxScaler1(data, min_val1, max_val1)
    output_data = temp_data[::-1]
    return output_data


def batch_generation(data, seq_len):

    no = len(data)
    dim = data.shape[-1]
    output = []
    for i in range(0, no-seq_len):
        if i+seq_len < no:
            output.append(data[i:i+seq_len, :])
    return output
