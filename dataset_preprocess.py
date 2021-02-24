import os
import numpy as np
import math
import configparser

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
  norm_data = 2 * norm_data - 1
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
    min_val = np.min(np.min(data, axis = 0), axis = 0)
    data = data - min_val

    max_val = np.max(np.max(data, axis = 0), axis = 0)
    norm_data = data / (max_val + 1e-7)
    #[test] min-max to (-1, 1)
    norm_data = 2 * norm_data - 1

    return norm_data, min_val, max_val

def path_preprocess(path):

    files_path = []
    file_names = []
    files = os.listdir(path)
    for f in files:
        file_name = f.rstrip('.csv')
        file_names.append(file_name)

    rndseq = np.random.RandomState(config.getint('default', 'random_state')).permutation(file_names)

    if len(files) != 0:
        for j in range(len(files)):
            file_path = path + '/' + str(rndseq[j]) + '.csv'
            files_path.append(file_path)
            

    return files_path


def data_preprocess(data, seq_len):
    
    no = len(data)
    dim = data.shape[-1]
    output = []
    for i in range(0, no-seq_len):
        if i+seq_len < no:
            output.append(data[i:i+seq_len, :])
    return output



