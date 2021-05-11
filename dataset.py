# -*- coding: utf-8 -*-
from dataset_preprocess import path_preprocess, MinMaxScaler1, MinMaxScaler2
import torch
from torch.utils.data import Dataset
import configparser
import numpy as np

config = configparser.ConfigParser()
config.read('Configure.ini', encoding="utf-8")


class WaferDataset(Dataset):
    # root_dir:資料集路徑
    # mode:有'train'、'test'兩種，依據使用需要來選擇取用訓練資料集、測試資料集
    def __init__(self, root_dir, mode='train', transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.path = path_preprocess(root_dir)

    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):

        ori_data = np.loadtxt(self.path[index], delimiter = ",",skiprows = 0)
        # prerpocess the data
        temp_data = ori_data[::-1]
        data, min_val, max_val = MinMaxScaler1(temp_data)
        data_time = len(data)
        data = torch.FloatTensor(data)
        data_time = torch.LongTensor(data_time)
        if self.mode == 'train':
            return data, data_time
        else:
            return data, data_time, min_val, max_val
