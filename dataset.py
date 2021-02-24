# -*- coding: utf-8 -*-
from dataset_preprocess import path_preprocess, MinMaxScaler1, MinMaxScaler2
import torch
from torch.utils.data import Dataset
import configparser
import numpy as np

config = configparser.ConfigParser()
config.read('Configure.ini', encoding="utf-8")


class SensorSignalDataset(Dataset):
    # root_dir:資料集路徑
    # mode:有'train'、'test'兩種，依據使用需要來選擇取用訓練資料集、測試資料集
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform

        self.path = path_preprocess(root_dir)

    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):

        ori_data = np.loadtxt(self.path[index], delimiter = ",",skiprows = 0)
        ori_data = np.around(ori_data, decimals=4)

        ori_data = ori_data[::-1]

        ori_data, min_val1, max_val1 = MinMaxScaler1(ori_data)
        data, min_val2, max_val2 = MinMaxScaler2(ori_data)

        data = torch.FloatTensor(data)
        data = np.around(data, decimals=4)

        return data, min_val1, max_val1, min_val2, max_val2
