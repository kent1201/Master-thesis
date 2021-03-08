# -*- coding: utf-8 -*-
from dataset_preprocess import batch_generation, data_preprocess
import torch
from torch.utils.data import Dataset
import configparser
import numpy as np

config = configparser.ConfigParser()
config.read('Configure.ini', encoding="utf-8")


class TimeSeriesDataset(Dataset):
    # root_dir:資料集路徑
    # mode:有'train'、'test'兩種，依據使用需要來選擇取用訓練資料集、測試資料集
    def __init__(self, root_dir, transform=None, seq_len=24):

        self.root_dir = root_dir
        self.transform = transform
        self.seq_len = seq_len
        # load data
        self.ori_data = np.loadtxt(self.root_dir, delimiter=",", skiprows=0)
        # Normalize the data
        self.dataset, _, _, _, _ = data_preprocess(
            self.ori_data, self.seq_len)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        temp_data = self.dataset[index]
        data = torch.FloatTensor(temp_data)
        # data = np.around(data, decimals=4)

        return data
