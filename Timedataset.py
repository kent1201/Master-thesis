# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import configparser
import numpy as np
from dataset_preprocess import data_preprocess

config = configparser.ConfigParser()
config.read('Configure.ini', encoding="utf-8")


class TimeSeriesDataset(Dataset):
    # root_dir:資料集路徑
    # mode:有'real'、'synthetic'兩種，依據使用需要來選擇取用訓練資料集、測試資料集
    def __init__(self, root_dir, seq_len, mode='real', transform=None):

        self.root_dir = root_dir
        self.transform = transform
        # load data
        ori_data = np.loadtxt(self.root_dir, delimiter=",", skiprows=0)
        # prerpocess the data
        if mode == 'real':
            data_set, data_set_time, max_seq_len = data_preprocess(
                ori_data, seq_len, 1)
        elif mode == 'synthetic':
            data_set, data_set_time, max_seq_len = data_preprocess(
                ori_data, seq_len, seq_len)
        self.dataset = torch.FloatTensor(data_set)
        self.dataset_time = torch.LongTensor(data_set_time)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # data = np.around(data, decimals=4)
        data = self.dataset[index]
        data_time = self.dataset_time[index]

        return data, data_time

    # def collate_fn(self, batch):
    #     """Minibatch sampling
    #     """
    #     # Pad sequences to max length
    #     X_mb = [X for X in batch[0]]

    #     # The actual length of each data
    #     T_mb = [T for T in batch[1]]
    #     return X_mb, T_mb
