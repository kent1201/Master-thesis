import math
import configparser
import os
import pandas as pd
from datetime import date
from Timedataset import TimeSeriesDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torchvision import transforms
import numpy as np
from tqdm import tqdm, trange
from Network.simple_discriminator import Simple_Discriminator
from Network.simple_predictor import Simple_Predictor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

from utils import train_test_divide
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
config.read('Configure.ini', encoding="utf-8")

# 固定演算法進行加速
# torch.backends.cudnn.benchmark = True

# gpu-used
CUDA_DEVICES = torch.device("cuda:"+config.get('default',
                                               'cuda_device_number') if torch.cuda.is_available() else "cpu")

synthetic_dataset_dir = config.get('GenTstVis', 'syntheticDataset_path') + '/' + config.get('GenTstVis', 'date_dir') + \
    '/' + config.get('GenTstVis', 'classification_dir') + '/' + \
    config.get('GenTstVis', 'synthetic_data_name')

real_dataset_dir = config.get('GenTstVis', 'Dataset_path')


d_num_epochs = config.getint('test', 'd_num_epochs')
p_num_epochs = config.getint('test', 'p_num_epochs')
batch_size = config.getint('test', 'batch_size')
learning_rate = config.getfloat('test', 'learning_rate')
seq_len = config.getint('GenTstVis', 'seq_len')
n_features = config.getint('GenTstVis', 'n_features')
PADDING_VALUE = config.getfloat('default', 'padding_value')
test_iteration = config.getint('test', 'test_iteration')

dis_curve = 0
pred_curve = 0


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def Discriminative(model, real_train_data_loader, real_test_data_loader, synthetic_train_data_loader, synthetic_test_data_loader):

    # model
    model = model.to(CUDA_DEVICES)
    model.train()

    # loss
    critertion = nn.BCEWithLogitsLoss()

    # optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    # print("Start Discriminator Training")

    logger = trange(d_num_epochs, desc=f"Epoch: 0, Loss: 0")

    loss_list = []

    for epoch in logger:

        training_loss = 0.0

        synthetic_train_data_loader_iterator = iter(
            synthetic_train_data_loader)

        for _, (real_inputs, real_time) in enumerate(real_train_data_loader):

            optimizer.zero_grad()

            fake_inputs, fake_time = next(synthetic_train_data_loader_iterator)

            if fake_inputs == None:
                break

            fake_inputs = fake_inputs.to(CUDA_DEVICES)
            fake_time = fake_time.to(CUDA_DEVICES)
            real_inputs = real_inputs.to(CUDA_DEVICES)
            real_time = real_time.to(CUDA_DEVICES)

            real_logit_output, real_output = model(real_inputs, real_time)
            fake_logit_output, fake_output = model(fake_inputs, fake_time)

            fake_label = torch.zeros_like(fake_logit_output)
            real_label = torch.ones_like(real_logit_output)

            d_loss_real = critertion(real_logit_output, real_label)
            d_loss_fake = critertion(fake_logit_output, fake_label)

            D_loss = d_loss_real + d_loss_fake

            D_loss.backward()
            optimizer.step()

            training_loss = D_loss.item()


        loss_list.append(D_loss.item())

        logger.set_description(f"Epoch: {epoch}, Loss: {training_loss:.4f}")

    plt.plot(loss_list, color='red', label='Discriminative')
    plt.title("Score loss")
    plt.xlabel('Epoch')
    plt.savefig('./Loss_curve/Score_loss_curve.png', bbox_inches='tight')

    # print("Finish Discriminator Training")

    model.eval()

    # print("Start Discriminator Testing")

    discriminative_score = 0.0
    synthetic_test_data_loader_iterator = iter(synthetic_test_data_loader)
    correct_results_sum = 0
    results_sum = 0
    y_output_final = []
    y_label_final = []

    with torch.no_grad():

        for i, (real_inputs, real_time) in enumerate(real_test_data_loader):

            fake_inputs, fake_time = next(synthetic_test_data_loader_iterator)

            if fake_inputs == None:
                break

            fake_inputs = fake_inputs.to(CUDA_DEVICES)
            fake_time = fake_time.to(CUDA_DEVICES)
            real_inputs = real_inputs.to(CUDA_DEVICES)
            real_time = real_time.to(CUDA_DEVICES)

            real_logit_output, real_output = model(real_inputs, real_time)
            fake_logit_output, fake_output = model(fake_inputs, fake_time)

            fake_label = torch.zeros_like(fake_output)
            real_label = torch.ones_like(real_output)

            temp_outputs = torch.cat((fake_output, real_output), 0)

            outputs = torch.round(temp_outputs).detach().cpu().numpy()

            # print("outputs: {}".format(outputs.shape))

            labels = torch.cat((fake_label, real_label), 0).detach().cpu().numpy()

            # print("labels: {}".format(labels.shape))

            if i == 0:
                y_output_final = np.squeeze(outputs)
                y_label_final = np.squeeze(labels)
            else:
                y_output_final = np.concatenate((y_output_final, np.squeeze(outputs)), axis = 0)
                y_label_final = np.concatenate((y_label_final, np.squeeze(labels)), axis = 0)

            # print("final labels: {}".format(y_label_final.shape))

            # print("final output: {}".format(y_output_final.shape))

            # correct_results_sum += (labels == outputs).sum().item()
            # results_sum += (labels.shape[0] * labels.shape[1])

    # acc = np.round((correct_results_sum / results_sum), 4)
    # (y_output_final>0.5) make (1, 0) -> (True, False)
    acc = accuracy_score(y_label_final, (y_output_final>0.5))
    discriminative_score = np.abs(0.5-acc)
    # print("discriminative_score: {}".format(discriminative_score))

    # print("Finish Discriminator Testing")

    return discriminative_score


def Predictive(model, synthetic_train_data_loader, real_test_data_loader, mode):

    # model
    model = model.to(CUDA_DEVICES)
    model.train()

    # loss
    critertion = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    # print("Start Predictive Training")

    loss_list = []

    logger = trange(p_num_epochs, desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:

        training_loss = 0.0

        num_examples = 0

        for i, (inputs, data_time) in enumerate(synthetic_train_data_loader):

            optimizer.zero_grad()

            bat, seq, dim = inputs.shape

            # X = inputs[0][:, :-1, :(dim-1)]
            X = inputs[:, :-1, :(dim-1)]
            # print("X: {}".format(X.shape))

            # Y = inputs[0][:, 1:, (dim-1)].detach().clone()
            Y = inputs[:, 1:, (dim-1)].detach().clone()
            # print("Y: {}".format(Y.shape))

            X = X.to(CUDA_DEVICES)
            Y = Y.to(CUDA_DEVICES)

            Xtime = torch.sub(data_time, 1)
            Xtime = Xtime.to(CUDA_DEVICES)

            y_pred = model(X, Xtime)
            y_pred = y_pred.squeeze()
            # print("y_pred: {}".format(y_pred.shape))

            loss = critertion(y_pred, Y)

            loss.backward()
            optimizer.step()

            training_loss = np.sqrt(loss.item())

        loss_list.append(loss.item())

        logger.set_description(f"Epoch: {epoch}, Loss: {training_loss:.4f}")

    if mode == 'TRTR':
        plt.plot(loss_list, color='blue', label='TRTR')
    elif mode == 'TSTR':
        plt.plot(loss_list, color='yellow', label='TSTR')
    elif mode == 'TMTR':
        plt.plot(loss_list, color='orange', label='TRTR')
    elif mode == 'TRTS':
        plt.plot(loss_list, color='black', label='TRTS')

    # print("Finish Predictive Training")

    model.eval()

    # print("Start Predictive Testing")

    # Compute the performance in terms of MAE
    with torch.no_grad():

        predictive_score = 0.0
        sum_absolute_errors = 0
        sum_examples = 0

        for i, (inputs, real_time) in enumerate(real_test_data_loader):

            bat, seq, dim = inputs.shape

            X = inputs[:, :-1, :(dim-1)]

            Y = inputs[:, 1:, (dim-1)]

            X = X.to(CUDA_DEVICES)
            Y = Y.to(CUDA_DEVICES)

            Xtime = torch.sub(real_time, 1)
            Xtime = Xtime.to(CUDA_DEVICES)

            Y_pred = model(X, Xtime)
            Y_pred = Y_pred.squeeze()

            sum_absolute_errors += torch.sum(
                torch.abs(torch.sub(Y_pred, Y))).item()

            sum_examples += Y.shape[0] * Y.shape[1]

        predictive_score = sum_absolute_errors / sum_examples

    # print("Finish Predictive Testing")

    return predictive_score


if __name__ == '__main__':

    real_dataset = TimeSeriesDataset(
        root_dir=real_dataset_dir, seq_len=seq_len)

    print("real_data_set: {}".format(len(real_dataset)))

    synthetic_dataset = TimeSeriesDataset(
        root_dir=synthetic_dataset_dir, seq_len=seq_len, mode='synthetic')

    print("synthetic_data_set: {}".format(len(synthetic_dataset)))

    real_train_dataset, real_test_dataset = train_test_divide(
        data_set=real_dataset, mode='test')

    synthetic_train_dataset, synthetic_test_dataset = train_test_divide(
        data_set=synthetic_dataset, mode='test')

    mix_dataset = ConcatDataset([real_train_dataset, synthetic_train_dataset])
    print("mix_dataset: {}".format(len(mix_dataset)))

    max_seq_len = real_data_set.max_seq_len if real_dataset.max_seq_len > synthetic_dataset.max_seq_len else synthetic_dataset.max_seq_len

    print("Max sequence length: {}".format(max_seq_len))

    # TRTR & discriminative
    real_data_loader = DataLoader(dataset=real_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    real_train_data_loader = DataLoader(dataset=real_train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    real_test_data_loader = DataLoader(dataset=real_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    synthetic_data_loader = DataLoader(dataset=synthetic_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    synthetic_train_data_loader = DataLoader(dataset=synthetic_train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    synthetic_test_data_loader = DataLoader(dataset=synthetic_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # TMTR
    mix_data_loader = DataLoader(dataset=mix_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    

    discriminative_score_list = []
    TRTR_score_list = []
    TRTS_score_list = []
    TSTR_score_list = []
    TMTR_score_list = []
    discriminative_score = 0.0
    TSTR_predictive_score = 0.0
    TRTS_predictive_score = 0.0
    TMTR_predictive_score = 0.0
    TRTR_predictive_score = 0.0


    for iteration in range(0, test_iteration):

        ## Discriminative score
        #============================================================================================================================#
        discriminator = Simple_Discriminator(
            time_stamp=max_seq_len,
            input_size=n_features,
            # hidden_dim = input_size / 2
            hidden_dim=(n_features // 2),
            output_dim=1,
            num_layers=2,
            padding_value=PADDING_VALUE,
            max_seq_len=max_seq_len,
        )

        discriminative_score = Discriminative(
            discriminator, real_train_data_loader, real_test_data_loader, synthetic_train_data_loader, synthetic_test_data_loader)

        # ## TSTR
        #============================================================================================================================#
        predictor = Simple_Predictor(
            time_stamp=max_seq_len-1,
            input_size=n_features-1,
            hidden_dim=(n_features // 2),
            output_dim=1,
            num_layers=5,
            padding_value=PADDING_VALUE,
            max_seq_len=max_seq_len-1
        )

        # predictive_train_loader, real_train_data_loader
        TSTR_predictive_score = Predictive(
            predictor, synthetic_data_loader, real_test_data_loader, mode='TSTR')

        ## TMTR
        #============================================================================================================================#
        predictor = Simple_Predictor(
            time_stamp=max_seq_len-1,
            input_size=n_features-1,
            hidden_dim=(n_features // 2),
            output_dim=1,
            num_layers=5,
            padding_value=PADDING_VALUE,
            max_seq_len=max_seq_len-1
        )

        TMTR_predictive_score = Predictive(
            predictor, mix_data_loader, real_test_data_loader, mode='TMTR')

        ## TRTR
        #============================================================================================================================#
        # predictor = Simple_Predictor(
        #     time_stamp=max_seq_len-1,
        #     input_size=n_features-1,
        #     hidden_dim=(n_features // 2),
        #     output_dim=1,
        #     num_layers=5,
        #     padding_value=PADDING_VALUE,
        #     max_seq_len=max_seq_len-1
        # )

        # TRTR_predictive_score = Predictive(
        #     predictor, real_train_data_loader, real_test_data_loader, mode='TRTR')

        ## TRTS
        #============================================================================================================================#
        predictor = Simple_Predictor(
            time_stamp=max_seq_len-1,
            input_size=n_features-1,
            hidden_dim=(n_features // 2),
            output_dim=1,
            num_layers=5,
            padding_value=PADDING_VALUE,
            max_seq_len=max_seq_len-1
        )

        TRTS_predictive_score = Predictive(
            predictor, real_data_loader, synthetic_test_data_loader, mode='TRTS')

        # print("iteration: {}, discriminative_score: {:.6f}, TRTS_score: {:.6f}, TSTR_score: {:.6f}, TMTR_score: {:.6f}, TRTR_score: {:.6f}".format(iteration, discriminative_score, TRTS_predictive_score, TSTR_predictive_score, TMTR_predictive_score, TRTR_predictive_score))
        print("iteration: {}, discriminative_score: {:.6f}, TRTS_score: {:.6f}, TSTR_score: {:.6f}, TMTR_score: {:.6f}".format(iteration, discriminative_score, TRTS_predictive_score, TSTR_predictive_score, TMTR_predictive_score))
        # print("iteration: {}, TSTR_score: {:.6f}, TMTR_score: {:.6f}".format(iteration, TSTR_predictive_score, TMTR_predictive_score))
        # print("iteration: {}, TSTR_score: {:.6f}, TMTR_score: {:.6f}, TRTR_score: {:.6f}, TRTS_score: {:.6f}".format(iteration, TSTR_predictive_score, TMTR_predictive_score, TRTR_predictive_score, TRTS_predictive_score))
        # print("iteration: {}, discriminative_score: {:.6f}".format(iteration, discriminative_score))

        plt.savefig('./Loss_curve/Score_loss_curve.png', bbox_inches='tight')
        plt.legend()
        plt.close()

        discriminative_score_list.append(discriminative_score)
        TSTR_score_list.append(TSTR_predictive_score)
        TMTR_score_list.append(TMTR_predictive_score)
        # TRTR_score_list.append(TRTR_predictive_score)
        TRTS_score_list.append(TRTS_predictive_score)

    mean_discriminative_score = np.mean(discriminative_score_list)
    mean_TSTR_score = np.mean(TSTR_score_list)
    mean_TMTR_score = np.mean(TMTR_score_list)
    # mean_TRTR_score = np.mean(TRTR_score_list)
    mean_TRTS_score = np.mean(TRTS_score_list)

    print("Discriminative score: {:.4f}".format(mean_discriminative_score))
    print("TRTS predictive score: {:.4f}".format(mean_TRTS_score))
    print("TSTR predictive score: {:.4f}".format(mean_TSTR_score))
    print("TMTR predictive score: {:.4f}".format(mean_TMTR_score))
    # print("TRTR predictive score: {:.4f}".format(mean_TRTR_score))
    
