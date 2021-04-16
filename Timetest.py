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
from tqdm import tqdm, trange
from Network.simple_discriminator import Simple_Discriminator
from Network.simple_predictor import Simple_Predictor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from Timedataset import TimeSeriesDataset
from utils import train_test_dataloader
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
config.read('Configure.ini', encoding="utf-8")

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
seq_len = config.getint('test', 'seq_len')
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

            real_outputs = model(real_inputs, real_time)
            fake_outputs = model(fake_inputs, fake_time)

            fake_label = torch.zeros_like(fake_outputs)
            real_label = torch.ones_like(real_outputs)

            outputs = torch.cat((real_outputs, fake_outputs), 0)

            labels = torch.cat((real_label, fake_label), 0)

            D_loss = critertion(outputs, labels)

            D_loss.backward()
            optimizer.step()

            training_loss = D_loss.item()


        loss_list.append(D_loss.item())

        logger.set_description(f"Epoch: {epoch}, Loss: {training_loss:.4f}")
    
    plt.plot(loss_list, color='red')
    plt.title("Training loss")
    plt.xlabel('Epoch')
    plt.savefig('./Loss_curve/loss_curve.png', bbox_inches='tight')

    # print("Finish Discriminator Training")

    model.eval()

    # print("Start Discriminator Testing")

    discriminative_score = 0.0
    synthetic_test_data_loader_iterator = iter(synthetic_test_data_loader)
    correct_results_sum = 0
    results_sum = 0

    with torch.no_grad():

        for i, (real_inputs, real_time) in enumerate(real_test_data_loader):

            fake_inputs, fake_time = next(synthetic_test_data_loader_iterator)

            if fake_inputs == None:
                break

            fake_inputs = fake_inputs.to(CUDA_DEVICES)
            fake_time = fake_time.to(CUDA_DEVICES)
            real_inputs = real_inputs.to(CUDA_DEVICES)
            real_time = real_time.to(CUDA_DEVICES)

            real_output = model(real_inputs, real_time)
            fake_output = model(fake_inputs, fake_time)

            fake_label = torch.zeros_like(fake_output)
            real_label = torch.ones_like(real_output)

            temp_outputs = torch.cat((fake_output, real_output), 0)

            # print("outputs: {}".format(outputs[-1]))

            outputs = torch.round(temp_outputs)

            # print("outputs: {}".format(outputs[-1]))

            labels = torch.cat((fake_label, real_label), 0)

            correct_results_sum += (labels == outputs).sum().item()

            results_sum += (labels.shape[0] * labels.shape[1])

    acc = np.round((correct_results_sum / results_sum), 4)
    discriminative_score = np.abs(0.5-acc)
    
    # print("Finish Discriminator Testing")

    return discriminative_score


def Predictive(model, synthetic_train_data_loader, real_test_data_loader):

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
            X = inputs[:, :-1, :]
            # print("X: {}".format(X.shape))

            # Y = inputs[0][:, 1:, (dim-1)].detach().clone()
            Y = inputs[:, 1:, -1].detach().clone()
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
    
    
    plt.plot(loss_list, color='blue')
    plt.savefig('./Loss_curve/loss_curve.png', bbox_inches='tight')
    plt.close()

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

            X = inputs[:, :-1, :]

            Y = inputs[:, 1:, -1]

            X = X.to(CUDA_DEVICES)
            Y = Y.to(CUDA_DEVICES)

            Xtime = torch.sub(real_time, 1)
            Xtime = Xtime.to(CUDA_DEVICES)

            Y_pred = model(X, Xtime)
            Y_pred = Y_pred.squeeze()

            sum_absolute_errors += torch.abs(
                torch.sum(torch.sub(Y_pred, Y))).item()

            sum_examples += Y.shape[0] * Y.shape[1]

        predictive_score = sum_absolute_errors / sum_examples

    # print("Finish Predictive Testing")

    return predictive_score


if __name__ == '__main__':

    real_data_set = TimeSeriesDataset(
        root_dir=real_dataset_dir, transform=None, seq_len=seq_len)

    print("real_data_set: {}".format(len(real_data_set)))

    synthetic_data_set = TimeSeriesDataset(
        root_dir=synthetic_dataset_dir, transform=None, seq_len=seq_len, mode='synthetic')

    print("synthetic_data_set: {}".format(len(synthetic_data_set)))

    max_seq_len = real_data_set.max_seq_len if real_data_set.max_seq_len > synthetic_data_set.max_seq_len else synthetic_data_set.max_seq_len

    print("Max sequence length: {}".format(max_seq_len))

    real_train_data_loader, real_test_data_loader = train_test_dataloader(
        data_set=real_data_set, mode='test')

    synthetic_train_data_loader, synthetic_test_data_loader = train_test_dataloader(
        data_set=synthetic_data_set, mode='test')

    predictive_train_loader = DataLoader(dataset=synthetic_data_set, batch_size=batch_size, shuffle=False, num_workers=1)
    
    predictive_test_loader = DataLoader(dataset=real_data_set, batch_size=batch_size, shuffle=False, num_workers=1)
    
    discriminative_score_list = []
    predictive_score_list = []

    for iteration in range(0, test_iteration):
        
        discriminator = Simple_Discriminator(
            time_stamp=max_seq_len, 
            input_size=27,
            # hidden_dim = input_size / 2
            hidden_dim=14,
            output_dim=1,
            num_layers=1,
            padding_value=PADDING_VALUE,
            max_seq_len=max_seq_len,
        )

        predictor = Simple_Predictor(
            time_stamp=max_seq_len-1,
            input_size=27,
            hidden_dim=14,
            output_dim=1,
            num_layers=1,
            padding_value=PADDING_VALUE,
            max_seq_len=max_seq_len-1
        )
        discriminative_score = Discriminative(
            discriminator, real_train_data_loader, real_test_data_loader, synthetic_train_data_loader, synthetic_test_data_loader)
        # predictive_train_loader, predictive_test_loader
        predictive_score = Predictive(
            predictor, real_train_data_loader, real_test_data_loader)
        
        # print("iteration: {}, predictive_score: {:.6f}".format(iteration, predictive_score))
        print("iteration: {}, discriminative_score: {:.6f}, predictive_score: {:.6f}".format(iteration, discriminative_score, predictive_score))

        discriminative_score_list.append(discriminative_score)
        predictive_score_list.append(predictive_score)
    
    mean_discriminative_score = np.mean(discriminative_score_list)
    mean_predictive_score = np.mean(predictive_score_list)

    print("Discriminative score: {:.4f}".format(mean_discriminative_score))
    print("Predictive score: {:.4f}".format(mean_predictive_score))
