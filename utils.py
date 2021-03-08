import torch
import torch.nn
from dataset import SensorSignalDataset
from torch.utils.data import DataLoader
import configparser
import random
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import numpy as np


config = configparser.ConfigParser()
config.read('Configure.ini', encoding="utf-8")

# manualSeed = config.getint('default', 'manualSeed')

manualSeed = random.randint(1, 10000)  # use if you want new results
# print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def random_generator(batch_size, seq_len, dim):

    # Z = torch.zeros(batch_size, seq_len, dim)

    # for i in range(batch_size):
    #     temp = torch.rand(seq_len, dim)
    #     temp = torch.add(torch.mul(temp, 2.0), -1.0)
    #     Z[i, :, :] = temp.detach().clone()

    Z = torch.rand(batch_size, seq_len, dim)
    # Z = torch.add(torch.mul(temp, 2.0), -1.0)
    return Z


def train_test_dataloader(data_set, mode='test'):

    train_dataset_size = int(config.getfloat(
        mode, 'trainset_percentage') * len(data_set))
    test_dataset_size = len(data_set) - train_dataset_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        data_set, [train_dataset_size, test_dataset_size])

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=config.getint(
        mode, 'batch_size'), shuffle=False, num_workers=1)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=config.getint(
        mode, 'batch_size'), shuffle=False, num_workers=1)

    return train_data_loader, test_data_loader


def _gradient_penalty(CUDA_DEVICES, discriminator, real_data, generated_data, gp_weight=10):

    # Tensor = torch.cuda.FloatTensor if CUDA_DEVICES else torch.FloatTensor
    alpha = torch.rand(real_data.size(0), 1, 1).to(CUDA_DEVICES)
    # alpha = Tensor(np.random.random(
    #     (real_data.size(0), 1, 1))).to(CUDA_DEVICES)
    interpolates = (alpha * real_data + ((1 - alpha) *
                                         generated_data)).requires_grad_(True)
    d_interpolates = discriminator(interpolates, None)
    # fake = Variable(Tensor(d_interpolates.size()).fill_(
    # 1.0).to(CUDA_DEVICES), requires_grad=False)
    fake = Variable(torch.ones_like(d_interpolates).to(
        CUDA_DEVICES), requires_grad=False)
    gradients = torch_grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    return gradient_penalty


if __name__ == '__main__':

    # for i in range(0, 10):
    #     Z = random_generator(128, 24, 6)
    #     print("Z: {}".format(Z[0]))

    real_dataset_dir = config.get(
        'default', 'Dataset_path') + '/' + config.get('default', 'classification_dir')

    train_data_loader, test_data_loader = train_test_dataloader(
        real_dataset_dir, 0.75)

    for i, inputs in enumerate(train_data_loader):

        X = inputs[0]

        print("[utils.py] i: {}, data loader: {}".format(i, X.shape))
