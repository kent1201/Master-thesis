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
random.seed(68)
torch.manual_seed(manualSeed)


def random_generator(batch_size, max_seq_len, z_dim, T_mb):
    """Random vector generation.

    Args:
    - batch_size: size of the random vector
    - z_dim: dimension of random vector
    - T_mb: time information for the random vector
    - max_seq_len: maximum sequence length

    Returns:
    - Z_mb: generated random vector
    """

    # for i in range(batch_size):
    #     temp = torch.rand(seq_len, dim)
    #     temp = torch.add(torch.mul(temp, 2.0), -1.0)
    #     Z[i, :, :] = temp.detach().clone()
    # Z = torch.add(torch.mul(temp, 2.0), -1.0)

    Z_mb = list()
    for i in range(batch_size):
        temp = np.zeros([max_seq_len, z_dim])
        temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
        temp[:T_mb[i], :] = temp_Z
        Z_mb.append(temp_Z)
    Z_mb = torch.FloatTensor(Z_mb)
    Z_mb = Variable(Z_mb)
    return Z_mb


def train_test_divide(data_set, mode='test'):

    train_dataset_size = int(config.getfloat(
        mode, 'trainset_percentage') * len(data_set))
    test_dataset_size = len(data_set) - train_dataset_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        data_set, [train_dataset_size, test_dataset_size])

    # train_data_loader = DataLoader(dataset=train_dataset, batch_size=config.getint(
    #     mode, 'batch_size'), shuffle=True, num_workers=1)
    # test_data_loader = DataLoader(dataset=test_dataset, batch_size=config.getint(
    #     mode, 'batch_size'), shuffle=True, num_workers=1)

    return train_dataset, test_dataset


def _gradient_penalty(CUDA_DEVICES, discriminator, real_data, generated_data, data_time, gp_weight=10):

    batch_size = real_data.size()[0]
    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(CUDA_DEVICES)
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.to(CUDA_DEVICES)

    # Calculate probability of interpolated examples
    prob_interpolated = discriminator(interpolated, data_time)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(
                               prob_interpolated.size()).to(CUDA_DEVICES),
                           create_graph=True, retain_graph=True)[0]

    gradients = gradients.view(batch_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = gp_weight * ((gradients_norm - 1) ** 2).mean()

    # # Tensor = torch.cuda.FloatTensor if CUDA_DEVICES else torch.FloatTensor
    # alpha = torch.rand(real_data.size(0), 1, 1).to(CUDA_DEVICES)
    # # alpha = Tensor(np.random.random(
    # #     (real_data.size(0), 1, 1))).to(CUDA_DEVICES)
    # interpolates = (alpha * real_data + ((1 - alpha) *
    #                                      generated_data)).requires_grad_(True)
    # d_interpolates = discriminator(interpolates, data_time)
    # # fake = Variable(Tensor(d_interpolates.size()).fill_(
    # # 1.0).to(CUDA_DEVICES), requires_grad=False)
    # fake = Variable(torch.ones_like(d_interpolates).to(
    #     CUDA_DEVICES), requires_grad=False)
    # gradients = torch_grad(
    #     outputs=d_interpolates,
    #     inputs=interpolates,
    #     grad_outputs=fake,
    #     create_graph=True,
    #     retain_graph=True,
    #     only_inputs=True,
    # )[0]
    # gradients = gradients.reshape(gradients.size(0), -1)
    # gradient_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()

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
