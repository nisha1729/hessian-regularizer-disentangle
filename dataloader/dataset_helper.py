import torch
import numpy as np
from torch.utils import data
import torchvision
from torchvision import transforms

from dataloader.colour_mnist import ColoredMNIST
from dataloader.dsprites import DspritesDataset
from dataloader.pendulum import PendulumDataset


def get_cifar_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    total_train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_len = int(0.9 * len(total_train_data))
    val_len = len(total_train_data) - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(total_train_data, [train_len, val_len])

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return train_dataset, val_dataset, test_dataset


def get_celeba_data():
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    total_train_data = torchvision.datasets.CelebA(root='./data', split='train', download=False, transform=transform)
    train_len = int(0.9 * len(total_train_data))
    val_len = len(total_train_data) - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(total_train_data, [train_len, val_len])

    test_dataset = torchvision.datasets.CelebA(root='./data', split='test', download=True, transform=transform)
    return train_dataset, val_dataset, test_dataset


def get_pendulum_data():
    total_train_data = PendulumDataset("./data/pendulum/train")
    train_len = int(0.9 * len(total_train_data))
    val_len = len(total_train_data) - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(total_train_data, [train_len, val_len])

    test_dataset = PendulumDataset("./data/pendulum/test")
    return train_dataset, val_dataset, test_dataset


def get_clr_mnist_data(cfg):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))])
    total_train_data = ColoredMNIST(cfg, root='./data', env='train', transform=transform)
    train_len = int(0.8 * len(total_train_data))
    val_len = len(total_train_data) - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(total_train_data, [train_len, val_len])
    test_dataset = ColoredMNIST(cfg, root='./data', env='test', transform=transform)
    return train_dataset, val_dataset, test_dataset


def get_dsprites_data(cfg):
    total_train_data = DspritesDataset(root='./data', env='train')
    train_len = int(0.8 * len(total_train_data))
    val_len = len(total_train_data) - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(total_train_data, [train_len, val_len])
    test_dataset = DspritesDataset(root='./data', env='test', eval_mode=cfg.TEST.EVAL_MODE)
    return train_dataset, val_dataset, test_dataset


def get_dsprites_test_data(cfg):
    test_dataset = DspritesDataset(root='./data', env='test', eval_mode=cfg.TEST.EVAL_MODE)
    return test_dataset
