import os

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
import pickle as pl

# code taken and adapted from https://colab.research.google.com/github/reiinakano/invariant-risk-minimization/blob/master/invariant_risk_minimization_colored_mnist.ipynb

def color_grayscale_arr(arr, red=True):
    """Converts grayscale image to either red or green"""
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])
    if red:
        arr = np.concatenate([arr,
                              np.zeros((h, w, 2), dtype=dtype)], axis=2)
    else:
        arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                              arr,
                              np.zeros((h, w, 1), dtype=dtype)], axis=2)
    return arr


class ColoredMNIST(datasets.VisionDataset):
    """
  Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

  Args:
    root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
    env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
    transform (callable, optional): A function/transform that  takes in an PIL image
      and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
      target and transforms it.
  """

    def __init__(self, cfg, root='./data', env='train', transform=None, target_transform=None):
        super(ColoredMNIST, self).__init__(root, transform=transform,
                                           target_transform=target_transform)
        self.cfg = cfg
        self.prepare_colored_mnist()
        self.transformation0 = transforms.RandomRotation((-45, 45))
        self.transformation1 = transforms.RandomAffine(0, (0.2, 0.2))

        if env in ['train', 'test']:
            self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt')
        else:
            raise RuntimeError(f'{env} env unknown. Valid envs are train, test, and all_train')

    def __getitem__(self, index):
        """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
        img, target = self.data_label_tuples[index]

        if self.transform is not None:
            img = self.transform(img)
        # if target == 1:   # for data augmentation experiment with 2 classes
        #     img = self.transformation0(img)
        # else:
        #     img = self.transformation1(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data_label_tuples)

    def prepare_colored_mnist(self):
        colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
        if os.path.exists(os.path.join(colored_mnist_dir, 'train.pt')) \
                and os.path.exists(os.path.join(colored_mnist_dir, 'test.pt')):
            print('Colored MNIST dataset already exists')
            return

        print('Preparing Colored MNIST')
        train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

        train_set = []
        test_set = []
        for idx, (im, label) in enumerate(train_mnist):
            if idx % 10000 == 0:
                print(f'Converting image {idx}/{len(train_mnist)}')
            im_array = np.array(im)

            if self.cfg.MODEL.C == 2:
                # Assign a binary label y to the image based on the digit
                if label == 1:
                    _label = 0
                elif label == 5:
                    _label = 1
                else:
                    _label = 555
            else:
                _label = label

            if np.random.uniform() < 0.5:
                color_red = True
            else:
                color_red = False

            colored_arr = color_grayscale_arr(im_array, red=color_red)

            if _label != 555:
                if idx < 48000:
                    train_set.append((Image.fromarray(colored_arr), _label))
                else:
                    test_set.append((Image.fromarray(colored_arr), _label))

        os.mkdir(colored_mnist_dir) if not os.path.isdir(colored_mnist_dir) else 0
        torch.save(train_set, os.path.join(colored_mnist_dir, 'train.pt'))
        torch.save(test_set, os.path.join(colored_mnist_dir, 'test.pt'))


def plot_dataset_digits(dataset):
    fig = plt.figure(figsize=(13, 8))
    columns = 6
    rows = 3
    # ax enables access to manipulate each of subplots
    ax = []

    for i in range(columns * rows):
        img, label = dataset[i]
        # create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title("Label: " + str(label))  # set title
        plt.imshow(img)

    plt.show()  # finally, render the plot


class ColorMNISTTransform(ColoredMNIST):
    def __init__(self, swap, **kwargs):
        super().__init__(**kwargs)
        self.swap = swap
        self.transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))])

    def __getitem__(self, idx):
        img, target = self.data_label_tuples[idx]
        if self.swap:
            np_img = np.array(img)
            img = Image.fromarray(np.stack([np_img[:, :, 1], np_img[:, :, 0], np_img[:, :, 2]], axis=2))

        img = self.transformation(img)

        return img, target


if __name__ == '__main__':
    train_set = ColoredMNIST(root='./data', env='train1')
    plot_dataset_digits(train_set)
