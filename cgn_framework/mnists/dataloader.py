import numpy as np
from PIL import Image, ImageColor
from pathlib import Path

import torch
import torch.nn.functional as F

from torch import tensor
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset

class ColoredMNIST(Dataset):
    def __init__(self, train, color_var=0.02):
        # get the colored mnist
        self.data_path = 'mnists/data/colored_mnist/mnist_10color_jitter_var_%.03f.npy'%color_var
        data_dic = np.load(self.data_path, encoding='latin1', allow_pickle=True).item()

        if train:
            self.ims = data_dic['train_image']
            self.labels = tensor(data_dic['train_label'], dtype=torch.long)
        else:
            self.ims = data_dic['test_image']
            self.labels = tensor(data_dic['test_label'], dtype=torch.long)

        self.T = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32), Image.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            ),
        ])

    def __getitem__(self, idx):
        ims, labels = self.T(self.ims[idx]), self.labels[idx]

        ret = {
            'ims': ims,
            'labels': labels,
        }

        return ret

    def __len__(self):
        return self.ims.shape[0]

class DoubleColoredMNIST(Dataset):

    def __init__(self, train=True):
        self.train = train
        self.mnist_sz = 32

        # get mnist
        mnist = datasets.MNIST('mnists/data', train=True, download=True)
        if train:
            ims, labels = mnist.data[:50000], mnist.targets[:50000]
        else:
            ims, labels = mnist.data[50000:], mnist.targets[50000:]

        self.ims_digit = torch.stack([ims, ims, ims], dim=1)
        self.labels = labels

        # colors generated by https://mokole.com/palette.html
        colors1 = [
            'darkgreen', 'darkblue', '#b03060',
            'orangered', 'yellow', 'burlywood', 'lime',
            'aqua', 'fuchsia', '#6495ed',
        ]
        # shift colors by X
        colors2 = [colors1[i-6] for i in range(len(colors1))]

        def get_rgb(x):
            t = torch.tensor(ImageColor.getcolor(x, "RGB"))/255.
            return t.view(-1, 1, 1)

        self.background_colors = list(map(get_rgb, colors1))
        self.object_colors = list(map(get_rgb, colors2))

        self.T = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        i = self.labels[idx] if self.train else np.random.randint(10)
        back_color = self.background_colors[i].clone()
        back_color += torch.normal(0, 0.01, (3, 1, 1))

        i = self.labels[idx] if self.train else np.random.randint(10)
        obj_color = self.object_colors[i].clone()
        obj_color += torch.normal(0, 0.01, (3, 1, 1))

        # get digit
        im_digit = (self.ims_digit[idx]/255.).to(torch.float32)
        im_digit = F.interpolate(im_digit[None,:], (self.mnist_sz, self.mnist_sz)).squeeze()
        im_digit = (im_digit > 0.1).to(int)  # binarize

        # plot digit onto the texture
        ims = im_digit*(obj_color) + (1 - im_digit)*back_color

        ret = {
            'ims': self.T(ims),
            'labels': self.labels[idx],
        }
        return ret

    def __len__(self):
        return self.labels.shape[0]

class WildlifeMNIST(Dataset):
    def __init__(self, train=True):
        self.train = train
        self.mnist_sz = 32
        inter_sz = 150

        # get mnist
        mnist = datasets.MNIST('mnists/data', train=True, download=True)
        if train:
            ims, labels = mnist.data[:50000], mnist.targets[:50000]
        else:
            ims, labels = mnist.data[50000:], mnist.targets[50000:]

        self.ims_digit = torch.stack([ims, ims, ims], dim=1)
        self.labels = labels

        # texture paths
        background_dir = Path('.') / 'mnists' / 'data' / 'textures' / 'background'
        self.background_textures = sorted([im for im in background_dir.glob('*.jpg')])
        object_dir = Path('.') / 'mnists' / 'data' / 'textures' / 'object'
        self.object_textures = sorted([im for im in object_dir.glob('*.jpg')])

        self.T_texture = transforms.Compose([
            transforms.Resize((inter_sz, inter_sz), Image.NEAREST),
            transforms.RandomCrop(self.mnist_sz, padding=3, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        # get textures
        i = self.labels[idx] if self.train else np.random.randint(10)
        back_text = Image.open(self.background_textures[i])
        back_text = self.T_texture(back_text)

        i = self.labels[idx] if self.train else np.random.randint(10)
        obj_text = Image.open(self.object_textures[i])
        obj_text = self.T_texture(obj_text)

        # get digit
        im_digit = (self.ims_digit[idx]/255.).to(torch.float32)
        im_digit = F.interpolate(im_digit[None, :], (self.mnist_sz, self.mnist_sz)).squeeze()
        im_digit = (im_digit > 0.1).to(int)  # binarize

        # plot digit onto the texture
        ims = im_digit*(obj_text) + (1 - im_digit)*back_text

        ret = {
            'ims': ims,
            'labels': self.labels[idx],
        }
        return ret

    def __len__(self):
        return self.labels.shape[0]

def get_dataloaders(dataset, batch_size, workers):
    if dataset == 'colored_MNIST':
        MNIST = ColoredMNIST
    elif dataset == 'double_colored_MNIST':
        MNIST = DoubleColoredMNIST
    elif dataset == 'wildlife_MNIST':
        MNIST = WildlifeMNIST
    else:
        raise TypeError(f"Unknown dataset: {dataset}")

    ds_train = MNIST(train=True)
    ds_test = MNIST(train=False)

    dl_train = DataLoader(ds_train, batch_size=batch_size,
                          shuffle=True, num_workers=workers)
    dl_test = DataLoader(ds_test, batch_size=batch_size*2,
                         shuffle=False, num_workers=workers)

    return dl_train, dl_test

TENSOR_DATASETS = [
    'colored_MNIST',
    'colored_MNIST_counterfactual',
    'colored_MNIST_gan',
    'double_colored_MNIST',
    'double_colored_MNIST_counterfactual',
    'double_colored_MNIST_gan',
    'wildlife_MNIST',
    'wildlife_MNIST_counterfactual',
    'wildlife_MNIST_gan',
]

def get_tensor_dataloaders(dataset, batch_size=64, combined=False):
    assert dataset in TENSOR_DATASETS, f"Unknown datasets {dataset}"

    if 'counterfactual' in dataset:
        tensor = torch.load(f'mnists/data/{dataset}.pth')

        if combined:
            # training data: original + counterfactual
            dataset_name = dataset.replace("counterfactual", "train")
            tensor_original_data = torch.load(f'mnists/data/{dataset_name}.pth')
            tensor[0] = torch.cat([tensor_original_data[0], tensor[0]], dim=0)
            tensor[1] = torch.cat([tensor_original_data[1], tensor[1]], dim=0)

        ds_train = TensorDataset(*tensor[:2])
        dataset = dataset.replace('_counterfactual', '')
    elif 'gan' in dataset:
        tensor = torch.load(f'mnists/data/{dataset}.pth')

        if combined:
            # training data: original + GAN-generated
            dataset_name = dataset.replace("gan", "train")
            tensor_original_data = torch.load(f'mnists/data/{dataset_name}.pth')
            tensor[0] = torch.cat([tensor_original_data[0], tensor[0]], dim=0)
            tensor[1] = torch.cat([tensor_original_data[1], tensor[1]], dim=0)

        ds_train = TensorDataset(*tensor[:2])
        dataset = dataset.replace('_gan', '')
    else:
        ds_train = TensorDataset(*torch.load(f'mnists/data/{dataset}_train.pth'))
    ds_test = TensorDataset(*torch.load(f'mnists/data/{dataset}_test.pth'))

    dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=4,
                          shuffle=True, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size*10, num_workers=4,
                         shuffle=False, pin_memory=True)

    return dl_train, dl_test
