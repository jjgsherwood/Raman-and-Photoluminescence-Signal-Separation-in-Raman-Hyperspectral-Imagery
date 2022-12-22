from utils.config import *

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

import copy
import random

import numpy as np
from sklearn.model_selection import train_test_split

class Vector_unit_normalization():
    def __call__(self, x):
        return x / np.sqrt((x**2).sum(axis=3, keepdims=True))

    def __repr__(self):
        return "Vector unit normalization"

class RandomFlip():
    def __init__(self):
        self.rng = np.random.default_rng()

    def __call__(self, x):
        return torch.flip(x, tuple(self.rng.choice([-3,-2], size=self.rng.integers(3), replace=False)))

    def __repr__(self):
        return "Random horizontal flip and/or vertical flip"

class Random_Rotate_90():
    def __call__(self, x):
        return torch.rot90(x, random.randint(0,3), [-3, -2])

    def __repr__(self):
        return "Random 90 degree rotation"

class TensorDataset(Dataset):
    """
    TensorDataset with support of transforms.
    """
    def __init__(self, data, transform=None, sample_size=5):
        data, self.labels = zip(*data)
        self.labels = torch.Tensor(np.array(self.labels))
        self.data = torch.Tensor(np.array(data))
        self.n = sample_size
        self.transform = transform
        self.patches_per_width = self.data.size(2)-self.n
        self.patches_per_height = self.data.size(1)-self.n
        self.patches_per_images = (self.patches_per_width)*(self.patches_per_height)

    def __getitem__(self, index):
        index_patch = index % self.patches_per_images
        index_hight = index_patch % self.patches_per_height
        index_width = index_patch // self.patches_per_height
        index_image = index // self.patches_per_images
        x = self.data[index_image:index_image+1, index_hight:index_hight+self.n, index_width:index_width+self.n,:]
        if self.transform:
            x = self.transform(x)

        return x, self.labels[index_image]

    def __len__(self):
        return self.data.size(0)*self.patches_per_images

def split_data_per_image(dataset, validation_per=VALIDATION_PER):
    n_images = dataset.data.shape[0]
    n_test_images = max(1,int(validation_per * n_images))
    print(n_images, n_test_images)
    indices = random.sample(range(n_images), n_test_images)
    reverse_indices = [i for i in range(n_images) if i not in indices]
    train_dataset = copy.copy(dataset)
    train_dataset.split(reverse_indices)
    test_dataset  = copy.copy(dataset)
    test_dataset.split(indices)
    return train_dataset, test_dataset

def load_rawdata(data, batch_size):
    # tensor_data = torch.Tensor(data)
    dataset = RawDataset(data)
    test_loader = DataLoader(dataset, batch_size, shuffle=False,
                             drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)

    return test_loader

def load_splitdata(data, batch_size, validation_per=VALIDATION_PER):
    dataset = SplitDataset(data)

    # test data and train data are split on image level
    # Thus a whole images is either test data or train data
    if validation_per is not None:
        train_dataset, test_dataset = split_data_per_image(dataset, validation_per)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
                                  drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    else:
        test_dataset = dataset
        train_loader = None

    test_loader = DataLoader(test_dataset, batch_size, shuffle=False,
                             drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)

    return train_loader, test_loader

class RawDataset(Dataset):
    """
    TensorDataset with support of transforms.
    """
    def __init__(self, data, transform=None):
        self.data = torch.Tensor(data)
        self.transform = transform

    def __getitem__(self, index):
        pixel = index % (self.data.size(1) * self.data.size(2))
        index_x = pixel % self.data.size(1)
        index_y = pixel // self.data.size(1)
        index_image = index // (self.data.size(1) * self.data.size(2))
        data = self.data[index_image, index_x, index_y, :]

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return self.data.size(0) * self.data.size(1) * self.data.size(2)

class SplitDataset(Dataset):
    """
    TensorDataset with support of transforms.
    """
    def __init__(self, data, transform=None):
        self.data, self.raman, self.photo = zip(*data)
        self.data = torch.Tensor(np.array(self.data))
        self.raman = torch.Tensor(np.array(self.raman))
        self.photo = torch.Tensor(np.array(self.photo))
        self.transform = transform

    def split(self, indices):
        self.data = self.data[indices]
        self.raman = self.raman[indices]
        self.photo = self.photo[indices]

    def __getitem__(self, index):
        pixel = index % (self.data.size(1) * self.data.size(2))
        index_x = pixel % self.data.size(1)
        index_y = pixel // self.data.size(1)
        index_image = index // (self.data.size(1) * self.data.size(2))
        data, raman, photo = (self.data[index_image, index_x, index_y, :],
                             self.raman[index_image, index_x, index_y, :],
                             self.photo[index_image, index_x, index_y, :])
        if self.transform:
            data, raman, photo = self.transform(data), self.transform(raman), self.transform(photo)

        return data, raman, photo

    def __len__(self):
        return self.data.size(0) * self.data.size(1) * self.data.size(2)
