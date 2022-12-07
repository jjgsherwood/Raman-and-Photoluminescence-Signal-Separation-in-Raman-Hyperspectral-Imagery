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

# def split_data(dataset, test_size=TEST_RATIO, seed=42):
#     """ Something might be wrong here """
#     np.random.seed(seed)
#     train, test = train_test_split(dataset.data, test_size=test_size)
#     train_dataset = copy.copy(dataset)
#     test_dataset = copy.copy(dataset)
#
#     train_dataset.data = train
#     test_dataset.data = test
#     return train_dataset, test_dataset

# def load_liver(data, batch_size, sample_size=5):
#     transform = transforms.Compose([
#         Vector_unit_normalization(),
#         RandomFlip(),
#         Random_Rotate_90(),
#     ])
#
#     # tensor_data = torch.Tensor(data)
#     dataset = TensorDataset(data, transform, sample_size)
#
#     # test data and train data are split on image level
#     # Thus a whole images is either test data or train data
#     train_dataset, test_dataset = split_data(dataset)
#
#     train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
#                               drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
#     test_loader = DataLoader(test_dataset, batch_size, shuffle=False,
#                              drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
#
#     return train_loader, test_loader
#
# def load_liver_all(data):
#     transform = transforms.Compose([
#         Vector_unit_normalization(),
#     ])
#
#     data = create_dataset_1x1(data)
#     return torch.Tensor(data)

def split_data_per_image(dataset, test_size=TEST_RATIO):
    n_images = dataset.data.shape[0]
    n_test_images = int(TEST_RATIO * n_images)
    indices = random.sample(range(n_images), n_test_images)
    reverse_indices = [i for i in range(n_images) if i not in indices]
    train_dataset = copy.copy(dataset)
    train_dataset.split(reverse_indices)
    test_dataset  = copy.copy(dataset)
    test_dataset.split(indices)
    return train_dataset, test_dataset

def load_splitdata(data, batch_size, test_size=TEST_RATIO):
    # tensor_data = torch.Tensor(data)
    dataset = SplitDataset(data)

    # test data and train data are split on image level
    # Thus a whole images is either test data or train data
    if test_size is not None:
        train_dataset, test_dataset = split_data_per_image(dataset, test_size)
    else:
        train_dataset = dataset

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
                              drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    if test_size is not None:
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False,
                                 drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    else:
        test_loader = None

    return train_loader, test_loader

class SplitDataset(Dataset):
    """
    TensorDataset with support of transforms.
    """
    def __init__(self, data, transform=None):
        self.data, self.raman, self.photo, self.labels = zip(*data)
        self.labels = torch.Tensor(np.array(self.labels))
        self.data = torch.Tensor(np.array(self.data))
        self.raman = torch.Tensor(np.array(self.raman))
        self.photo = torch.Tensor(np.array(self.photo))
        self.transform = transform

    def split(self, indices):
        self.data = self.data[indices]
        self.raman = self.raman[indices]
        self.photo = self.photo[indices]
        self.labels = self.labels[indices]

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

#         return data, raman, photo, (index_image, index_x, index_y)
        return data, raman, photo, self.labels[index_image]

    def __len__(self):
        return self.data.size(0) * self.data.size(1) * self.data.size(2)
