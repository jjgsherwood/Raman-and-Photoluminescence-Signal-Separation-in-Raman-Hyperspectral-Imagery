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

def split_data(dataset, test_size=TEST_RATIO, seed=42):
    np.random.seed(seed)
    train, test = train_test_split(dataset.data, test_size=test_size)
    train_dataset = copy.copy(dataset)
    test_dataset = copy.copy(dataset)

    train_dataset.data = train
    test_dataset.data = test
    return train_dataset, test_dataset

def load_liver(data, batch_size, sample_size=5):
    transform = transforms.Compose([
        Vector_unit_normalization(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        Random_Rotate_90(),
    ])

    # tensor_data = torch.Tensor(data)
    dataset = TensorDataset(data, transform, sample_size)

    # test data and train data are split on image level
    # Thus a whole images is either test data or train data
    train_dataset, test_dataset = split_data(dataset)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
                              drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False,
                             drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)

    return train_loader, test_loader

def load_liver_all(data):
    transform = transforms.Compose([
        Vector_unit_normalization(),
    ])

    data = create_dataset_1x1(data)
    return torch.Tensor(data)
