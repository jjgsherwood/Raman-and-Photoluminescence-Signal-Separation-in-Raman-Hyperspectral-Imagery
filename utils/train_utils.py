from datetime import datetime

import torch
import torch.nn as nn
from torch.distributions import Normal

import numpy as np
import math

from utils.config import *

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams['figure.dpi'] = 100

def train(model, optimizer, loader, loss_func, acc_func, log_step=None, device=torch.device('cuda')):
    model.train()

    for batch_idx, (x, *y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(device)
        y_ = model(x)

        loss = loss_func(y, y_, x)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer.step()

        if log_step:
            if batch_idx % log_step == 0:
                if SHOW_WEIGHTS:
                    for layer in model.net:
                        try:
                            layer.weight
                        except AttributeError:
                            continue
                        print(layer)
                        data = layer.weight.detach().numpy()
                        print(data.shape)
                        plt.imshow(data.reshape(-1,data.shape[-1]))
#                         for f in data.reshape(-1,data.shape[-1]):
#                             plt.plot(f)
                        plt.show()

                if SHOW_GRADS:
                    for layer in model.net:
                        try:
                            layer.weight
                        except AttributeError:
                            continue
                        print(layer)
                        data = layer.weight.grad.detach().numpy().flatten()
                        plt.hist(data, 50, density=True)
                        plt.show()

                print('  {}| {:5d}/{:5d}| bits: {:2.6f}'.format(
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), batch_idx,
                    len(loader), acc_func(y, y_, x) / torch.mean(torch.sum(x, 1))
                ), flush=True)

def test(model, loader, loss_func, acc_func, log_step, device=torch.device('cuda')):
    model.eval()
    acc = []

    with torch.no_grad():
        tmp = []
        for batch_idx, (x, *y) in enumerate(loader):
            x = x.to(device)
            y_ = model(x)
            loss = loss_func(y, y_, x) / torch.mean(torch.sum(x, 1))
            if log_step:
                if batch_idx % log_step == 0:
                    print('  {}| {:5d}/{:5d}| bits: {:2.6f}'.format(
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), batch_idx,
                        len(loader), acc_func(y, y_, x, data="Validation") / torch.mean(torch.sum(x, 1))
                    ), flush=True)

            tmp.append(loss.cpu().detach().numpy())
        print(f"Validation average loss: {np.mean(tmp)}")
