from datetime import datetime

import torch
import torch.nn as nn
from torch.distributions import Normal

import numpy as np


def train(model, optimizer, loader, loss_func, acc_func, log_step=None, device=torch.device('cuda')):
    model.train()

    for batch_idx, (x, *y) in enumerate(loader):
        x = x.to(device)      
        y_ = model(x)

        loss = loss_func(y, y_)

        optimizer.zero_grad()
        model.zero_grad()
        loss.backward()

        optimizer.step()

        if log_step:
            if batch_idx % log_step == 0:
                print('  {}| {:5d}/{:5d}| bits: {:2.2f}'.format(
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), batch_idx,
                    len(loader), acc_func(y, y_)
                ), flush=True)

def test(model, loader, loss_func, acc_func, device=torch.device('cuda')):
    model.eval()
    acc = []

    with torch.no_grad():
        for x, *y in loader:
            x = x.to(device)
            y_ = model(x)
            acc.append(acc_func(y, y_))
            
    return np.mean(acc)