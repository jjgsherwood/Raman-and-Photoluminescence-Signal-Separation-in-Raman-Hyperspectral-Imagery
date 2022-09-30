from datetime import datetime

import torch
import torch.nn as nn
from torch.distributions import Normal

import numpy as np
import math


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
                print('  {}| {:5d}/{:5d}| bits: {:2.6f}'.format(
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), batch_idx,
                    len(loader), acc_func(y, y_, x) / torch.mean(torch.sum(x, 1))
                ), flush=True)

def test(model, loader, loss_func, acc_func, device=torch.device('cuda')):
    model.eval()
    acc = []

    with torch.no_grad():
        tmp = []
        for batch_idx, (x, *y) in enumerate(loader):
            x = x.to(device)
            y_ = model(x)
            loss = acc_func(y, y_, x) / torch.mean(torch.sum(x, 1))
#             print('{}| Validation {:5d}/{:5d}| bits: {:2.6f}'.format(
#                 datetime.now().strftime('%Y-%m-%d %H:%M:%S'), batch_idx,
#                 len(loader), loss
#             ), flush=True)
            
            tmp.append(loss.cpu().detach().numpy())
        print(f"Validation average loss: {np.mean(tmp)}")
            