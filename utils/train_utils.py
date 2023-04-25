from datetime import datetime

import torch
import torch.nn as nn

import numpy as np

from utils.config import *

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams['figure.dpi'] = 100

def MAPE(x, y):
    return np.mean(np.abs(1 - y/x))

def RMSPE(x, y):
    return np.mean(np.sqrt(np.mean((1 - y/x)**2, 1)))

def MSE(x,y):
    return np.mean((x-y)**2)

def RMSE(x,y):
    return np.sqrt(np.mean((x-y)**2))

def MSGE(y):
    return np.mean(np.diff(y, axis=-1)**2)

def TMSGE(x, y):
    return np.mean((np.diff(x, axis=-1) - np.diff(y, axis=-1))**2)

def train(model, optimizer, loader, loss_func, acc_func, log_step=None, device=torch.device('cuda')):
    model.train()
    loss_lst = {"train_loss":[],
                "smoothness":[],
                "compared_grad":[],
                "% error":[],
                "MSE photo":[],
                "MSE raman":[]}
    for batch_idx, (x, *y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(device)
        y_ = model(x)

        loss = loss_func(y, y_, x)

        loss.backward()
        raman, photo = y
        raman = raman.numpy()
        photo = photo.numpy()
        new_raman, new_photo = y_[1].cpu().detach().numpy(), y_[0].cpu().detach().numpy()
        new_x = x.cpu().detach().numpy()
        loss_lst["train_loss"].append(loss.cpu().detach().numpy())
        loss_lst["smoothness"].append(MSGE(new_photo)/np.mean(new_x))
        loss_lst["compared_grad"].append(TMSGE(photo, new_photo))
        loss_lst["% error"].append(MAPE(photo, new_photo))
        loss_lst["MSE photo"].append(RMSE(photo, new_photo))
        loss_lst["MSE raman"].append(RMSE(raman, new_raman))

        # seems to help stabalize the learning
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
                    len(loader), acc_func(y, y_, x) #/ torch.mean(x)
                ), flush=True)
    return loss_lst

def test(model, loader, loss_func, acc_func, log_step, device=torch.device('cuda')):
    model.eval()
    acc = []

    with torch.no_grad():
        tmp = []
        for batch_idx, (x, *y) in enumerate(loader):
            x = x.to(device)
            y_ = model(x)
            loss = loss_func(y, y_, x) #/ torch.mean(x)
            if log_step:
                if batch_idx % log_step == 0:
                    print('  {}| {:5d}/{:5d}| bits: {:2.6f}'.format(
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), batch_idx,
                        len(loader), acc_func(y, y_, x, data="Validation") #/ torch.mean(x)
                    ), flush=True)

            tmp.append(loss.cpu().detach().numpy())
        print(f"Validation average loss: {np.mean(tmp)}")
