import os

import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

import torch
import torch.nn as nn
import torch.optim as optim

from utils import module, config
from utils import train_utils as train
from utils import dataset_utils as dataset

import matplotlib.pyplot as plt
import pickle

class SupervisedSplitting():
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        _use_cuda = torch.cuda.is_available() and kwargs['cuda']
        if _use_cuda:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        self.device = torch.device('cuda' if _use_cuda else 'cpu')
        print(f"device: {self.device}")

    def fit(self, data, saved_NN, load_file=None):
        self.model = module.Conv_FFT().to(self.device)
        parameters = filter(lambda x: x.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(parameters, lr=self.kwargs['lr'])
        self.train_loader, self.test_loader = dataset.load_splitdata(data, self.kwargs['batch_size'])
        loss_lst = {"train_loss":[],
                    "smoothness":[],
                    "compared_grad":[],
                    "% error":[],
                    "MSE photo":[],
                    "MSE raman":[],
                    "validation_loss":[]}

        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.65)
        milestones = list(range(15,100,5))
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)

        for epoch in range(self.kwargs['epochs']):
            # old code should not do anything unless something weird goes wrong this gives some safety
            if os.path.exists(f"{saved_NN}//Conv_FFT_model_epoch{epoch}.pt"):
                print(f"epoch {epoch} is already trained")
                if not os.path.exists(f"{saved_NN}//Conv_FFT_model_epoch{epoch+1}.pt"):
                    self.model = torch.load(f"{saved_NN}//Conv_FFT_model_epoch{epoch}.pt", map_location=self.device)
                    parameters = filter(lambda x: x.requires_grad, self.model.parameters())
                    self.optimizer = optim.Adam(parameters, lr=self.kwargs['lr'])
                continue

            if load_file is not None:
                print(f"continue training using the save neurel network: {load_file}")
                self.model = torch.load(load_file, map_location=self.device)
                parameters = filter(lambda x: x.requires_grad, self.model.parameters())
                self.optimizer = optim.Adam(parameters, lr=self.kwargs['lr'])
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.65)

            if epoch >= config.PRETRAINING_PHASE_1:
                self.model.set_training(1)
            if epoch >= config.PRETRAINING_PHASE_2:
                self.model.set_training(2)

            print('-'*50)
            print('Epoch {:3d}/{:3d}'.format(epoch, self.kwargs['epochs']))
            start_time = datetime.now()
            loss = train.train(self.model, self.optimizer, self.train_loader, self.kwargs['loss_func'], self.kwargs['acc_func'], self.kwargs['log_step'], self.device)
            loss_lst["train_loss"] += loss["train_loss"]
            loss_lst["smoothness"] += loss["smoothness"]
            loss_lst["compared_grad"] += loss["compared_grad"]
            loss_lst["% error"] += loss["% error"]
            loss_lst["MSE photo"] += loss["MSE photo"]
            loss_lst["MSE raman"] += loss["MSE raman"]

            end_time = datetime.now()
            time_diff = relativedelta(end_time, start_time)
            print('Elapsed time: {}h {}m {}s'.format(time_diff.hours, time_diff.minutes, time_diff.seconds))
            loss = train.test(self.model, self.test_loader, self.kwargs['loss_func'], self.kwargs['acc_func'], self.kwargs['log_step'], self.device)
            loss_lst["validation_loss"].append(loss)
            torch.save(self.model, f"{saved_NN}//Conv_FFT_model_epoch{epoch}.pt")

            self.scheduler.step()

        with open(f"{saved_NN}//stats.pickle", 'wb') as handle:
            pickle.dump(loss_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)


        # plt.rcParams['figure.figsize'] = (50, 15)
        # plt.rcParams['figure.dpi'] = 250
        # plt.rcParams['lines.linewidth'] = 0.1

        # plt.title("train_loss")
        # plt.plot(range(len(loss_lst["train_loss"])), loss_lst["train_loss"])
        # plt.xticks(ticks=np.linspace(0,len(loss_lst["train_loss"]),self.kwargs['epochs']) ,labels=range(self.kwargs['epochs']))
        # plt.xlim(0,len(loss_lst["train_loss"]))
        # plt.xlabel("epochs")
        # plt.ylabel("train loss")
        # plt.show()
        #
        # plt.title("smoothness")
        # plt.plot(range(len(loss_lst["train_loss"])), loss_lst["smoothness"])
        # plt.xticks(ticks=np.linspace(0,len(loss_lst["train_loss"]),self.kwargs['epochs']) ,labels=range(self.kwargs['epochs']))
        # plt.xlim(0,len(loss_lst["train_loss"]))
        # plt.xlabel("epochs")
        # plt.ylabel("photoluminescence MSGE")
        # plt.show()
        #
        # plt.title("compared_grad")
        # plt.plot(range(len(loss_lst["train_loss"])), loss_lst["compared_grad"])
        # plt.xticks(ticks=np.linspace(0,len(loss_lst["train_loss"]),self.kwargs['epochs']) ,labels=range(self.kwargs['epochs']))
        # plt.xlim(0,len(loss_lst["train_loss"]))
        # plt.xlabel("epochs")
        # plt.ylabel("photoluminescence TMSGE")
        # plt.show()
        #
        # plt.title("% error")
        # plt.plot(range(len(loss_lst["train_loss"])), loss_lst["% error"])
        # plt.xticks(ticks=np.linspace(0,len(loss_lst["train_loss"]),self.kwargs['epochs']) ,labels=range(self.kwargs['epochs']))
        # plt.xlim(0,len(loss_lst["train_loss"]))
        # plt.xlabel("epochs")
        # plt.ylabel("photoluminescence MAPE")
        # plt.show()
        #
        # plt.title("RMSE photo")
        # plt.plot(range(len(loss_lst["train_loss"])), loss_lst["MSE photo"])
        # plt.xticks(ticks=np.linspace(0,len(loss_lst["train_loss"]),self.kwargs['epochs']) ,labels=range(self.kwargs['epochs']))
        # plt.xlim(0,len(loss_lst["train_loss"]))
        # plt.xlabel("epochs")
        # plt.ylabel("photoluminescence RMSE")
        # plt.show()
        #
        # plt.title("RMSE raman")
        # plt.plot(range(len(loss_lst["train_loss"])), loss_lst["MSE raman"])
        # plt.xticks(ticks=np.linspace(0,len(loss_lst["train_loss"]),self.kwargs['epochs']) ,labels=range(self.kwargs['epochs']))
        # plt.xlim(0,len(loss_lst["train_loss"]))
        # plt.xlabel("epochs")
        # plt.ylabel("Raman RMSE")
        # plt.show()
        return self

    def predict(self, data, load_file):
        """
        predicts the splitting of data on new data given a trained Conv_FFT.
        """
        self.model = torch.load(load_file, map_location=self.device)
        self.model.eval()

        dataloader = dataset.load_rawdata(data, self.kwargs['batch_size'])
        photo = np.empty((np.prod(data.shape[:-1]), data.shape[-1]))
        raman = np.empty((np.prod(data.shape[:-1]), data.shape[-1]))
        for batch_idx, x in enumerate(dataloader):
            x = x.to(self.device)
            y_1, y_2, *_ = self.model(x)
            photo[batch_idx*self.kwargs['batch_size']:(batch_idx+1)*self.kwargs['batch_size']] = y_1.cpu().detach().numpy()
            raman[batch_idx*self.kwargs['batch_size']:(batch_idx+1)*self.kwargs['batch_size']] = y_2.cpu().detach().numpy()
        return photo.reshape(data.shape), raman.reshape(data.shape)
