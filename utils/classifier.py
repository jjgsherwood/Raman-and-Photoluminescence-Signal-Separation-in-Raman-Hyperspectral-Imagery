


class SupervisedClassifier(BaseEstimator):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        _use_cuda = torch.cuda.is_available() and kwargs['cuda']
        if _use_cuda:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        self.device = torch.device('cuda' if _use_cuda else 'cpu')
        print(f"device: {self.device}")

    def fit(self, data):
        self.model = Conv_FFT().to(self.device)

        parameters = filter(lambda x: x.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(parameters, lr=self.kwargs['lr'])
        self.train_loader, self.test_loader = dataset.load_splitdata(data, self.kwargs['batch_size'])

        for epoch in range(self.kwargs['epochs']):
            if os.path.exists(f"{saved_NN}//Conv_FFT_model_epoch{epoch}.pt"):
                print(f"epoch {epoch} is already trained")
                if not os.path.exists(f"{saved_NN}//Conv_FFT_model_epoch{epoch+1}.pt"):
                    self.model = torch.load(f"{saved_NN}//Conv_FFT_model_epoch{epoch}.pt", map_location=self.device)
                    parameters = filter(lambda x: x.requires_grad, self.model.parameters())
                    self.optimizer = optim.Adam(parameters, lr=self.kwargs['lr'])
                continue

            if epoch >= 3:
                self.model.set_training(1)
            if epoch >= 6:
                self.model.set_training(2)

            print('-'*50)
            print('Epoch {:3d}/{:3d}'.format(epoch, self.kwargs['epochs']))
            start_time = datetime.now()
            train.train(self.model, self.optimizer, self.train_loader, self.kwargs['loss_func'], self.kwargs['acc_func'], self.kwargs['log_step'], self.device)
            end_time = datetime.now()
            time_diff = relativedelta(end_time, start_time)
            print('Elapsed time: {}h {}m {}s'.format(time_diff.hours, time_diff.minutes, time_diff.seconds))
            train.test(self.model, self.test_loader, self.kwargs['loss_func'], self.kwargs['acc_func'], self.kwargs['log_step'], self.device)
            torch.save(self.model, f"{saved_NN}//Conv_FFT_model_epoch{epoch}.pt")
        return self

    def predict(self, data=None):
        """
        """
        if data is not None:
            _, dataloader = dataset.load_splitdata(data, self.kwargs['batch_size'], test_size=None)
            for batch_idx, (x, *y) in enumerate(dataloader):
                x = x.to(self.device)
                y_1, y_2, y_3, y_4 = self.model(x)
                y1, y2, *_ = y
                yield x.cpu().detach().numpy(), y1, y2, y_1.cpu().detach().numpy(), y_2.cpu().detach().numpy(), y_3.cpu().detach().numpy(), y_4.cpu().detach().numpy()
        else:
            for batch_idx, (x, *y) in enumerate(self.test_loader):
                x = x.to(self.device)
                y_1, y_2, *_ = self.model(x)
                y1, y2, *_ = y
                yield x.cpu().detach().numpy(), y1, y2, y_1.cpu().detach().numpy(), y_2.cpu().detach().numpy()
