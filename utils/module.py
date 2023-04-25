import torch
import torch.nn as nn

LOSS1 = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
LOSS2 = nn.L1Loss(size_average=None, reduce=None, reduction='mean')

def loss_func(y, y_, x):
    raman, photo = y
    raman = raman.to(y_[0].device)
    photo = photo.to(y_[0].device)
    return (LOSS1(y_[0], photo) + LOSS1(y_[1], raman)) / torch.mean(x) + LOSS2(y_[0][:,1:] - y_[0][:,:-1], photo[:,1:] - photo[:,:-1])

def acc_func(y, y_, x, data="train"):
    """
    This function is especially usefull to show things each log step.
    """
    # y_1, y_2, y_3, y_4 = y_
    # y_1, y_2, y_3, y_4 = y_1.cpu().detach().numpy(), y_2.cpu().detach().numpy(), y_3.cpu().detach().numpy(), y_4.cpu().detach().numpy()
    # x = x.cpu().detach().numpy()
    # plt.title(f"plot {data} data")
    # plt.plot(x[0], label='raw')
    # plt.plot(y_1[0] + y_2[0], label='raman+photo')
    # plt.plot(np.abs(x[0]-y_1[0]-y_2[0]), label='noise', color='orange')
    # plt.plot(y[0][0], label='raman', color='c')
    # plt.plot(y[1][0], label='photo', color='r')
    # plt.plot(y_1[0], label='Conv1/photo', color='g')
    # plt.plot(y_2[0], label='Conv2/raman', color='brown')
    # plt.plot(y_3[0], label='Conv/pre_photo', color='b')
    # plt.plot(y_4[0], label='Conv/pre_raman', color='r')
    # plt.ylim(ymin=-10)
    # plt.xlim(xmin=0, xmax=1300)
    # plt.legend()
    # plt.show()
    return loss_func(y, y_, x)

class SelectLayer(nn.Module):
    def __init__(self, layer_index):
        super().__init__()
        self.layer_index = layer_index

    def forward(self, x):
        return x[:,self.layer_index]

class Conv_FFT(nn.Module):
    def __init__(self, num_input_channels : int = 2, base_channel_size: int = 16, act_fn : object = nn.GELU, groups : int = 2, **kwargs):
        super().__init__()
        c_hid = base_channel_size
        c_hid_2 = base_channel_size//2

        # FFT CNN split
        self.net = nn.Sequential(
            nn.Conv1d(num_input_channels, c_hid, kernel_size=11, padding=5, groups=groups, bias=False),
            act_fn(),
            nn.Conv1d(c_hid, c_hid, kernel_size=9, padding=4, groups=groups, bias=False),
            act_fn(),
            nn.Conv1d(c_hid, c_hid, kernel_size=7, padding=3, groups=groups, bias=False),
            act_fn(),
            nn.Conv1d(c_hid, c_hid, kernel_size=7, padding=3, groups=2*groups, bias=False),
            act_fn(),
            nn.Conv1d(c_hid, c_hid, kernel_size=5, padding=2, groups=2*groups, bias=False),
            act_fn(),
            nn.Conv1d(c_hid, c_hid, kernel_size=3, padding=1, groups=2*groups, bias=False),
            act_fn(),
            nn.Conv1d(c_hid, 2*num_input_channels, kernel_size=3, padding=1, groups=2*groups, bias=False)
        )

        # photo smooth part
        self.smooth_phase2 = nn.Sequential(
            # select only the photo input
            SelectLayer(slice(1,2)),
            nn.Conv1d(1, c_hid_2, kernel_size=5, padding=2, groups=1, bias=False),
            act_fn(),
            nn.Conv1d(c_hid_2, c_hid_2, kernel_size=5, padding=2, groups=1, bias=False),
            act_fn(),
            nn.Conv1d(c_hid_2, c_hid_2, kernel_size=5, padding=2, groups=1, bias=False),
            act_fn(),
            nn.Conv1d(c_hid_2, 1, kernel_size=3, padding=1, groups=1),
            SelectLayer(0)
        )
        self.smooth_phase1 = nn.Sequential(SelectLayer(1))

        # raman acc part
        self.raman_phase3 = nn.Sequential(
            nn.Conv1d(3, c_hid_2, kernel_size=5, padding=2, groups=1, bias=False),
            act_fn(),
            nn.Conv1d(c_hid_2, c_hid_2, kernel_size=5, padding=2, groups=1, bias=False),
            act_fn(),
            nn.Conv1d(c_hid_2, c_hid_2, kernel_size=5, padding=2, groups=1, bias=False),
            act_fn(),
            nn.Conv1d(c_hid_2, 1, kernel_size=3, padding=1, groups=1),
            SelectLayer(0)
        )
        self.raman_phase1 = nn.Sequential(SelectLayer(2))

        self.set_training(0)

    def set_training(self, value):
        if value == 0:
            self.smooth = self.smooth_phase1
            self.raman = self.raman_phase1
        elif value == 1:
            self.smooth = self.smooth_phase2
            self.raman = self.raman_phase1
        elif value == 2:
            self.smooth = self.smooth_phase2
            self.raman = self.raman_phase3

    def forward(self, x):
        n_wavenumbers = x.shape[-1]
        # rfft to go from wavenumbers to frequencies
        x0 = torch.fft.rfft(x, dim=1, norm='backward')
        x0 = torch.stack((x0.real, x0.imag), 1)
        x0 = self.net(x0)
        # this has 4 outputs 0,1 are real and 2,3 are imaginary
        x1, x2 = x0[:,[0,2]], x0[:,[1,3]]
        x1, x2 = torch.transpose(x1, -2, -1).contiguous(), torch.transpose(x2, -2, -1).contiguous()
        # make two complex signals which than can be inverted with the irfft to get wavenumbers.
        x1, x2 = torch.view_as_complex(x1), torch.view_as_complex(x2)
        x1, x2 = torch.fft.irfft(x1, n=n_wavenumbers, dim=1, norm='backward'), torch.fft.irfft(x2, n=n_wavenumbers, dim=1, norm='backward')
        # stack the raw, photo and raman such that each network can use what it needs.
        x3 = torch.stack((x, x1, x2),1)
        return self.smooth(x3), self.raman(x3), x1, x2
