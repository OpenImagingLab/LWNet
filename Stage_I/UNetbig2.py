import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from torch.fft import fftshift, fft2
import torchvision

class DownBlock(nn.Module):
    def __init__(self, num_convs, inchannels, outchannels, pool=True):
        super(DownBlock, self).__init__()
        blk = []
        if pool:
            blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
        for i in range(num_convs):
            if i == 0:
                blk.append(nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1))
            else:
                blk.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1))
            blk.append(nn.ReLU(inplace=True))
        self.layer = nn.Sequential(*blk)

    def forward(self, x):
        return self.layer(x)

class UpBlock(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(UpBlock, self).__init__()
        self.convt = nn.ConvTranspose2d(inchannels, outchannels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.convt(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, nchannels=1, nclasses=1):
        super(UNet, self).__init__()
        self.down1 = DownBlock(2, nchannels, 8, pool=False)
        self.down2 = DownBlock(3, 8, 16)
        self.down3 = DownBlock(3, 16, 32)
        self.down4 = DownBlock(3, 32, 64)
        self.down5 = DownBlock(3, 64, 128)
        self.up1 = UpBlock(128, 64)
        self.up2 = UpBlock(64, 32)
        self.up3 = UpBlock(32, 16)
        self.up4 = UpBlock(16, 8)
        self.out = nn.Sequential(nn.Conv2d(8, nclasses, kernel_size=1))

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x


class TVLoss(torch.nn.Module):
    """
    TV loss
    """
    def __init__(self, weight=1):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, input):
        batch_size = input.size()[0]
        channel = input.size()[1]
        h_x = input.size()[2]
        w_x = input.size()[3]
        x = torch.linspace(-1, 1, h_x-1)
        X, Y = torch.meshgrid(x, x)
        rou2 = X ** 2 + Y ** 2
        zero,rou2 = torch.tensor(0, dtype=torch.float32),torch.tensor(rou2, dtype=torch.float32)
        rou2 = rou2.repeat(batch_size,channel,1,1)
        if torch.cuda.is_available():
            rou2, zero = rou2.cuda(), zero.cuda()
        count_h = self._tensor_size(input[:, :, 1:, :])
        count_w = self._tensor_size(input[:, :, :, 1:])
        # h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        # w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        h_tv = torch.pow((input[:, :, 1:, 1:] - input[:, :, :h_x - 1, 1:]), 2)
        w_tv = torch.pow((input[:, :, 1:, 1:] - input[:, :, 1:, :w_x - 1]), 2)
        h_tv = torch.where(rou2>=1,zero,h_tv).sum()
        w_tv = torch.where(rou2>=1,zero,w_tv).sum()
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

def wavefront_fitting(WF):
    m = WF.size(0)
    x = torch.linspace(-1, 1, m)
    [Y,X] = torch.meshgrid(x, x)
    rho = abs(torch.sqrt(X** 2 + Y**2))
    theta = torch.arctan2(Y, X)
    Data = torch.zeros((m**2))
    index =0
    A = torch.zeros((m**2,37))
    for i in range(m):
        for j in range(m):
            if (WF[i,j]!=0 and (not torch.isnan(WF[i,j].detach()))) or abs((i-m//2)*(j-m//2))<100:
                rou = rho[i,j]
                Data[index] = WF[i,j]
                A[index, 0] = 1  # Z00
                A[index, 1] = 4 ** .5 * rou * torch.sin(theta[i,j])
                A[index, 2] = 3 ** .5 * (2 * rou ** 2 - 1)  # Z20
                A[index, 3] = 6 ** .5 * rou ** 2 * torch.cos(2 * theta[i,j])  # Z22
                A[index, 4] = 8 ** .5 * (3 * rou ** 3 - 2 * rou) * torch.sin(theta[i,j])
                A[index, 5] = 8 ** .5 * rou ** 3 * torch.sin(3 * theta[i,j])
                A[index, 6] = 5 ** .5 * (6 * rou ** 4 - 6 * rou ** 2 + 1)
                A[index, 7] = 10 ** .5 * (4 * rou ** 4 - 3 * rou ** 2) * torch.cos(2 * theta[i,j])
                A[index, 8] = 10 ** .5 * rou ** 4 * torch.cos(4 * theta[i,j])  # Z22
                A[index, 9] = 12 ** .5 * (10 * rou ** 5 - 12 * rou ** 3 + 3 * rou) * torch.sin(theta[i,j])
                A[index, 10] = 12 ** .5 * (5 * rou ** 5 - 4 * rou ** 3) * torch.sin(3 * theta[i,j])
                A[index, 11] = 12 ** .5 * rou ** 5 * torch.sin(5 * theta[i,j])
                A[index, 12] = 7 ** .5 * (20 * rou ** 6 - 30 * rou ** 4 + 12 * rou ** 2 - 1)
                A[index, 13] = 14 ** .5 * (15 * rou ** 6 - 20 * rou ** 4 + 6 * rou ** 2) * torch.cos(2 * theta[i,j])
                A[index, 14] = 14 ** .5 * (6 * rou ** 6 - 5 * rou ** 4) * torch.cos(4 * theta[i,j])
                A[index, 15] = 14 ** .5 * rou ** 6 * torch.cos(6 * theta[i,j])
                A[index, 16] = 16 ** .5 * (35 * rou ** 7 - 60 * rou ** 5 + 30 * rou ** 3 - 4 * rou) * torch.sin(theta[i,j])
                A[index, 17] = 16 ** .5 * (21 * rou ** 7 - 30 * rou ** 5 + 10 * rou ** 3) * torch.sin(3 * theta[i,j])
                A[index, 18] = 16 ** .5 * (7 * rou ** 7 - 6 * rou ** 5) * torch.sin(5 * theta[i,j])
                A[index, 19] = 16 ** .5 * rou ** 7 * torch.sin(7 * theta[i,j])
                A[index, 20] = 9 ** .5 * (70 * rou ** 8 - 140 * rou ** 6 + 90 * rou ** 4 - 20 * rou ** 2 + 1)  # Z60
                index = index +1

    Data =Data[0:index]
    A = A[0:index]
    # A = A.numpy()
    # Data = Data.numpy()
    B = torch.linalg.inv(np.dot(A.T,A)+0.0001*np.eye(37))
    zer_co = torch.dot(torch.dot(B,A.T),Data)
    fmatrix = torch.dot(A ,zer_co)
    error = sum(abs(fmatrix - Data)) / index
    return zer_co,error







