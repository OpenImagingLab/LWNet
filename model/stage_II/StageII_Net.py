# --coding:utf-8--
import torch.nn as nn
import torch
from torch.fft import fftshift, fft2
import torchvision

class zw(nn.Module):
    def __init__(self):
        super(zw, self).__init__()

    def forward(self, zer_co):
        M = 256
        x = torch.linspace(-1, 1, M)
        Y, X = torch.meshgrid(x, x, indexing='ij')
        rho = torch.abs(torch.sqrt(X ** 2 + Y ** 2))
        WF = torch.zeros(M, M)
        theta = torch.atan2(Y,X)
        A = torch.zeros((M, M, 21))
        A[:, :, 0] = torch.ones(M, M)  # Z00
        A[:, :, 1] = 4 ** .5 * rho * torch.sin(theta)
        A[:, :, 2] = 3 ** .5 * (2 * rho ** 2 - torch.ones(M, M))  # Z20
        A[:, :, 3] = 6 ** .5 * rho ** 2 * torch.cos(2 * theta)  # Z22
        A[:, :, 4] = 8 ** .5 * (3 * rho ** 3 - 2 * rho) * torch.sin(theta)
        A[:, :, 5] = 8 ** .5 * rho ** 3 * torch.sin(3 * theta)
        A[:, :, 6] = 5 ** .5 * (6 * rho ** 4 - 6 * rho ** 2 + torch.ones(M, M))
        A[:, :, 7] = 10 ** .5 * (4 * rho ** 4 - 3 * rho ** 2) * torch.cos(2 * theta)
        A[:, :, 8] = 10 ** .5 * rho ** 4 * torch.cos(4 * theta)  # Z22
        A[:, :, 9] = 12 ** .5 * (10 * rho ** 5 - 12 * rho ** 3 + 3 * rho) * torch.sin(theta)
        A[:, :, 10] = 12 ** .5 * (5 * rho ** 5 - 4 * rho ** 3) * torch.sin(3 * theta)
        A[:, :, 11] = 12 ** .5 * rho ** 5 * torch.sin(5 * theta)
        A[:, :, 12] = 7 ** .5 * (20 * rho ** 6 - 30 * rho ** 4 + 12 * rho ** 2 - torch.ones(M, M))
        A[:, :, 13] = 14 ** .5 * (15 * rho ** 6 - 20 * rho ** 4 + 6 * rho ** 2) * torch.cos(2 * theta)
        A[:, :, 14] = 14 ** .5 * (6 * rho ** 6 - 5 * rho ** 4) * torch.cos(4 * theta)
        A[:, :, 15] = 14 ** .5 * rho ** 6 * torch.cos(6 * theta)
        A[:, :, 16] = 16 ** .5 * (35 * rho ** 7 - 60 * rho ** 5 + 30 * rho ** 3 - 4 * rho) * torch.sin(theta)
        A[:, :, 17] = 16 ** .5 * (21 * rho ** 7 - 30 * rho ** 5 + 10 * rho ** 3) * torch.sin(3 * theta)
        A[:, :, 18] = 16 ** .5 * (7 * rho ** 7 - 6 * rho ** 5) * torch.sin(5 * theta)
        A[:, :, 19] = 16 ** .5 * rho ** 7 * torch.sin(7 * theta)
        A[:, :, 20] = 9 ** .5 * (70 * rho ** 8 - 140 * rho ** 6 + 90 * rho ** 4 - 20 * rho ** 2 + torch.ones(M, M))  # Z60
        for i in range(21):
            WF = WF + A[:, :, i] * zer_co[0,i]
        WF = torch.where(rho >= 1, 0, WF)
        return WF.float()

class FT(nn.Module):
    def __init__(self):
        super(FT, self).__init__()

    def forward(self,WF):
        M = WF.size(0)
        W =nn.ZeroPad2d(2*M)(WF)
        phase = torch.exp(-1j * 2 * torch.pi * W)
        phase=torch.where(phase==1,0,phase)
        AP = abs(fftshift(fft2(phase))) ** 2
        H = torchvision.transforms.CenterCrop(M)
        AP =H(AP)
        AP = AP / torch.max(AP)
        return AP.float()


class StageII(nn.Module):
    def __init__(self):
        super(StageII, self).__init__()
        self.fc0 = nn.Linear(5, 5)
        self.fc1 = nn.Linear(11,11)
        self.linears = nn.ModuleList([nn.Linear(1,1) for _ in range(5)])
        self.res = nn.ModuleList([nn.Linear(1, 1) for _ in range(5)])
        self.fc2 = zw()
        self.fc3 = FT()
        self.lrelu = nn.LeakyReLU(negative_slope=0.5, inplace=True)
        self.init_weights()

    def forward(self, limit):
        x = 0.1 * torch.ones(16)
        out1 = self.lrelu(self.fc0(x[0:5]))
        #8 10 11 13 14 15 16 17 18 19 20
        out2 = torch.tanh(self.fc1(x[5:]))  * limit[5:]
        if torch.rand(1)<=0.9521:
            out4 = torch.tanh(self.linears[0](out1[1].repeat(1, 1))) * out1[1]
        else:
            out4 = torch.tanh(self.linears[0](out1[1].repeat(1, 1))) * out1[1] + self.res[0](out1[1].repeat(1, 1))

        if torch.rand(1) <= 0.9181:
            out9 = torch.tanh(self.linears[1](out4)) * out4
        else:
            out9 = torch.tanh(self.linears[1](out4)) * out4 + self.res[1](out4) * out4

        if torch.rand(1) <= 0.9687:
            out5 = torch.tanh(self.linears[2](out1[1].repeat(1, 1))) * out1[1]
        else:
            out5 = torch.tanh(self.linears[2](out1[1].repeat(1, 1))) * out1[1] + self.res[2](out1[1].repeat(1, 1))* out1[1]

        if torch.rand(1) <= 0.9370:
            out7 = torch.tanh(self.linears[3](out1[3].repeat(1, 1))) * out1[3]
        else:
            out7 = torch.tanh(self.linears[3](out1[3].repeat(1, 1))) * out1[3] + self.res[3](out1[3].repeat(1, 1))* out1[3]

        if torch.rand(1) <= 0.8582:
            out12 = torch.tanh(self.linears[4](out1[4].repeat(1, 1))) * out1[4]
        else:
            out12 = torch.tanh(self.linears[4](out1[4].repeat(1, 1))) * out1[4] + self.res[4](out1[4].repeat(1, 1)) * out1[4]

        zer = torch.cat([out1[0:4].repeat(1, 1),out4,out5,out1[4].repeat(1, 1),out7,out2[0].repeat(1, 1),
                         out9,out2[1:3].repeat(1, 1),out12,out2[3:].repeat(1, 1)],dim=1)
        wf = self.fc2(zer)
        out = self.fc3(wf)
        return zer.float(),wf.float(),out.float()

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

# stricter version
class StageII_s(nn.Module):
    def __init__(self):
        super(StageII_s, self).__init__()
        self.fc0 = nn.Linear(5, 5)
        self.fc1 = nn.Linear(11,11)
        self.linears = nn.ModuleList([nn.Linear(1,1) for _ in range(5)])
        self.fc2 = zw()
        self.fc3 = FT()
        self.lrelu = nn.LeakyReLU(negative_slope=0.5, inplace=True)
        self.init_weights()

    def forward(self, x,limit):
        out1 = self.lrelu(self.fc0(x[0:5]))
        #8 10 11 13 14 15 16 17 18 19 20
        out2 = torch.tanh(self.fc1(x[5:]))  * limit[5:]
        out4 = torch.tanh(self.linears[0](out1[1].repeat(1, 1))) * out1[1]
        out9 = torch.tanh(self.linears[1](out4)) * out4
        out5 = torch.tanh(self.linears[2](out1[1].repeat(1, 1))) * out1[1]
        out7 = torch.tanh(self.linears[3](out1[3].repeat(1, 1))) * out1[3]
        out12 = torch.tanh(self.linears[4](out1[4].repeat(1, 1))) * out1[4]
        zer = torch.cat([out1[0:4].repeat(1, 1),out4,out5,out1[4].repeat(1,1),out7,out2[0].repeat(1,1),
                         out9,out2[1:3].repeat(1,1),out12,out2[3:].repeat(1, 1)],dim=1)
        wf = self.fc2(zer)
        out = self.fc3(wf)
        return zer,wf,out

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)