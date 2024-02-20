# --coding:utf-8--
# from FFT import *
import torch
import torchvision
import torch.nn as nn
# import torch.fft as fft
import pandas as pd
import numpy as np
import math
from torch.fft import fftshift, fft2

#input zernike coefficient and pixel size(M*M),output wavefront
def zernike_wavefront(zer_co, M):
    x = torch.linspace(-1, 1, M)
    Y, X = torch.meshgrid(x, x)
    rho = torch.abs(torch.sqrt(X ** 2 + Y ** 2))
    WF = torch.zeros(M, M)
    theta = torch.atan2(Y, X)
    A = torch.zeros((M, M, 21))
    A[:, :, 0] = torch.ones(M, M)  # Z00
    A[:, :, 1] = 4**.5*rho * torch.sin(theta)
    A[:, :, 2] = 3**.5*(2 * rho ** 2 - torch.ones(M, M))  # Z20
    A[:, :, 3] = 6**.5*rho ** 2 * torch.cos(2 * theta)  # Z22
    A[:, :, 4] = 8**.5*(3 * rho ** 3 - 2 * rho) * torch.sin(theta)
    A[:, :, 5] = 8**.5*rho ** 3 * torch.sin(3 * theta)
    A[:, :, 6] = 5**.5*(6 * rho ** 4 - 6 * rho ** 2 + torch.ones(M, M))
    A[:, :, 7] = 10**.5*(4 * rho ** 4 - 3 * rho ** 2) * torch.cos(2 * theta)
    A[:, :, 8] = 10**.5*rho ** 4 * torch.cos(4 * theta)  # Z22
    A[:, :, 9] = 12**.5*(10 * rho ** 5 - 12 * rho ** 3 + 3 * rho) * torch.sin(theta)
    A[:, :, 10] = 12**.5*(5 * rho ** 5 - 4 * rho ** 3 ) * torch.sin(3*theta)
    A[:, :, 11] = 12**.5* rho ** 5 * torch.sin(5*theta)
    A[:, :, 12] = 7**.5*(20 * rho ** 6 - 30 * rho ** 4 + 12 * rho ** 2 - torch.ones(M, M))
    A[:, :, 13] = 14**.5*(15 * rho ** 6 - 20 * rho ** 4 + 6 * rho ** 2)* torch.cos(2*theta)
    A[:, :, 14] = 14**.5*(6 * rho ** 6 - 5 * rho ** 4 ) * torch.cos(4*theta)
    A[:, :, 15] = 14**.5*rho ** 6 * torch.cos(6 * theta)
    A[:, :, 16] = 16**.5*(35 * rho ** 7 - 60 * rho ** 5 + 30 * rho**3 -4*rho) * torch.sin(theta)
    A[:, :, 17] = 16**.5*(21 * rho ** 7 - 30 * rho ** 5 + 10 * rho**3) * torch.sin(3*theta)
    A[:, :, 18] = 16**.5*(7 * rho ** 7 - 6 * rho ** 5) * torch.sin(5 * theta)
    A[:, :, 19] = 16**.5*rho ** 7 * torch.sin(7 * theta)
    A[:, :, 20] = 9**.5*(70 * rho ** 8 - 140 * rho ** 6 + 90 * rho ** 4 - 20 * rho ** 2 + torch.ones(M, M)) # Z60

    for i in range(21):
        WF = WF + A[:, :, i] * zer_co[i]

    WF = torch.where(rho >= 1, 0, WF)

    return WF

#input wavefront,output far field diffraction intensity AP
def FourierTransform(WF):
    M = WF.size(0)
    W =nn.ZeroPad2d(2*M)(WF)
    phase = torch.exp(1j * 2 * torch.pi * W)
    phase=torch.where(phase==1,0,phase)
    AP = abs(fftshift(fft2(phase))) ** 2
    H = torchvision.transforms.CenterCrop(M)
    AP =H(AP)
    AP = AP / torch.max(AP)

    return AP

#itorchut wavefront distribution, output fitting coefficient and error
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
            if (WF[i,j]!=0 and (not np.isnan(WF[i,j]))) or abs((i-m//2)*(j-m//2))<100:
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
    A = A.numpy()
    Data = Data.numpy()
    B = np.linalg.inv(np.dot(A.T,A)+0.0001*np.eye(37))
    zer_co = np.dot(np.dot(B,A.T),Data)
    fmatrix = np.dot(A ,zer_co)
    error = sum(abs(fmatrix - Data)) / index
    return zer_co,error


def dataloader(dir,filename,degree):
    filestr = dir + filename
    sheetstr = '%s'%degree
    Datatemp = pd.read_excel(filestr, sheetstr, header=None, index_col=None)
    Data = np.array(Datatemp, dtype=np.float64)
    Data = torch.tensor(Data)
    return Data

def crop(image):
    # M = image.size(0)
    TotalE = torch.sum(image)
    for i in range(7):
        pixelnum = (i+2)*32
        H = torchvision.transforms.CenterCrop(pixelnum)
        croped = H(image)
        if torch.sum(croped)> 0.99*TotalE:
            break
    return croped,pixelnum

def compare_PSNR(img1,img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean((img1-img2)**2)
    if mse == 0:
        return 100

    pixel_max = 1
    return 20*math.log10(pixel_max / math.sqrt(mse))



