# --coding:utf-8--
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.fft import fftshift, fft2,ifft2
import pandas as pd

#input wavefront,output far field diffraction intensity AP
def FourierTransform(WF):
    if isinstance(WF, np.ndarray):
        WF = torch.from_numpy(WF)
    M = WF.size(0)
    W =nn.ZeroPad2d(2*M)(WF)
    # W = nn.ZeroPad2d(0)(WF)
    phase = torch.exp(-1j * 2 * torch.pi * W)
    phase=torch.where(phase==1,0,phase)
    AP = abs(fftshift(fft2(phase))) ** 2
    H = torchvision.transforms.CenterCrop(M)
    AP =H(AP)
    AP = AP / torch.max(AP)
    return AP.numpy()


def zernike_wavefront(zer_co, A,rou,BS):
    WF = torch.zeros(BS,256,256)
    WF = WF
    for i in range(21):
        WF = WF + A[..., i] *zer_co[:,i,:].unsqueeze(2)
    WF = torch.where(rou >= 1, 0, WF)
    return WF

def fitting_prepare(BS):
    m = 256
    x = torch.linspace(-1, 1, m)
    [Y, X] = torch.meshgrid(x, x, indexing='ij')
    rou = abs(torch.sqrt(X ** 2 + Y ** 2))
    theta = torch.arctan2(Y, X)
    mask = rou.reshape(256 * 256, 1) < 1

    A = torch.zeros((256,256, 21))
    A[..., 0] = 1  # Z00
    A[..., 1] = 4 ** .5 * rou * torch.sin(theta)
    A[..., 2] = 3 ** .5 * (2 * rou ** 2 - 1)  # Z20
    A[..., 3] = 6 ** .5 * rou ** 2 * torch.cos(2 * theta)  # Z22
    A[..., 4] = 8 ** .5 * (3 * rou ** 3 - 2 * rou) * torch.sin(theta)
    A[..., 5] = 8 ** .5 * rou ** 3 * torch.sin(3 * theta)
    A[..., 6] = 5 ** .5 * (6 * rou ** 4 - 6 * rou ** 2 + 1)
    A[..., 7] = 10 ** .5 * (4 * rou ** 4 - 3 * rou ** 2) * torch.cos(2 * theta)
    A[..., 8] = 10 ** .5 * rou ** 4 * torch.cos(4 * theta)  # Z22
    A[..., 9] = 12 ** .5 * (10 * rou ** 5 - 12 * rou ** 3 + 3 * rou) * torch.sin(theta)
    A[..., 10] = 12 ** .5 * (5 * rou ** 5 - 4 * rou ** 3) * torch.sin(3 * theta)
    A[..., 11] = 12 ** .5 * rou ** 5 * torch.sin(5 * theta)
    A[..., 12] = 7 ** .5 * (20 * rou ** 6 - 30 * rou ** 4 + 12 * rou ** 2 - 1)
    A[..., 13] = 14 ** .5 * (15 * rou ** 6 - 20 * rou ** 4 + 6 * rou ** 2) * torch.cos(2 * theta)
    A[..., 14] = 14 ** .5 * (6 * rou ** 6 - 5 * rou ** 4) * torch.cos(4 * theta)
    A[..., 15] = 14 ** .5 * rou ** 6 * torch.cos(6 * theta)
    A[..., 16] = 16 ** .5 * (35 * rou ** 7 - 60 * rou ** 5 + 30 * rou ** 3 - 4 * rou) * torch.sin(theta)
    A[..., 17] = 16 ** .5 * (21 * rou ** 7 - 30 * rou ** 5 + 10 * rou ** 3) * torch.sin(3 * theta)
    A[..., 18] = 16 ** .5 * (7 * rou ** 7 - 6 * rou ** 5) * torch.sin(5 * theta)
    A[..., 19] = 16 ** .5 * rou ** 7 * torch.sin(7 * theta)
    A[..., 20] = 9 ** .5 * (70 * rou ** 8 - 140 * rou ** 6 + 90 * rou ** 4 - 20 * rou ** 2 + 1)  # Z60

    B = A.reshape(256**2,1,21)
    c = []
    for i in range(21):
        c.append(torch.masked_select(B[...,i],mask))

    matrix = torch.stack(c,dim=1)
    matrix = matrix.repeat(BS,1,1)
    mask = mask.repeat(BS,1,1)
    A = A.repeat(BS,1,1,1)
    rou = rou.repeat(BS,1,1)
    return matrix,mask,A,rou


def up_limit_of_zer(zerpath,fov):
    zer = pd.read_excel(zerpath, sheet_name='Sheet1', header=None, index_col=None)
    info = pd.read_excel(zerpath, sheet_name='Sheet2', header=None, index_col=None)
    zer = zer.drop(columns=[1, 4, 7, 9, 12, 14, 15, 17, 19, 22, 24, 26, 29, 31, 33, 35]).values
    fov_info = np.expand_dims(info.iloc[:, 3].values, 1)
    fov_info = np.repeat(fov_info,21,axis=1)
    mask = (fov_info <= fov)
    select_zer = torch.masked_select(torch.from_numpy(zer), torch.from_numpy(mask))
    select_zer = select_zer.reshape(-1, 21)
    max_zer,max_index = torch.max(select_zer,dim=0)
    max_zer = max_zer.to(dtype=torch.float32)
    max_zer = np.delete(max_zer, [4, 5, 7, 9, 12])
    return max_zer.float()

def wavefront_fitting(WF,matrix,mask,BS):
    data = torch.masked_select(WF.reshape(BS,256**2,1),mask)
    length = len(data)
    data = data.reshape(BS,length//BS,1)
    matrixT = matrix.transpose(1,2)
    B = torch.linalg.inv(torch.bmm(matrixT,matrix)  + 0.0001 * torch.eye(21))
    temp = torch.bmm(B,matrixT)
    zer_co = torch.bmm(temp,data)
    fmatrix = torch.bmm(matrix, zer_co)
    error = torch.sum(abs(fmatrix - data),dim=1) / length

    return zer_co,error

def zernike2wavefront(zer_co, M):
    x = torch.linspace(-1, 1, M)
    Y, X = torch.meshgrid(x, x, indexing='ij')
    rho = torch.abs(torch.sqrt(X ** 2 + Y ** 2))
    WF = torch.zeros(M, M)
    theta = torch.atan2(Y,X)
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
    zer_co = np.reshape(zer_co, (-1, 1))
    for i in range(21):
        WF = WF + A[:, :, i] * zer_co[i]
    WF = torch.where(rho >= 1, 0, WF)
    return WF