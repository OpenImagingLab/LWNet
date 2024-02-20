# --coding:utf-8--
import pandas as pd
import numpy as np
import math
from torch.fft import fftshift, fft2
from function_for_PR2 import *
import matplotlib.pyplot as plt
import os
import scipy.io as scio

M = 256
Filestr = './Lens_Zernike_Lib.xlsx'
PSFstr = './mat/PSF'
wfstr = './mat/Wavefront'
Datatemp = pd.read_excel(Filestr,sheet_name='Sheet1',header=None, index_col=None)
filename = pd.read_excel(Filestr,sheet_name='Sheet2',header=None, index_col=None)
Datatemp = Datatemp.drop(columns=[1,4,7,9,12,14,15,17,19,22,24,26,29,31,33,35])
Data = np.array(Datatemp, dtype=np.float64)
length = Data.shape[0]
index = 0
sorted_zer = np.zeros([length,21])
sorted_filename = pd.DataFrame([])
x = np.linspace(-1,1,M)
X,Y = np.meshgrid(x,x)


read_index = 0

while read_index < length:
    WF = zernike_wavefront(Data[read_index,:], M)
    W =nn.ZeroPad2d(2*M)(WF)
    phase = torch.exp(1j * 2 * torch.pi * W)
    phase=torch.where(phase==1,0,phase)
    AP = abs(fftshift(fft2(phase))) ** 2
    AP = AP / torch.max(AP)
    H = torchvision.transforms.CenterCrop(M)
    PSF =H(AP)
    pv = torch.max(WF)- torch.min(WF)

    if sum(sum(PSF))>0.985*sum(sum(AP)):
        print(read_index,filename.iloc[read_index,0][0:-4])
        PSFdir = PSFstr + '/'+ filename.iloc[read_index,0][0:-4]+'.mat'
        wfdir = wfstr + '/' + filename.iloc[read_index,0][0:-4]+'.mat'
        scio.savemat(wfdir, {'wavefront': WF.numpy(),'zernike': Data[read_index,:],'filename':filename.iloc[read_index,1],'wavelenth':filename.iloc[read_index,2],'fov':filename.iloc[read_index,3],'pv':pv.numpy()})
        scio.savemat(PSFdir, {'PSF': PSF.numpy(),'zernike': Data[read_index,:],'filename':filename.iloc[read_index,1],'wavelenth':filename.iloc[read_index,2],'fov':filename.iloc[read_index,3]})

    else:
        print('PSF is too big:',filename.iloc[read_index,0])
        # plt.subplot(121)
        # plt.imshow(PSF)
        # plt.subplot(122)
        # plt.imshow(AP)
        # plt.show()
        read_index = read_index + 1
        continue

    read_index = read_index +1






