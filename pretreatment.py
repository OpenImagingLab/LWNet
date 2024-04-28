# --coding:utf-8--
import numpy as np
import torch
from model.stage_I.UNet import *
from utils_optics import *
from torch.nn import functional as F
# from torchvision.transforms import CenterCrop, Resize
import matplotlib.pyplot as plt



def precondition(gt):
    if isinstance(gt,np.ndarray):
        gt = torch.from_numpy(gt)
    if len(gt.shape) == 2:
        gt = gt.unsqueeze(0).unsqueeze(0)
    model = UNet()
    model_path = '.\model\stage_I\model_0601_dict_0.1TV.pth'
    A = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(A)
    matrix, mask, A, _ = fitting_prepare(1)
    zer_co_arr = []
    errorarr = []
    outputarr = []
    x = torch.linspace(-1, 1, 256)
    [Y, X] = torch.meshgrid(x, x, indexing='ij')
    rou = abs(torch.sqrt(X ** 2 + Y ** 2))
    M = 10
    for shift_y in range(M):
        trans = torch.tensor([[1.0,0.0,0.0],[0.0,1.0,0.01*(shift_y-M/2)]]).unsqueeze(0)
        grid = F.affine_grid(trans,gt.shape,align_corners=True)
        shift_gt = F.grid_sample(gt,grid)
        output = model(shift_gt)
        outputarr.append(np.where(rou>=1,0,output.squeeze().detach().numpy()))
        zer_co, error = wavefront_fitting(output, matrix, mask, 1)
        zer_co_arr.append(zer_co.detach().numpy())
        errorarr.append(error.detach().numpy())
    error_arr = np.array(errorarr).reshape(M)
    zer_arr = np.array(zer_co_arr).reshape(M,21)
    index = np.argmin(error_arr)
    return zer_arr[index,:],outputarr[index]


def sample(gt,initial, num=50):
    gt = gt.squeeze()
    if isinstance(initial,torch.Tensor):
        initial = initial.detach().numpy()
    np.random.seed(0)
    lowbound1 = np.where(initial[0:10]>0,0,1.2*initial[0:10])
    upbound1  = np.where(initial[0:10]<0,0,1.2*initial[0:10])
    lowbound2 = np.where(initial[10:]>0,0,initial[10:])
    upbound2  = np.where(initial[10:]<0,0,initial[10:])
    lowbound = np.hstack((lowbound1,lowbound2))
    upbound = np.hstack((upbound1,upbound2))
    limit = np.stack([upbound,lowbound],axis=1)
    limit = np.sort(limit,axis=1)
    ub,lb = limit[:,1], limit[:,0]
    pos = (lb + np.random.random((num, 21))* (ub - lb))
    pbestfit = np.zeros(num)

    for i in range(num):
        WF = zernike2wavefront(pos[i,:], 256)
        psf = FourierTransform(WF)
        pbestfit[i] = sum(sum((psf - gt)**2))

    index = np.argmin(pbestfit)
    gBest, gBestfit = pos[index,:], pbestfit[index]

    return torch.from_numpy(gBest).to(torch.float32)

