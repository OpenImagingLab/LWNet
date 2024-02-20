import glob
import numpy as np
import torch
import random
import h5py
import time
import torch.utils.data as data

class Dataset_from_h5(data.Dataset):

    def __init__(self, src_path, train=True):
        self.path = src_path
        self.if_train = train
        h5f = h5py.File(self.path, 'r')
        self.keys = list(h5f.keys())
        if self.if_train:
            random.shuffle(self.keys)
        h5f.close()

    def __getitem__(self, index):
        h5f = h5py.File(self.path, 'r')
        key = self.keys[index]
        data = np.array(h5f[key]).reshape(h5f[key].shape)
        h5f.close()

        input = data[:, :, 0]
        label = data[:, :, 1]
        input = input.reshape([257,256,1])
        input1 = np.delete(input,256,axis=0)
        label = label.reshape([257, 256, 1])
        zernike = label[256, 0:21, 0]
        label1 = np.delete(label, 256, axis=0)       
        input1 = torch.from_numpy(np.ascontiguousarray(np.transpose(input1, (2, 0, 1)))).float()
        label1 = torch.from_numpy(np.ascontiguousarray(np.transpose(label1, (2, 0, 1)))).float()

        # return input1, label1,zernike
        return input1, label1

    def __len__(self):
        return len(self.keys)



