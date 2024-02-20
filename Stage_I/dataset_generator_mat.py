import h5py
import os
import glob
import numpy as np
import scipy.io as scio

def gen_dataset(src_input_files, src_label_files, dst_path):

    length = len(src_input_files)
    all = np.linspace(0, length-1, length)
    index1 = np.random.choice(all, int(0.9*length), replace=False)
    index1 = index1.astype('int32')
    H = np.in1d(all, index1)
    B = np.where(H, np.nan, all)
    other = B[~np.isnan(B)]
    index2 = other
    index2 = index2.astype('int32')

    h5py_path = os.path.join(dst_path, "train_psf_wavefront.h5")
    h5f = h5py.File(h5py_path, 'w')
    i = 1
    for img_idx in index1:
        print("Now processing img pairs of %s", os.path.basename(src_input_files[img_idx]))
        # print(i)
        img_input = scio.loadmat(src_input_files[img_idx])
        img_label = scio.loadmat(src_label_files[img_idx])
        img_input_pair = np.zeros((257, 256, 1))
        img_input_pair[0:256,:,:] = img_input['PSF'].reshape([256,256,1])
        img_label1 = img_label['wavefront'].reshape([256,256,1])
        img_label2 = img_label['zernike'].reshape([1,21,1])
        img_label_pair = np.zeros((257, 256, 1))
        img_label_pair[0:256,:,:] = img_label1
        img_label_pair[256,0:21,:] = img_label2
        # concate input and label together
        img_pair = np.concatenate([img_input_pair, img_label_pair], 2)
        # save into h5py file
        h5f.create_dataset(str(img_idx), data=img_pair)
        i=i+1
    h5f.close()

    h5py_path = os.path.join(dst_path, "valid_psf_wavefront.h5")
    h5f = h5py.File(h5py_path, 'w')
    i=0
    for img_idx in index2:
        print("Now processing img pairs of %s", os.path.basename(src_input_files[img_idx]))
        # print(i)
        img_input = scio.loadmat(src_input_files[img_idx])
        img_label = scio.loadmat(src_label_files[img_idx])
        img_input_pair = np.zeros((257, 256, 1))
        img_input_pair[0:256, :, :] = img_input['PSF'].reshape([256, 256, 1])
        img_label1 = img_label['wavefront'].reshape([256, 256, 1])
        img_label2 = img_label['zernike'].reshape([1, 21, 1])
        img_label_pair = np.zeros((257, 256, 1))
        img_label_pair[0:256, :, :] = img_label1
        img_label_pair[256, 0:21, :] = img_label2
        # concate input and label together
        img_pair = np.concatenate([img_input_pair, img_label_pair], 2)
        # save into h5py file
        h5f.create_dataset(str(img_idx), data=img_pair)
        i = i+1
    h5f.close()


if __name__ == "__main__":
    src_input_path = './mat/PSF'
    src_label_path = './mat/Wavefront'

    dst_path = './h5py'

    src_input_files = sorted(glob.glob(src_input_path + "/*.mat"))
    src_label_files = sorted(glob.glob(src_label_path + "/*.mat"))

    print("start dataset generation!")
    gen_dataset(src_input_files, src_label_files, dst_path)