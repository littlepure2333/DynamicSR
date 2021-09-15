'''
Once bin files of DIV2K dataset has been generated
This script can create an index list sorted by std (descending)
index is the (i,j)th patch
high std may indicates difficult sample, which is more worth training
'''

import os
import math
import numpy as np
import pickle
import imageio
import torch
from tqdm import tqdm
from torch.nn.functional import interpolate
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from data.utils_image import rgb2ycbcr


def is_bin_file(filename, scale):
    return any(filename.endswith(ext) for ext in ['x{}.pt'.format(scale)])

def calc_std(patch, mode=0):
    if mode == 0:
        std = np.std(patch)
        return std
    elif mode == 1:
        std0 = np.std(patch[:,:,0])
        std1 = np.std(patch[:,:,1])
        std2 = np.std(patch[:,:,2])
        std = (std0 + std1 + std2) / 3
        return std
    elif mode == 2: #y
        std0 = np.std(patch[:,:,0])
        return std0

if __name__ == '__main__':
    ################## parameters
    scale = 2
    lr_dir = '/data/shizun/dataset/DIV2K/bin/DIV2K_train_LR_bicubic/X{}/'.format(scale)
    hr_dir = '/data/shizun/dataset/DIV2K/bin/DIV2K_train_HR/'
    patch_size = 192 # the size is for hr patch
    # patch_size = 64 # the size is for hr patch
    std_mode = 1
    color_space = "rgb"
    # color_space = "ycbcr"
    #################

    lr_patch_size = patch_size // scale
    all_files = os.listdir(lr_dir)
    files = []
    for f in all_files:
        if is_bin_file(f,scale):
            files.append(f)
    files.sort()
    print('number of files:', len(files))
    for i, file in enumerate(files):
        # if i < 730: continue

        print("[{}/{}] processing [{}]...".format(i, len(files), file))
        # get lr
        lr_file = os.path.join(lr_dir, file)
        with open(lr_file, 'rb') as _f:
            lr = pickle.load(_f) # (W,H,3)
            if color_space == "ycbcr":
                lr = rgb2ycbcr(lr,only_y=False)
            # lr = torch.from_numpy(lr).float()

        print("lr shape:{}, std mode: {}, color space: {}, patch size: {}".format(lr.shape,std_mode, color_space, patch_size))

        # sliding window
        index_std = []
        for ix in tqdm(range(lr.shape[1] - lr_patch_size + 1)): # W
            for iy in range(lr.shape[0] - lr_patch_size + 1):   # H
                ip = lr_patch_size
                lr_patch = lr[iy:iy + ip, ix:ix + ip, :]
                lr_patch_std = calc_std(lr_patch, mode=std_mode)
                
                index_std.append([iy,ix,lr_patch_std])
        # Descending order, higher std, more important
        index_std = sorted(index_std, key=lambda a: a[2], reverse=True) # sorted by psnr (N,3)
        index_std = np.array(index_std)

        # save patch index sorted by psnr
        psnr_file = lr_file.replace(".pt","_std{}_{}_p{}_new.pt".format(std_mode, color_space, patch_size))
        # psnr_file = lr_file.replace(".pt","_std{}_{}.pt".format(std_mode, color_space))
        # psnr_file = lr_file.replace(".pt","_std{}.pt".format(std_mode))
        # psnr_file = lr_file.replace(".pt","_std.pt")
        with open(psnr_file, 'wb') as _f:
            pickle.dump(index_std, _f)
        print("saved {}".format(psnr_file))