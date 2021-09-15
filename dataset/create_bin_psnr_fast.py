'''
Once bin files of DIV2K dataset has been generated
This script can create an index list sorted by psnr (ascending)
index is the (i,j)th patch
low psnr indicates difficult sample, which is more worth training
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
import time

def is_bin_file(filename, scale):
    return any(filename.endswith(ext) for ext in ['x{}.pt'.format(scale)])

def my_calc_psnr(dpm_patch, scale=2, rgb_range=255):
    shave = scale + 6
    valid = dpm_patch[shave:-shave, shave:-shave]
    # mse = valid.mean()
    mse = valid.sum()
    if mse == 0:
        return 1000
    else:
        return -10 * math.log10(mse)


def calc_psnr(sr, hr, scale=2, rgb_range=255, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()
    if mse == 0:
        return 1000
    else:
        return -10 * math.log10(mse)

if __name__ == '__main__':
    ################## parameters
    rgb_range = 255
    scale = 2
    lr_dir = '/data/shizun/dataset/DIV2K/bin/DIV2K_train_LR_bicubic/X{}/'.format(scale)
    hr_dir = '/data/shizun/dataset/DIV2K/bin/DIV2K_train_HR/'
    patch_size = 192 # the size is for hr patch
    
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
        if i > 2: break
        # if i < 598: continue

        print("[{}/{}] processing [{}]...".format(i, len(files), file))
        # get lr
        lr_file = os.path.join(lr_dir, file)
        with open(lr_file, 'rb') as _f:
            lr = pickle.load(_f) # (W,H,3)
            lr_tensor = torch.from_numpy(lr).float()

        # get hr
        hr_file = os.path.join(hr_dir, file.replace('x{}.pt'.format(scale), '.pt'))
        with open(hr_file, 'rb') as _f:
            hr = pickle.load(_f) # (W,H,3)
            # hr = torch.from_numpy(hr).float()

        # get sr
        sr_tensor = interpolate(
            lr_tensor.permute(2,0,1).unsqueeze(0), # (1,3,W,H)
            scale_factor=scale, 
            mode='bilinear',
            align_corners=False).clamp(min=0, max=255)
        sr_tensor = sr_tensor.squeeze().permute(1,2,0) # (W,H,3)
        sr = sr_tensor.numpy()

        print("lr shape:{} sr shape:{} hr shape:{}".format(lr.shape, sr.shape, hr.shape))

        # precompute diff-power map
        diff_norm = (sr - hr) / rgb_range
        diff_norm_pow = np.power(diff_norm, 2)
        dpm = np.sum(diff_norm_pow, axis=2)

        # sum speedup
        shave = scale + 6
        mn = (patch_size - shave*2) * (patch_size - shave*2)
        dpm = dpm / (mn * 3)  # channel = 3

        # sliding window
        index_psnr = []
        t = 0
        # for ix in range(lr.shape[1] - patch_size + 1):
        for ix in tqdm(range(lr.shape[1] - lr_patch_size + 1)): # W
            for iy in range(lr.shape[0] - lr_patch_size + 1):   # H
                # determine index
                tp = patch_size
                tx, ty = scale * ix, scale * iy

                dpm_patch = dpm[ty:ty + tp, tx:tx + tp]
                tic = time.time()
                # most time are consumed in "my_calc_psnr"
                psnr = my_calc_psnr(dpm_patch)
                toc = time.time()
                tt = toc - tic
                t = t + tt

                index_psnr.append([ix,iy,psnr])
        print("time: {:.3f}".format(t))
        index_psnr = sorted(index_psnr, key=lambda a: a[2]) # sorted by psnr (N,3)
        index_psnr = np.array(index_psnr)

        # save patch index sorted by psnr
        psnr_file = lr_file.replace(".pt","_psnr_fast.pt")
        with open(psnr_file, 'wb') as _f:
            pickle.dump(index_psnr, _f)