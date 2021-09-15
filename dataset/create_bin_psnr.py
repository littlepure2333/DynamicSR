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


def is_bin_file(filename, scale):
    return any(filename.endswith(ext) for ext in ['x{}.pt'.format(scale)])

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

def get_patch_index(*args, patch_size=96, scale=2, multi=False, input_large=False):
    ih, iw = args[0].shape[:2]

    if not input_large:
        p = scale if multi else 1
        tp = p * patch_size
        ip = tp // scale
    else:
        tp = patch_size
        ip = patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    return ret

if __name__ == '__main__':
    ################## parameters
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
        # if i < 598: continue

        print("[{}/{}] processing [{}]...".format(i, len(files), file))
        # get lr
        lr_file = os.path.join(lr_dir, file)
        with open(lr_file, 'rb') as _f:
            lr = pickle.load(_f) # (W,H,3)
            lr = torch.from_numpy(lr).float()

        # get hr
        hr_file = os.path.join(hr_dir, file.replace('x{}.pt'.format(scale), '.pt'))
        with open(hr_file, 'rb') as _f:
            hr = pickle.load(_f) # (W,H,3)
            hr = torch.from_numpy(hr).float()

        # get sr
        sr = interpolate(
            lr.permute(2,0,1).unsqueeze(0), # (1,3,W,H)
            scale_factor=scale, 
            mode='bilinear',
            align_corners=False).clamp(min=0, max=255)
        sr = sr.squeeze().permute(1,2,0) # (W,H,3)

        print("lr shape:{} sr shape:{} hr shape:{}".format(lr.shape, sr.shape, hr.shape))

        # sliding window
        index_psnr = []
        # for ix in range(lr.shape[1] - patch_size + 1):
        for ix in tqdm(range(lr.shape[1] - lr_patch_size + 1)): # W
            for iy in range(lr.shape[0] - lr_patch_size + 1):   # H
                tp = patch_size
                tx, ty = scale * ix, scale * iy
                hr_patch = hr[ty:ty + tp, tx:tx + tp, :] # (W,H,3)
                sr_patch = sr[ty:ty + tp, tx:tx + tp, :] # (W,H,3)
                hr_patch = hr_patch.permute(2,0,1) # (3,W,H)
                sr_patch = sr_patch.permute(2,0,1) # (3,W,H)

                try:
                    psnr = calc_psnr(sr_patch, hr_patch, scale)
                except Exception as e:
                    print("{}:{},{}:{}".format(ty, ty + tp, tx, tx + tp))
                    print(hr_patch.shape)
                    print(sr_patch.shape)
                    raise e
                
                index_psnr.append([ix,iy,psnr])
        # Ascending order, lower psnr, more important
        index_psnr = sorted(index_psnr, key=lambda a: a[2], reverse=False) # sorted by psnr (N,3)
        index_psnr = np.array(index_psnr)

        # save patch index sorted by psnr
        psnr_file = lr_file.replace(".pt","_psnr.pt")
        with open(psnr_file, 'wb') as _f:
            pickle.dump(index_psnr, _f)