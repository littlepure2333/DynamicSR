'''
Once bin files of DIV2K dataset has been generated
This script can create an index list sorted by psnr (ascending)
index is the (u,v)th patch (u - y axis, v - x axis)
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
from numba import jit

def is_bin_file(filename, scale):
    return any(filename.endswith(ext) for ext in ['x{}.pt'.format(scale)])

# cumsum in Numba currently only supports the first argument. I.e. none of axis, dtype or out are implemented.
def box_filter(imSrc, patch_size):
    '''BOXFILTER   O(1) time box filtering using cumulative sum. 
    
    Definition imDst(x, y)=sum(sum(imSrc(x:x+r,y:y+r))). 
    Running time independent of r.

    Args:
        imSrc (np.array): source image, shape(hei,wid).
        patch_size (int): box filter size. (r)
    
    Returns:
        imDst (np.array): img after filtering, shape(hei-r+1,wid-r+1).
    '''
    [hei,wid] = imSrc.shape
    imDst = np.zeros_like(imSrc)

    # cumulative sum over Y axis
    imCum = np.cumsum(imSrc,axis=0)
    imDst[0,:] = imCum[patch_size-1,:]
    imDst[1:hei-patch_size+1,:] = imCum[patch_size:,:] - imCum[0:hei-patch_size,:]

    # cumulative sum over X axis
    imCum = np.cumsum(imDst,axis=1)
    imDst[:,0] = imCum[:,patch_size-1]
    imDst[:,1:wid-patch_size+1] = imCum[:,patch_size:] - imCum[:,0:wid-patch_size]

    # cut the desired area
    imDst = imDst[:hei-patch_size+1,:wid-patch_size+1]

    return imDst

@jit(nopython=True)
def cal_dp_map(diff_norm, patch_size):
    diff_norm_pow = np.power(diff_norm, 2)
    dpm = np.sum(diff_norm_pow, axis=2)
    mn = patch_size * patch_size
    dpm = dpm / (mn * 3)  # channel = 3
    return dpm

@jit(nopython=True)
def cal_psnr_map(sum_map,scale,eps):
    if mode == 'up': 
        sum_map = sum_map[::scale,::scale]
    sum_map = sum_map + eps # avoid zero value
    psnr_map = -10 * np.log10(sum_map)
    return psnr_map

@jit(nopython=True)
def psnr_sort(psnr_map, iy, ix):
    index_psnr = np.hstack((iy, ix, psnr_map.reshape(-1,1)))
    sort_index = np.argsort(index_psnr[:,-1])
    index_psnr = index_psnr[sort_index]
    return index_psnr

if __name__ == '__main__':
    ################## parameters
    eps = 1e-9
    rgb_range = 255
    scale = 4
    lr_dir = '/data/shizun/dataset/DIV2K/bin/DIV2K_train_LR_bicubic/X{}/'.format(scale)
    hr_dir = '/data/shizun/dataset/DIV2K/bin/DIV2K_train_HR/'
    hr_patch_size = 384 # the size is for hr patch
    # mode = "down"  # downsampling
    mode = "up"  # upsampling
    #################

    lr_patch_size = hr_patch_size // scale
    all_files = os.listdir(lr_dir)
    files = []
    for f in all_files:
        if is_bin_file(f,scale):
            files.append(f)
    files.sort()
    print('number of files:', len(files))
    t0 = time.time()
    for i, file in enumerate(files):
        # if i > 2: break
        # if i < 10: continue

        print("[{}/{}] processing [{}]...".format(i, len(files), file))
        t1 = time.time()
        # get lr
        lr_file = os.path.join(lr_dir, file)
        with open(lr_file, 'rb') as _f:
            lr = pickle.load(_f) # (W,H,3)
            lr_tensor = torch.from_numpy(lr).float()

        # get hr
        hr_file = os.path.join(hr_dir, file.replace('x{}.pt'.format(scale), '.pt'))
        with open(hr_file, 'rb') as _f:
            hr = pickle.load(_f) # (W,H,3)
            hr_tensor = torch.from_numpy(hr).float()

        # get sr
        if mode == 'up':
            sr_tensor = interpolate(
                lr_tensor.permute(2,0,1).unsqueeze(0), # (1,3,W,H)
                scale_factor=scale, 
                mode='bilinear',
                align_corners=False).clamp(min=0, max=255)
        elif mode == 'down':
            sr_tensor = interpolate(
                hr_tensor.permute(2,0,1).unsqueeze(0), # (1,3,W,H)
                scale_factor=1/scale, 
                mode='bilinear',
                align_corners=False).clamp(min=0, max=255)
        sr_tensor = sr_tensor.squeeze().permute(1,2,0) # (W,H,3)
        sr = sr_tensor.numpy()

        print("lr shape:{} sr shape:{} hr shape:{}".format(lr.shape, sr.shape, hr.shape))

        # make up digital error
        if (hr.shape[0] != sr.shape[0]) or (hr.shape[1] != sr.shape[1]):
            print("sr and hr shape mismatch! make up digital error...")
            hr = hr[:sr.shape[0],:sr.shape[1],:sr.shape[2]]
            print("lr shape:{} sr shape:{} hr shape:{}".format(lr.shape, sr.shape, hr.shape))

        # make shaves
        if mode == 'up':
            shave = scale + 6
            patch_size = hr_patch_size - shave*2
        elif mode == 'down':
            shave = (scale + 6) // scale
            patch_size = lr_patch_size - shave*2

        # precompute diff-power map
        if mode == 'up':
            diff_norm = (sr - hr) / rgb_range
        elif mode == 'down':
            diff_norm = (sr - lr) / rgb_range
        diff_norm = diff_norm[shave:-shave, shave:-shave, ...]

        dpm = cal_dp_map(diff_norm, patch_size)

        # box filtering
        sum_map = box_filter(dpm,patch_size)

        # calculate psnr map
        psnr_map = cal_psnr_map(sum_map,scale,eps)
        [hei, wid] = psnr_map.shape

        # generate index
        iy = np.arange(hei).reshape(-1,1).repeat(wid,axis=1).reshape(-1,1)
        ix = np.arange(wid).reshape(1,-1).repeat(hei,axis=0).reshape(-1,1)

        # sort index by psnr
        index_psnr = psnr_sort(psnr_map, iy, ix)

        # save patch index sorted by psnr
        psnr_file = lr_file.replace(".pt","_psnr_{}_p{}.pt".format(mode, hr_patch_size))
        print("saving {}".format(psnr_file))
        with open(psnr_file, 'wb') as _f:
            pickle.dump(index_psnr, _f)
        t5 = time.time()
        
        tt = t5 - t1
        print("process time: {:.5f}".format(tt))
    print("total time: {:.5f}".format(t5-t0))