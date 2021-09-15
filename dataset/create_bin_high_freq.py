'''
Once bin files of DIV2K dataset has been generated
This script can create an index list sorted by high-frequency (descending).
Calculating high-frequency by Canny, Sobel, or Laplacian.
index is the (u,v)th patch (u - y axis, v - x axis)
high high-frequency indicates difficult sample, which is more worth training
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
import cv2
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

# @jit(nopython=True)
def value_sort(value_map, iy, ix):
    index_value = np.hstack((iy, ix, value_map.reshape(-1,1)))
    sort_index = np.argsort(index_value[:,-1])
    index_value = index_value[sort_index]
    return index_value

if __name__ == '__main__':
    ################## parameters
    eps = 1e-9
    rgb_range = 255
    scale = 2
    lr_dir = '/data/shizun/dataset/DIV2K/bin/DIV2K_train_LR_bicubic/X{}/'.format(scale)
    hr_dir = '/data/shizun/dataset/DIV2K/bin/DIV2K_train_HR/'
    hr_patch_size = 192 # the size is for hr patch
    # filter = 'Canny'
    # filter = 'Sobel'
    filter = 'Laplacian'
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
            lr_gray = cv2.cvtColor(lr,cv2.COLOR_RGB2GRAY)

        print("lr shape:{}".format(lr.shape))

        if filter == 'Canny':
            lr_filtered = cv2.Canny(lr_gray, 100, 150)
        elif filter == 'Sobel':
            x = cv2.Sobel(lr_gray, cv2.CV_16S, 1, 0)
            y = cv2.Sobel(lr_gray, cv2.CV_16S, 0, 1)

            absX = cv2.convertScaleAbs(x)  # 转回uint8
            absY = cv2.convertScaleAbs(y)

            lr_filtered = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        elif filter == 'Laplacian':
            gray_lap = cv2.Laplacian(lr_gray,cv2.CV_16S,ksize = 3)
            lr_filtered = cv2.convertScaleAbs(gray_lap)

        # box filtering
        lr_filtered_map = box_filter(lr_filtered, lr_patch_size)

        [hei, wid] = lr_filtered_map.shape

        # generate index
        iy = np.arange(hei).reshape(-1,1).repeat(wid,axis=1).reshape(-1,1)
        ix = np.arange(wid).reshape(1,-1).repeat(hei,axis=0).reshape(-1,1)

        # sort index by value
        index_value = value_sort(lr_filtered_map, iy, ix)

        # save patch index sorted by psnr
        save_file = lr_file.replace(".pt","_{}_p{}.pt".format(filter, hr_patch_size))
        print("saving {}".format(save_file))
        with open(save_file, 'wb') as _f:
            pickle.dump(index_value, _f)
        t5 = time.time()
        
        tt = t5 - t1
        print("process time: {:.5f}".format(tt))
    print("total time: {:.5f}".format(t5-t0))