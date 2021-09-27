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

def is_bin_file(filename, scale, filter, patch_size):
    return any(filename.endswith(ext) for ext in ['x{}_{}_p{}.pt'.format(scale, filter, patch_size)])

if __name__ == '__main__':
    ################## parameters
    eps = 1e-9
    rgb_range = 255
    scale = 2
    lr_dir = '/data/shizun/DIV2K/bin/DIV2K_train_LR_bicubic/X{}/'.format(scale)
    hr_patch_size = 192 # the size is for hr patch
    data_range = '1-800/801-810' # train/test data range
    # filter = 'Canny'
    # filter = 'Sobel'
    filter = 'Laplacian'
    #################
    lr_patch_size = hr_patch_size // scale
    print("lr patch size:{}".format(lr_patch_size))
    all_files = os.listdir(lr_dir)
    files = []
    for f in all_files:
        if is_bin_file(f,scale,filter,hr_patch_size):
            files.append(f)
    files.sort()
    print('number of files:', len(files))

    # specify data range
    data_range = [r.split('-') for r in data_range.split('/')]
    for i, stage in enumerate(['train', 'val']):
        begin, end = list(map(lambda x: int(x), data_range[i]))
        files_stage = files[begin - 1:end]
        print("Generating {} stage metric list: {}-{}".format(stage, begin, end))

        total_psnr = []
        total_img_iy_ix = []

        t0 = time.time()

        # build image_no - iy - ix - psnr list of whole dataset
        pbar = tqdm(files_stage)
        for i, file in enumerate(pbar):
            pbar.set_description("Loading [{}]".format(file))
            img_no = int(file.split("x")[0])
            file_path = os.path.join(lr_dir, file)
            with open(file_path, 'rb') as _f:
                iy_ix_psnr = pickle.load(_f)
            img_repeat = np.array(img_no).repeat(iy_ix_psnr.shape[0]).reshape(-1,1)
            img_iy_ix = np.hstack((img_repeat,iy_ix_psnr[:,:2]))
            psnr = iy_ix_psnr[:,-1].reshape(-1,1)
            total_img_iy_ix.append(img_iy_ix)
            total_psnr.append(psnr)
            # print(img_iy_ix_psnr[0])
        
        print("building total list...")
        total_img_iy_ix = np.vstack(total_img_iy_ix).astype(np.int16)
        total_psnr = np.vstack(total_psnr)
        
        # sort by psnr (descending)
        print("sorting by psnr (descending)...")
        sort_index = np.argsort(-total_psnr[:,-1])
        total_img_iy_ix = total_img_iy_ix[sort_index]
        total_len = total_img_iy_ix.shape[0]

        # save sorted list
        save_file = os.path.join(lr_dir,"total_x{}_descending_{}.pt".format(scale, stage))
        print("saving {}".format(save_file))
        with open(save_file, 'wb') as _f:
            pickle.dump(total_img_iy_ix, _f)

        save_file = os.path.join(lr_dir,"hard_x{}_descending_{}.pt".format(scale, stage))
        print("saving {}".format(save_file))
        with open(save_file, 'wb') as _f:
            pickle.dump(total_img_iy_ix[:total_len//3], _f)

        save_file = os.path.join(lr_dir,"midd_x{}_descending_{}.pt".format(scale, stage))
        print("saving {}".format(save_file))
        with open(save_file, 'wb') as _f:
            pickle.dump(total_img_iy_ix[total_len//3:total_len*2//3], _f)

        save_file = os.path.join(lr_dir,"easy_x{}_descending_{}.pt".format(scale, stage))
        print("saving {}".format(save_file))
        with open(save_file, 'wb') as _f:
            pickle.dump(total_img_iy_ix[total_len*2//3:], _f)

        # report total processing time
        t1 = time.time()
        print("total time: {:.5f} s".format(t1-t0)) 

