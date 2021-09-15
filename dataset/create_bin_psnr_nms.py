'''
This script should be used after generating psnr_map by "create_bin_psnr_map.py" script.
This script can create a sparse index_psnr list (ascending) by by throwing darts.
index is the (u,v)th patch (u - y axis, v - x axis)
low psnr indicates difficult sample, which is more worth training
'''

import os
import time
import pickle
import numpy as np

scale = 2
patch_size_hr = 192
patch_size_lr = patch_size_hr // scale
threshold = 0.001
n_nms = 1000
new_suffix = "_psnr_nms_1000.pt"

def is_bin_file(filename, scale):
    return any(filename.endswith(ext) for ext in ['x{}_psnr_map_up.pt'.format(scale)])

def reverse_psnr_map(psnr_map_file):
    with open(psnr_map_file, 'rb') as _f:
        heatmap = pickle.load(_f)
    # print(heatmap.shape)
    # reverse the map, the smaller value, the bigger
    heatmap = 1/heatmap

    return heatmap

def gen_nms_mask(patch_size_lr, threshold):
    # generate nms mask
    nms_mask = np.ones((patch_size_lr*2+1, patch_size_lr*2+1))
    center = (patch_size_lr,patch_size_lr)
    for iy in range(nms_mask.shape[0]):
        for ix in range(nms_mask.shape[1]):
            I = np.abs((iy - center[0]) * (ix - center[1])) # intersection
            U = 2*np.power(patch_size_lr,2) - I       # Union
            iou = I/U
            # print(iou)
            if iou < threshold:
                nms_mask[iy][ix] = 0
    return nms_mask

def gen_psnr_nms(heatmap, n_nms):
    selected_list = []
    for i in range(n_nms):
        selected = np.unravel_index(heatmap.argmax(), heatmap.shape)
        x1 = max(0, selected[1]-patch_size_lr)
        x2 = min(heatmap.shape[1], selected[1] + patch_size_lr+1)
        y1 = max(0, selected[0]-patch_size_lr)
        y2 = min(heatmap.shape[0], selected[0] + patch_size_lr+1)
        # print(x1,x2,y1,y2)
        # print(heatmap[y1:y2,x1:x2].shape)
        x3 = max(0, patch_size_lr - selected[1])
        x4 = min(patch_size_lr*2+1, heatmap.shape[1]-selected[1]+patch_size_lr)
        y3 = max(0, patch_size_lr - selected[0])
        y4 = min(patch_size_lr*2+1, heatmap.shape[0]-selected[0]+patch_size_lr)
        # print(x3,x4,y3,y4)
        # print(nms_mask[y3:y4,x3:x4].shape)
        heatmap[y1:y2,x1:x2] = nms_mask[y3:y4,x3:x4]*heatmap[y1:y2,x1:x2]

        selected_list.append(selected)
        # break
    selected_list = np.array(selected_list)
    return selected_list

if __name__ == '__main__':
    psnr_map_dir = "/data/shizun/dataset/DIV2K/bin/DIV2K_train_LR_bicubic/X{}/".format(scale)
    all_files = os.listdir(psnr_map_dir)
    files = []
    for f in all_files:
        if is_bin_file(f,scale):
            files.append(f)
    files.sort()
    print('number of files:', len(files))

    # generate nms mask
    nms_mask = gen_nms_mask(patch_size_lr, threshold)

    for i, file in enumerate(files):
        # if i > 3: break
        # if i < 598: continue
        tic = time.time()

        print("[{}/{}] processing [{}]...".format(i, len(files), file))
        psnr_map_file = os.path.join(psnr_map_dir, file)

        # load psnr map and reverse it
        heatmap = reverse_psnr_map(psnr_map_file)

        # generate index_psnr list sparsed by nms
        selected_list = gen_psnr_nms(heatmap, n_nms)

        psnr_file = psnr_map_file.replace("_psnr_map_up.pt", new_suffix)
        print("saving {}".format(psnr_file))
        with open(psnr_file, 'wb') as _f:
            pickle.dump(selected_list, _f)
        toc = time.time()
        print("time: {}".format(toc - tic))