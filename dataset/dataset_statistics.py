'''
Once bin files of high frequency index list has been generated
This script can do the statistics of the dataset
'''

import os
import pickle
import numpy as np
from matplotlib import pyplot as plt 
# os.sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import glob

if __name__ == '__main__':
    ################## parameters
    scale = 2
    hr_patch_size = 192 # the size is for hr patch
    lr_dir = '/data/shizun/DIV2K/bin/DIV2K_train_LR_bicubic/X{}/'.format(scale)
    suffix = "_Canny_p192_s24.pt"
    #################

    files = glob.glob(lr_dir+'*x{}'.format(scale)+suffix)
    files.sort()
    files = files[-10:]
    print('number of files:', len(files))

    id_iy_ix_metric_list = []

    for f in files:
        name = os.path.basename(f)
        index = int(name.split('x')[0])

        with open(f, 'rb') as _f:
            iy_ix_metric = pickle.load(_f) # [iy, ix, metric]
        
        id = np.array([index for _ in range(iy_ix_metric.shape[0])]).reshape(-1,1)
        id_iy_ix_metric = np.hstack((id, iy_ix_metric))
        id_iy_ix_metric_list.append(id_iy_ix_metric)


    all_id_iy_ix_metric = np.vstack(id_iy_ix_metric_list)


    # index_value = np.hstack((iy, ix, value_map.reshape(-1,1)))
    sort_index = np.argsort(all_id_iy_ix_metric[:,-1])
    all_id_iy_ix_metric = all_id_iy_ix_metric[sort_index] # ascending, from easy to hard


    # save patch index sorted by psnr
    save_file = lr_dir+'statistics_val'+suffix
    # print("saving {}".format(save_file))
    with open(save_file, 'wb') as _f:
        pickle.dump(all_id_iy_ix_metric, _f)
        
    metric = all_id_iy_ix_metric[:,-1]
    metric = metric[metric>0]
    lr_patch_size = hr_patch_size / scale
    metric = metric / lr_patch_size
    # metric = np.log(metric+1)
    # hist, bin_edges = np.histogram(metric, bins=32)
    
    plt.hist(metric, bins=10) 
    plt.title("histogram") 
    plt.xlabel('metric')
    plt.ylabel('frequency')
    # plt.show()
    plt.savefig("dataset/histogram_x{}{}.png".format(scale, suffix.split('.')[0]),dpi=300)