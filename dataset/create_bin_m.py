import os
import numpy as np
import pickle
import imageio
from scipy.io import loadmat


def is_image_file(filename):
    return any(filename.endswith(ext) for ext in ['.png', '.jpg', '.jpeg'])


scales = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
          3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0]

folder_png = 'Urban100/LR_bicubic'
folder_bin = 'Urban100/bin/LR_bicubic'
kernel_PCA_file = 'PCA_P_RK_X2'
P = loadmat(kernel_PCA_file)['P']  # (dim, k*k)
print('kernel PCA size:', P.shape)


def process_scale(scale):
    folder1 = '%s/X%.2f' % (folder_png, scale)
    folder2 = '%s/X%.2f' % (folder_bin, scale)
    print(folder1, folder2)
    os.makedirs(folder2, exist_ok=True)
    files = os.listdir(folder1)
    print('num of files:', len(files))

    for file in files:
        if is_image_file(file):
            file_name = os.path.splitext(file)[0]
            file_in = os.path.join(folder1, file)
            file_out = os.path.join(folder2, file_name + '.pt')
            # load image
            img = imageio.imread(file_in)
            img = np.stack((img,)*3, axis=-1) if img.ndim == 2 else img  # (h,w) -> (h,w,c)
            img = img[:, :, :3] if img.shape[2] == 4 else img  # (h,w,c)
            # load kernel
            kernel_vec = np.zeros((P.shape[0], 1), dtype=np.float)
            kernel_vec = np.reshape(kernel_vec, (-1))
            dump_dict = {'img': img, 'kernel': kernel_vec}
            print(file_in, file_out, dump_dict.keys())
            with open(file_out, 'wb') as _f:
                pickle.dump(dump_dict, _f)


if __name__ == '__main__':
    for scale in scales:
        process_scale(scale=scale)
