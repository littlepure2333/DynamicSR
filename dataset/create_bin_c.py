import os
import numpy as np
import pickle
import imageio
from scipy.io import loadmat


def is_image_file(filename):
    return any(filename.endswith(ext) for ext in ['.png', '.jpg', '.jpeg'])


if __name__ == '__main__':
    folder1 = 'DIV2K/LR_RK/X2.00'
    folder2 = 'DIV2K/bin/LR_RK/X2.00'
    kernel_PCA_file = 'PCA_P_RK_X2'

    os.makedirs(folder2, exist_ok=True)
    files = os.listdir(folder1)
    print('num of files:', len(files))
    P = loadmat(kernel_PCA_file)['P']  # (dim, k*k)
    print('kernel PCA size:', P.shape)

    for file in files:
        if is_image_file(file):
            file_name = os.path.splitext(file)[0]
            file_in = os.path.join(folder1, file)
            file_out = os.path.join(folder2, file_name + '.pt')
            # load image
            img = imageio.imread(file_in)
            img = img[:, :, :3] if img.shape[2] == 4 else img  # (h,w,c)
            # load kernel
            # kernel = loadmat(os.path.join(folder1, file_name + '.mat'))['ker']
            # kernel_vec = np.dot(P, np.reshape(kernel, (-1), order="F"))  # (dim, k*k)*(k*k) = dim
            kernel_vec = np.zeros((P.shape[0], 1), dtype=np.float)
            kernel_vec = np.reshape(kernel_vec, (-1))
            dump_dict = {'img': img, 'kernel': kernel_vec}

            # kernel_vec = kernel_vec.reshape(1, 1, -1).astype(np.float)  # (1,1,dim)
            # kernel_map = np.tile(kernel_vec, (img.shape[0], img.shape[1], 1))  # (1,1,dim)->(h,w,dim)
            # img = np.concatenate((img, kernel_map), axis=2)  # (h,w,dim+c)
            # print(file_in,file_out, img.shape)
            # with open(file_out, 'wb') as _f:
            #     pickle.dump(img, _f)

            print(file_in, file_out, dump_dict.keys())
            with open(file_out, 'wb') as _f:
                pickle.dump(dump_dict, _f)
