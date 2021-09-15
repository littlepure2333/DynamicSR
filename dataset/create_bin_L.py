import os
import numpy as np
import pickle
import imageio
import cv2
from scipy.io import loadmat


def is_image_file(filename):
    return any(filename.endswith(ext) for ext in ['.png', '.jpg', '.jpeg'])


if __name__ == '__main__':
    folder0 = 'DIV2K/HR'
    folder1 = 'DIV2K/LR_RK/X2'
    folder2 = 'DIV2K/bin/LR_RK+L/X2'
    kernel_PCA_file = 'PCA_P_RK_X2'

    os.makedirs(folder2, exist_ok=True)
    files = os.listdir(folder1)
    print('num of files:', len(files))
    P = loadmat(kernel_PCA_file)['P']  # (dim, k*k)
    print('kernel PCA size:', P.shape)

    for file in files:
        if is_image_file(file):
            file_name = os.path.splitext(file)[0]
            file_nameL = file_name[:4]+'.png'
            file_in = os.path.join(folder1, file)
            file_out = os.path.join(folder2, file_name + '.pt')
            file_L = os.path.join(folder0, file_nameL)
            # load image
            img = imageio.imread(file_in)
            imgL = imageio.imread(file_L)  # (h,w,c)
            img = img[:, :, :3] if img.shape[2] == 4 else img  # (h,w,c)
            # resize image
            oH,oW,oC = imgL.shape[0], imgL.shape[1], imgL.shape[2]  # (oH,oW,oC)
            imgR = cv2.resize(img,(oW,oH),interpolation=cv2.INTER_CUBIC)
            img = imgR
            # load kernel
            kernel = loadmat(os.path.join(folder1, file_name + '.mat'))['ker']
            kernel_vec = np.dot(P, np.reshape(kernel, (-1), order="F"))  # (dim, k*k)*(k*k) = dim
            dump_dict = {'img':img,'kernel':kernel_vec}
            print(file_in,file_out,dump_dict.keys())
            with open(file_out, 'wb') as _f:
                pickle.dump(dump_dict,_f)
