import os
import numpy as np
import pickle
import imageio


def is_image_file(filename):
    return any(filename.endswith(ext) for ext in ['.png', '.jpg', '.jpeg'])


if __name__ == '__main__':
    folder1 = 'DIV2K/LR_RKsk/X4'
    folder2 = 'DIV2K/bin/LR_RKsk/X4'
    os.makedirs(folder2, exist_ok=True)
    files = os.listdir(folder1)
    print('num of files:', len(files))
    for file in files:
        if is_image_file(file):
            file_name = os.path.splitext(file)[0]
            file_in = os.path.join(folder1, file)
            file_out = os.path.join(folder2, file_name + '.pt')
            print(file_in, file_out)
            img = imageio.imread(file_in)
            img = img[:, :, :3] if img.shape[2] == 4 else img  # remove alpha channel
            with open(file_out, 'wb') as _f:
                pickle.dump(img, _f)
