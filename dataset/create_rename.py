import os
import os.path as osp
import numpy as np
import shutil


folder = 'DIV2K/LR_bicubic/X3/'
names = os.listdir(folder)
print(len(names))

for name in names:
    print(name)
    src = osp.join(folder,name)
    basename = osp.splitext(name)[0]
    dst = osp.join(folder,'%sx3.png' % basename)
    shutil.move(src,dst)
    


