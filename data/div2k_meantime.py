import os
import random
import pickle
import imageio
from tqdm.std import trange
from data import srdata
from data import common
from augments import cutblur

class DIV2K_MEANTIME(srdata.SRData):
    def __init__(self, args, name='DIV2K', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        self.data_partion = args.data_partion
        self.file_suffix = args.file_suffix
        self.data_part_list = args.data_part_list
        self.cutblur = args.cutblur
        super(DIV2K_MEANTIME, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )
        self.file_list = []
        self.img_iy_ix_list = []
        for i in range(len(self.data_part_list)):
            if train:
                self.file_list.append(os.path.join(self.dir_lr_bin, "{}_train.pt".format(self.data_part_list[i])))
            else:
                self.file_list.append(os.path.join(self.dir_lr_bin, "{}_val.pt".format(self.data_part_list[i])))
            with open(self.file_list[i], 'rb') as _f:
                self.img_iy_ix_list.append(pickle.load(_f))

    
    def __getitem__(self, idx):
        pairs = []
        for i in range(len(self.data_part_list)):
            lr, hr, filename, iy, ix = self._load_file(i, idx)
            pair = self.get_patch(lr, hr, iy, ix)
            pair = common.set_channel(*pair, n_channels=self.args.n_colors)
            pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
            
            lr, hr = pair_t
            pairs.append(lr)
            pairs.append(hr)

        return pairs[0], pairs[1], pairs[2], pairs[3], pairs[4], pairs[5]

    def _load_file(self, data_index, idx):
        idx = self._get_index(idx)
        img, iy, ix = self.img_iy_ix_list[data_index][idx]
        filename = str(img).rjust(4,'0')
        f_hr = os.path.join(self.dir_hr_bin, '{}.pt'.format(filename))
        f_lr = os.path.join(self.dir_lr_bin, '{}x{}.pt'.format(filename, self.scale[0]))

        if self.args.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr)
            lr = imageio.imread(f_lr)
        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)

        return lr, hr, filename, int(iy), int(ix)

    def get_patch(self, lr, hr, iy, ix):
        ########################
        def _get_patch(*args, iy, ix, data_partion=0.7, patch_size=96, scale=2, multi=False, input_large=False):
            # ih, iw = args[0].shape[:2]

            if not input_large:
                p = scale if multi else 1
                tp = p * patch_size
                ip = tp // scale
            else:
                tp = patch_size
                ip = patch_size

            if not input_large:
                tx, ty = scale * ix, scale * iy
            else:
                tx, ty = ix, iy

            ret = [
                args[0][iy:iy + ip, ix:ix + ip, :],
                *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
            ]

            return ret
        ###########################
        scale = self.scale[self.idx_scale]
        lr, hr = _get_patch(
            lr, hr, 
            iy=iy, ix=ix,
            data_partion=self.data_partion,
            patch_size=self.args.patch_size,
            scale=scale,
            multi=(len(self.scale) > 1),
            input_large=self.input_large
        )
        if not self.args.no_augment: lr, hr = common.augment(lr, hr)

        return lr, hr

    def _scan(self):
        names_hr, names_lr = super(DIV2K_MEANTIME, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        self.dir_hr_bin = os.path.join(self.apath, 'bin', 'DIV2K_train_HR')
        self.dir_lr_bin = os.path.join(self.apath, 'bin', 'DIV2K_train_LR_bicubic', 'X{}'.format(self.scale[0]))
        self.ext = ('.png', '.png')

    def set_data_partion(self, value):
        assert(0 <= value <= 1)
        self.data_partion = value

    def __len__(self):
        if self.train:
            return len(self.images_hr)
        else:
            return len(self.img_iy_ix_list) // 1000

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx * 1000