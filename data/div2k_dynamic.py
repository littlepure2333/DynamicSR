import os
from data import srdata
from augments import cutblur
import pickle
import numpy as np
import imageio
from data import common

class DIV2K_DYNAMIC(srdata.SRData):
    def __init__(self, args, name='DIV2K_DYNAMIC', train=True, benchmark=False):
        # self.assistant = args.assistant
        data_range = [r.split('-') for r in args.data_range.split('/')]
        self.trian = train
        self.bins = args.bins
        with open(args.statistics_file, 'rb') as _f:
            self.statistics = pickle.load(_f) # all_id_iy_ix_metric
        self.hist, self.bin_edges = np.histogram(self.statistics[:,-1], bins=self.bins)
        self.bins_index = 0
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(DIV2K_DYNAMIC, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )
    
    def __getitem__(self, idx):
        idx = np.sum(self.hist[:self.bins_index])+idx
        image_idx, iy, ix, metric = self.statistics[idx]
        lr, hr, filename = self._load_file(image_idx)
        pair = self.get_patch(lr, hr, iy, ix)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        lr, hr = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
        filename = "{}_{}_{}".format(filename, iy, ix)

        return lr, hr, filename

    def _load_file(self, image_idx):
        filename = str(image_idx).rjust(4,'0')
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

        return lr, hr, filename

    def get_patch(self, lr, hr, iy, ix):
        ########################
        def _get_patch(*args, iy, ix, patch_size=96, scale=2, multi=False, input_large=False):
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
            patch_size=self.args.patch_size,
            scale=scale,
            multi=(len(self.scale) > 1),
            input_large=self.input_large
        )
        if not self.args.no_augment: lr, hr = common.augment(lr, hr)

        return lr, hr

    
    def _scan(self):
        names_hr, names_lr = super(DIV2K_DYNAMIC, self)._scan()
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

    def __len__(self):
        return int(self.hist[self.bins_index])

