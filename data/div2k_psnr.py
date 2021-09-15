import os
import random
import pickle
from data import srdata
from data import common
from augments import cutblur

class DIV2K_PSNR(srdata.SRData):
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
        self.cutblur = args.cutblur
        super(DIV2K_PSNR, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )
    
    def __getitem__(self, idx):
        lr, hr, filename, psnr_index = self._load_file(idx)
        pair = self.get_patch(lr, hr, psnr_index)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
        
        lr, hr = pair_t

        if self.cutblur is None:
            return lr, hr, filename
        elif self.cutblur > 0:
            hr, lr = cutblur(hr, lr, alpha = self.cutblur, train=self.train)
        elif self.cutblur == 0:
            hr, lr = cutblur(hr, lr, alpha = self.cutblur, train=False)

        # if self.cutblur > 0:
        #     pair_t[1], pair_t[0] = cutblur(pair_t[1], pair_t[0], alpha = self.cutblur)

        return lr, hr, filename

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]
        f_psnr = f_lr.replace('.pt', self.file_suffix)

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr)
            lr = imageio.imread(f_lr)
        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)
            with open(f_psnr, 'rb') as _f:
                psnr_index = pickle.load(_f)

        return lr, hr, filename, psnr_index

    def get_patch(self, lr, hr, psnr_index):
        ########################
        def _get_patch(*args, psnr_index, data_partion=0.7, patch_size=96, scale=2, multi=False, input_large=False):
            # ih, iw = args[0].shape[:2]
            n_patch = int(psnr_index.shape[0] * data_partion)

            if not input_large:
                p = scale if multi else 1
                tp = p * patch_size
                ip = tp // scale
            else:
                tp = patch_size
                ip = patch_size
                
            if n_patch == 0:
                index = 0
            elif n_patch > 0:    # positive order
                index = random.randrange(0, n_patch)
            else: # n_patch < 0: # reverse order
                index = random.randrange(n_patch, 0)    
            iy = int(psnr_index[index][0])
            ix = int(psnr_index[index][1])

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
        if self.train:
            lr, hr = _get_patch(
                lr, hr, 
                psnr_index=psnr_index,
                data_partion=self.data_partion,
                patch_size=self.args.patch_size,
                scale=scale,
                multi=(len(self.scale) > 1),
                input_large=self.input_large
            )
            if not self.args.no_augment: lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]
        

        return lr, hr

    def _scan(self):
        names_hr, names_lr = super(DIV2K_PSNR, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DIV2K_PSNR, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        if self.input_large: self.dir_lr += 'L'

    def set_data_partion(self, value):
        assert(0 <= value <= 1)
        self.data_partion = value