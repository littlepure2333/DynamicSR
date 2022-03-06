import os
import math
from decimal import Decimal
import numpy as np

import utility
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn.utils as utils
from tqdm import tqdm
from model.patchnet import PatchNet
# import lpips

from data.utils_image import calculate_ssim

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ssim = args.ssim

        self.ckp = ckp
        # self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.device = torch.device('cpu' if args.cpu else 'cuda')

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

        if self.args.patchnet:
            self.patchnet = PatchNet(args).to(self.device)

    def train(self):
        raise NotImplementedError("train is not implemented yet.")

    def test(self):
        torch.set_grad_enabled(False)

        self.ckp.write_log('\nEvaluation:')
        exit_len = int(self.args.n_resblocks/self.args.exit_interval)
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        # for bins_index in range(self.args.bins):
        for bins_index in [self.args.bin_index]:
            self.loader_test[0].dataset.bins_index = bins_index
            runtime_len = self.args.n_test_samples if len(self.loader_test[0].dataset) > self.args.n_test_samples else len(self.loader_test[0].dataset)
            self.ckp.add_log(
                torch.zeros(1, len(self.loader_test), exit_len)
            )
            for idx_data, d in enumerate(self.loader_test):
                for idx_scale, scale in enumerate(self.scale):
                    d.dataset.set_scale(idx_scale)
                    dl = iter(d)
                    ssim_total = 0
                    save_dict = {}

                    for idx_item in tqdm(range(runtime_len), ncols=80):
                        lr, hr, filename = dl.next()
                        lr, hr = self.prepare(lr, hr)

                        sr = self.model(lr, idx_scale)
                        if self.args.model == "EDSR_dynamic":
                            for i, sr_i in enumerate(sr):
                                sr_i = utility.quantize(sr_i, self.args.rgb_range)
                                save_dict['SR-{}'.format(i)] = sr_i
                                item_psnr = utility.calc_psnr(sr_i, hr, scale, self.args.rgb_range, dataset=d)
                                self.ckp.log[-1, idx_data, i] += item_psnr.cpu()
                        elif self.args.model == "EDSR":
                            save_dict['SR'] = sr
                            item_psnr = utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=d)
                            self.ckp.log[-1, idx_data, 0] += item_psnr.cpu()
                        else:
                            raise NotImplementedError

                        if self.ssim:
                            sr_np = sr.squeeze().cpu().permute(1,2,0).numpy()
                            hr_np = hr.squeeze().cpu().permute(1,2,0).numpy()
                            ssim = calculate_ssim(sr_np, hr_np)
                            ssim_total += ssim
                        if self.args.save_gt:
                            save_dict['LR'] = lr
                            save_dict['HR'] = hr

                        if self.args.save_results:
                            self.ckp.save_results_dynamic(d, filename[0], save_dict, scale)
                        torch.cuda.empty_cache()

                    self.ckp.log[-1, idx_data, :] /= runtime_len

                    psnr_list = ["{}:{:.3f}".format(i, self.ckp.log[-1, idx_data, i]) for i in range(exit_len)]
                    psnr_list = ','.join(psnr_list)

                    self.ckp.write_log(
                        '[{} x{}]\t[{}/{}bins]\tPSNR: {})'.format(
                            d.dataset.name,
                            scale,
                            bins_index,
                            self.args.bins - 1,
                            psnr_list
                        )
                    )

                    if self.ssim:
                        self.ckp.write_log(
                            '[{} x{}]\tSSIM: {:.4f}'.format(
                                d.dataset.name,
                                scale,
                                ssim_total/len(d)
                            )
                        )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()
        
        # save psnr log
        torch.save(self.ckp.log, self.ckp.get_path('psnr_log.pt'))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

