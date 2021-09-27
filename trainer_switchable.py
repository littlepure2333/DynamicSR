import data
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
        # self.lpips_alex = args.lpips_alex
        # self.lpips_vgg = args.lpips_vgg
        # self.loss_fn_alex = lpips.LPIPS(net='alex')
        # self.loss_fn_vgg = lpips.LPIPS(net='vgg')

        self.ckp = ckp
        self.loader = loader
        self.data_part, self.width_mult = None, None
        # self.loader_train = loader[0].loader_train
        # self.loader_test = loader[0].loader_test
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
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        if self.args.final_data_partion != None and self.args.final_data_partion <= 1:
            total_epochs = self.args.epochs
            increment = (self.args.final_data_partion - self.args.data_partion) * (epoch - 1) / (total_epochs - 1)
            new_data_partion = self.args.data_partion + increment
            self.loader_train.dataset.datasets[0].set_data_partion(new_data_partion)
            self.ckp.write_log("current data_partion is: {}".format(new_data_partion))
            
        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            loss.backward()

            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                ssim_total = 0

                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    b,c,h,w = sr.size()

                    save_list = [sr]
                    item_psnr = utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=d)
                    self.ckp.log[-1, idx_data, idx_scale] += item_psnr
                    if self.args.save_gt:
                        save_list.extend([lr, hr])
                    if self.ssim:
                        sr = sr.squeeze().cpu().permute(1,2,0).numpy()
                        hr = hr.squeeze().cpu().permute(1,2,0).numpy()
                        ssim = calculate_ssim(sr, hr)
                        ssim_total += ssim

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)
                    torch.cuda.empty_cache()

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)

                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
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

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

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
            epoch = self.optimizer.get_last_epoch()
            if epoch >= self.args.epochs: return True

            '''
            # discrete mode --------------👇
            part_epochs = self.args.epochs // len(self.args.data_part_list)
            if epoch % (part_epochs) == 0: # should change data

                # harder sub-mode
                # part_index = epoch // (part_epochs)
                # easier sub-mode
                part_index = len(self.args.data_part_list)-1 - epoch // (part_epochs)

                self.data_part = self.args.data_part_list[part_index]
                self.loader_train = self.loader[part_index].loader_train
                self.loader_test = self.loader[part_index].loader_test

                # should change model
                self.width_mult = self.args.width_mult_list[part_index]
                self.model.apply(lambda m: setattr(m, 'width_mult', self.width_mult))
            # discrete mode --------------👆
            '''
            # recurrent mode --------------👇
            # harder sub-mode
            # part_index = epoch % len(self.args.data_part_list)
            # easier sub-mode
            part_index = len(self.args.data_part_list)-1 - epoch % len(self.args.data_part_list)

            self.data_part = self.args.data_part_list[part_index]
            self.loader_train = self.loader[part_index].loader_train
            self.loader_test = self.loader[part_index].loader_test

            # should change model
            self.width_mult = self.args.width_mult_list[part_index]
            self.model.apply(lambda m: setattr(m, 'width_mult', self.width_mult))
            # recurrent mode --------------👆
            # '''

            # log
            self.ckp.write_log(
                '[Epoch {}]\tdata part: {}\t model channel: {}x{}'.format(epoch+1, self.data_part, self.args.n_feats, self.width_mult)
            )
            return epoch >= self.args.epochs

