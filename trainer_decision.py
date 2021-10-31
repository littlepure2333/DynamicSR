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
        self.loader_train = loader.loader_train
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
            sr, decisions = self.model(lr, 0)
            loss = 0

            # sum decision mode
            for sr_i, de_i in zip(sr, decisions):
                loss += self.loss(sr_i, de_i, hr)
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
        torch.cuda.empty_cache()

    def test(self):
        torch.set_grad_enabled(False)

        self.ckp.write_log('\nEvaluation:')
        exit_len = int(self.args.n_resblocks/self.args.exit_interval)
        self.model.eval()

        if self.args.save_results: self.ckp.begin_background()
        timer_test = utility.timer()

        d = self.loader_test[0]
        d.dataset.set_scale(0)
        ssim_total = 0
        save_dict = {}

        for bins_index in range(self.args.bins):
            d.dataset.bins_index = bins_index
            runtime_len = self.args.n_test_samples if len(self.loader_test[0].dataset) > self.args.n_test_samples else len(self.loader_test[0].dataset)
            dl = iter(d)
            for idx_item in tqdm(range(runtime_len), ncols=80):
                lr, hr, filename = dl.next()
                lr, hr = self.prepare(lr, hr)

                sr, exit_index, de = self.model(lr, 0)
                save_dict['SR'] = sr
                item_psnr = utility.calc_psnr(sr, hr, self.scale[0], self.args.rgb_range, dataset=d)
                self.ckp.add_log(
                    torch.zeros(2)
                )
                self.ckp.log[0] = item_psnr
                self.ckp.log[1] = exit_index

                if self.ssim:
                    sr_np = sr.squeeze().cpu().permute(1,2,0).numpy()
                    hr_np = hr.squeeze().cpu().permute(1,2,0).numpy()
                    ssim = calculate_ssim(sr_np, hr_np)
                    ssim_total += ssim
                if self.args.save_gt:
                    save_dict['LR'] = lr
                    save_dict['HR'] = hr

                if self.args.save_results:
                    self.ckp.save_results_dynamic(d, filename[0], save_dict, self.scale[0])
                torch.cuda.empty_cache()

                self.ckp.write_log(
                    '[{} x{}]\t[{}]\tPSNR: {:.3f}\t exit at {}/{} de:{:6f})'.format(
                        d.dataset.name,
                        self.scale[0],
                        filename,
                        self.ckp.log[0],
                        int(self.ckp.log[1]),
                        exit_len-1,
                        float(de)
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

