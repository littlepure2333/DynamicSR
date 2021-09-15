import os
import math
import random
import time
import utility
import torch
import torch.nn.functional as F
import torch.nn.utils as utils
import augments
from tqdm import tqdm
from decimal import Decimal


class Trainer:
    def __init__(self, args, my_loader, my_model, my_loss, ckp):
        """init function"""
        self.args = args
        self.scale = args.scale  # [x2, x3, x4, x5, x6 ...]
        self.idx_scale = 0
        self.ckp = ckp
        self.loader_train = my_loader.loader_train
        self.loader_test = my_loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        """training function"""
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()
        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()

        for batch, (lr, hr, krl, _) in enumerate(self.loader_train):
            lr, hr, krl = self.prepare_train(lr, hr, krl)
            timer_data.hold()
            timer_model.tic()
            self.optimizer.zero_grad()
            loss = 0

            # iterate all scales
            # for self.idx_scale in range(len(self.scale)):
            #     key = str(self.scale[self.idx_scale])
            #     lr_, hr_, krl_ = lr[key], hr[key], krl[key]  # lr, hr, krl is dict
            #     sr_ = self.model(lr_, self.idx_scale, krl=krl_)  # (lr, idx_scale, krl)
            #     loss += self.loss(sr_, hr_)
            # loss /= len(self.scale)

            # choose a random scale
            self.idx_scale = random.randint(0, len(self.scale) - 1)
            key = str(self.idx_scale)
            lr_, hr_, krl_ = lr[key], hr[key], krl[key]
            # lr_, hr_, krl_ = lr, hr, krl

            # adaptive
            meta_ = {'masks': []}
            sr_, pred_, depth_ = self.model(lr_, self.idx_scale, krl=krl_, meta=meta_)
            loss_l1, _ = self.loss(sr_, hr_, meta_)
            loss_pred = 0.01 * (pred_.mean((2, 3)) - depth_).clamp_min_(0).mean()
            loss = loss_l1 + loss_pred
            cost_perc = pred_.sum() / (self.args.n_resblocks * pred_.size(0) * pred_.size(2) * pred_.size(3))

            # backward
            loss.backward()

            # clip gradient
            if self.args.gclip > 0:
                utils.clip_grad_value_(self.model.parameters(), self.args.gclip)

            self.optimizer.step()
            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s\t{}\t{}\t{}'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release(),
                    self.idx_scale,
                    cost_perc,
                    loss_pred.item()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        """testing function"""
        torch.set_grad_enabled(False)
        epoch = self.optimizer.get_last_epoch()  # epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))  # (1, #num, #scale)
        log_ssim = torch.zeros(1, len(self.loader_test), len(self.scale))  # (1,#num, #scale)
        log_perc = torch.zeros(1, len(self.loader_test), len(self.scale))  # (1,#num, #scale)

        self.model.eval()
        timer_test = utility.timer()

        if self.args.save_results:
            self.ckp.begin_background()

        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):

                d.dataset.set_scale(idx_scale)

                for lr, hr, krl, filename in tqdm(d, ncols=80):
                    lr, hr, krl = self.prepare_test(lr, hr, krl)

                    # cutblur
                    # meta = {'masks': [], 'gumbel_temp': 1.0, 'gumbel_noise': False, 'epoch': epoch}
                    # lr = F.interpolate(lr, scale_factor=self.scale[self.idx_scale], mode='nearest')
                    # sr, meta = self.model(lr, idx_scale, krl=krl, meta=meta)
                    # _, meta = self.loss(sr, hr, meta)

                    # adaptive
                    meta = {'masks': []}
                    sr, pred, depth = self.model(lr, self.idx_scale, krl=krl, meta=meta)
                    print(depth)
                    # loss_l1, _ = self.loss(sr, hr, meta)
                    cost_perc = pred.sum() / (self.args.n_resblocks * pred.size(0) * pred.size(2) * pred.size(3))
                    print(pred)
                    print(cost_perc)
                    exit(0)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range,
                        # dataset=d
                    )

                    log_ssim[-1, idx_data, idx_scale] += utility.calc_ssim(
                        sr, hr, scale, benchmark=d.dataset.benchmark
                    )

                    log_perc[-1, idx_data, idx_scale] += cost_perc

                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)  # /num_sample
                log_ssim[-1, idx_data, idx_scale] /= len(d)  # /num_sample
                log_perc[-1, idx_data, idx_scale] /= len(d)  # /num_sample

                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} SSIM: {:.3f}, Perc: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        log_ssim[-1, idx_data, idx_scale],
                        log_perc[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
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

    def prepare_train(self, *args):
        """prepare data for training"""
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(device)

        def _prepare_dict(tensor_dict):
            for k, v in tensor_dict.items():
                tensor_dict[k] = _prepare(v)
            return tensor_dict

        return [_prepare_dict(a) for a in args]

    def prepare_test(self, *args):
        """prepare data for testing"""
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        """terminate"""
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs


if __name__ == '__main__':
    print(__file__)
