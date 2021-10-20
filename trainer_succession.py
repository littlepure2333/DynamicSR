import os
# import math
from decimal import Decimal
import numpy as np

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm
# from model.patchnet import PatchNet

from data.utils_image import calculate_ssim
# import time

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ssim = args.ssim
        self.lpips_alex = args.lpips_alex
        self.lpips_vgg = args.lpips_vgg

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.prev_epoch = 0
        self.exit_list = self.model.model.exit_list
        self.exit_index = 0
        self.best = None
        # start from the first exit
        self.model.apply(lambda m: setattr(m, 'exit_index', self.exit_index))

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1 + self.prev_epoch
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\t[Exit {}/{}]\tLearning rate: {:.2e}'.format(epoch, self.exit_index, len(self.exit_list)-1, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
            
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

        epoch = self.optimizer.get_last_epoch() + self.prev_epoch
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
                psnr_list = []

                for lr, hr, filename in tqdm(d, ncols=80):
                    # t0 = time.time()
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    # t1 = time.time()
                    # print("t1-t0:{:.3f}".format(t1-t0))
                    #print('sr',sr.size())
                    #print('hr',hr.size())
                    b,c,h,w = sr.size()

                    save_list = [sr]
                    # self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                    #     sr, hr, scale, self.args.rgb_range, dataset=d
                    # )
                    item_psnr = utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=d)
                    self.ckp.log[-1, idx_data, idx_scale] += item_psnr
                    # t2 = time.time()
                    # print("t2-t1:{:.3f}".format(t2-t1))
                    if self.args.save_psnr_list:
                        psnr_list.append(item_psnr)
                    if self.ssim:
                        sr = sr.squeeze().cpu().permute(1,2,0).numpy()
                        hr = hr.squeeze().cpu().permute(1,2,0).numpy()
                        ssim = calculate_ssim(sr, hr)
                        ssim_total += ssim
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)
                    torch.cuda.empty_cache()
                    # t3 = time.time()
                    # print("t3-t2:{:.3f}".format(t3-t2))

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)

                self.best = self.ckp.log.max(0) # return max value and index
                self.current_exit_best = list(self.ckp.log[self.prev_epoch:].max(0))
                self.current_exit_best[1] += self.prev_epoch
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {}) (Exit{} Best: {:.3f} @ epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        self.best[0][idx_data, idx_scale],
                        self.best[1][idx_data, idx_scale] + 1,
                        self.exit_index,
                        self.current_exit_best[0][idx_data, idx_scale],
                        self.current_exit_best[1][idx_data, idx_scale] + 1
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
                if self.args.save_psnr_list:
                    psnr_list_np = np.array(psnr_list)
                    np.save(os.path.join(self.ckp.dir, "psnr_list.pt"), psnr_list_np)

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(self.best[1][0, 0] + 1 == epoch))

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
            epoch = self.optimizer.get_last_epoch() + self.prev_epoch
            if self.best and ((epoch - int(self.current_exit_best[1])) > self.args.conv_thre): # This exit has converged, then optimize next exit
                # if all exit has converged, then finish the training
                if self.exit_index >= len(self.exit_list) - 1: return True 

                self.ckp.write_log(
                    'Exit {} has converged, now optimize at exit {}'.format(self.exit_index, self.exit_index+1), refresh=True
                )

                # if freeze the converged exit's parameters
                if self.args.freeze: 
                    for m in self.model.model.head.parameters():
                        m.requires_grad = False
                    for m in self.model.model.body[:self.exit_list[self.exit_index]+1].parameters():
                        m.requires_grad = False
                    if not self.args.shared_tail:
                        for m in self.model.model.tail[:].parameters():
                            m.requires_grad = False
                        for m in self.model.model.tail[self.exit_index+1].parameters():
                            m.requires_grad = True
                self.optimizer = self.optimizer = utility.make_optimizer(self.args, self.model)
                # print parameters frozen status
                for name, param in self.model.named_parameters(): 
                    param_info = str(name) + ' ' + str(param.size()) + ' ' + str(param.requires_grad)
                    self.ckp.write_log(param_info)
                
                # update status
                self.prev_epoch = epoch
                self.exit_index += 1

                # else optimize next exit
                self.model.apply(lambda m: setattr(m, 'exit_index', self.exit_index))
            
            return False
            # return epoch > self.args.epochs

