import os
import math
from decimal import Decimal
import numpy as np
from numpy.lib.function_base import average

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
        self.patch_size = args.patch_size
        self.step = args.step
        # self.lpips_alex = args.lpips_alex
        # self.lpips_vgg = args.lpips_vgg
        # self.loss_fn_alex = lpips.LPIPS(net='alex')
        # self.loss_fn_vgg = lpips.LPIPS(net='vgg')

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
            sr = self.model(lr, 0)
            loss = 0
            # distill mode
            # for i in range(len(sr) - 1):
            #     loss += self.loss(sr[i],sr[-1].detach())
            # loss += self.loss(sr[-1], hr)
            # loss.backward()

            # increase distill mode
            # for i in range(len(sr) - 1):
            #     loss += self.loss(sr[i],sr[-1].detach()) * (i+1)
            # loss += self.loss(sr[-1], hr) * len(sr)
            # loss.backward()

            # post distill mode
            # sr_last_detach = sr[-1].detach()
            # loss = self.loss(sr[-1], hr)
            # loss.backward()
            # for i in reversed(range(len(sr) - 1)):
            #     sr = self.model(lr, 0)
            #     loss = self.loss(sr[i],sr_last_detach)
            #     loss.backward()
            #     torch.cuda.empty_cache()

            # increase mode
            # for i, sr_i in enumerate(sr):
            #     loss += self.loss(sr_i, hr) * (i+1)
            # loss.backward()

            # sum mode
            # for i, sr_i in enumerate(sr):
            #     loss += self.loss(sr_i, hr)
            # loss.backward()

            # sum decision mode
            for sr_i in sr:
                loss += self.loss(sr_i, hr)
            loss.backward()

            # forward every mode
            # loss = self.loss(sr[0], hr)
            # loss.backward()
            # for i in range(1, len(sr)):
            #     sr = self.model(lr, 0)
            #     loss = self.loss(sr[i], hr)
            #     loss.backward()
            #     torch.cuda.empty_cache()
            
            # only last mode
            # loss = self.loss(sr[-1], hr)
            # loss.backward()
            

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

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        exit_len = int(self.args.n_resblocks/self.args.exit_interval)
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), exit_len)
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                ssim_total = 0
                # lpips_vgg_total = 0
                # lpips_alex_total = 0
                save_dict = {}

                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    for i, sr_i in enumerate(sr):
                        sr_i = utility.quantize(sr_i, self.args.rgb_range)
                        save_dict['SR-{}'.format(i)] = sr_i
                        item_psnr = utility.calc_psnr(sr_i, hr, scale, self.args.rgb_range, dataset=d).cpu()
                        self.ckp.log[-1, idx_data, i] += item_psnr
                    
                    b,c,h,w = sr_i.size()
                    if self.ssim:
                        sr_i_np = sr_i.squeeze().cpu().permute(1,2,0).numpy()
                        hr_np = hr.squeeze().cpu().permute(1,2,0).numpy()
                        ssim = calculate_ssim(sr_i_np, hr_np)
                        ssim_total += ssim
                    # if self.lpips_vgg:
                        #print('sr',torch.reshape(torch.Tensor(sr),(1,c,h,w)).size())
                        #print('hr',torch.reshape(torch.Tensor(hr),(1,c,h,w)).size())
                    #     lpips_vgg = self.loss_fn_vgg(torch.reshape(torch.Tensor(sr_i),(b,c,h,w)),torch.reshape(torch.Tensor(hr),(b,c,h,w)))
                    #     lpips_vgg_total += lpips_vgg
                    # if self.lpips_alex:
                    #     lpips_alex = self.loss_fn_alex(torch.reshape(torch.Tensor(sr_i),(b,c,h,w)),torch.reshape(torch.Tensor(hr),(b,c,h,w)))
                    #     lpips_alex_total += lpips_alex
                        #print('alex',lpips_alex_total[0][0][0][0]/len(d),type(lpips_alex_total))
                    if self.args.save_gt:
                        save_dict['LR'] = lr
                        save_dict['HR'] = hr

                    if self.args.save_results:
                        self.ckp.save_results_dynamic(d, filename[0], save_dict, scale)
                    torch.cuda.empty_cache()

                self.ckp.log[-1, idx_data, :] /= len(d)

                best = self.ckp.log.max(0)

                psnr_list = ["{}:{:.3f}".format(i, self.ckp.log[-1, idx_data, i]) for i in range(exit_len)]
                psnr_list = ','.join(psnr_list)

                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {})'.format(
                        d.dataset.name,
                        scale,
                        psnr_list
                    )
                )
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, -1],
                        best[0][idx_data, -1],
                        best[1][idx_data, -1] + 1
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
                # if self.lpips_vgg:
                #      self.ckp.write_log(
                #         '[{} x{}]\tLPIPS-vgg: {:.4f}'.format(
                #             d.dataset.name,
                #             scale,
                #             lpips_vgg_total[0][0][0][0]/len(d)
                #         )
                #     )
                # if self.lpips_alex:
                #      self.ckp.write_log(
                #         '[{} x{}]\tLPIPS-alex: {:.4f}'.format(
                #             d.dataset.name,
                #             scale,
                #             lpips_alex_total[0][0][0][0]/len(d)
                #         )
                #     )

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

    def test_only_dynamic(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        if self.args.model.find("RCAN") >= 0:
            exit_len = int(self.args.n_resgroups/self.args.exit_interval)
        elif self.args.model.find("EDSR") >= 0:
            exit_len = int(self.args.n_resblocks/self.args.exit_interval)
        elif self.args.model.find("FSRCNN") >= 0:
            exit_len = int(4/self.args.exit_interval)
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
                save_dict = {}
                psnr_list = []
                exit_list = torch.zeros((1,exit_len))
                AVG_exit = 0

                for lr, hr, filename in d:
                    lr, hr = self.prepare(lr, hr) # (B,C,H,W)
                    lr_list, num_h, num_w, new_h, new_w = utility.crop(lr, self.patch_size//scale, self.step//scale)

                    sr_list = []
                    avg_exit = 0

                    # for lr_patch, hr_patch in zip(lr_list, hr_list):
                    pbar = tqdm(lr_list, ncols=120)
                    for lr_patch in pbar:
                        sr_patch, exit_index, decision = self.model(lr_patch, idx_scale)
                        pbar.set_description("[{}/{}exit]: {}".format(exit_index, exit_len-1, decision))
                        sr_list.append(sr_patch)
                        exit_list[-1,exit_index] += 1
                        avg_exit += exit_index

                    sr = utility.combine(sr_list, num_h, num_w, new_h*scale, new_w*scale, self.patch_size, self.step)
                    save_dict['SR'] = sr
                    hr = hr[:, :, 0:new_h*scale, 0:new_w*scale]

                    item_psnr = utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=d)
                    self.ckp.log[-1, idx_data, idx_scale] += item_psnr.cpu()
                    avg_exit = utility.calc_avg_exit(exit_list[-1])
                    avg_flops, avg_flops_percent = utility.calc_flops(exit_list[-1], self.args.model, scale, self.args.exit_interval)
                    self.ckp.write_log("{}\tPSNR:{:.3f}\taverage exit:[{:.2f}/{}]\tFlops:{:.2f}GFlops ({:.2f}%)".format(filename, item_psnr, avg_exit, exit_len-1, avg_flops, avg_flops_percent))
                    psnr_list.append(item_psnr)

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
                    exit_list = torch.cat([exit_list,torch.zeros((1,exit_len))])

                self.ckp.log[-1, idx_data, :] /= len(d)

                best = self.ckp.log.max(0)
                AVG_exit = utility.calc_avg_exit(exit_list)
                AVG_flops, AVG_flops_percent = utility.calc_flops(exit_list, self.args.model, scale, self.args.exit_interval)
                    
                psnr_list = ["{}:{:.3f}".format(i, psnr) for i, psnr in enumerate(psnr_list)]
                psnr_list = ','.join(psnr_list)

                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {})'.format(
                        d.dataset.name,
                        scale,
                        psnr_list
                    )
                )
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})\tThreshold: {}\tAverage exits:[{:.2f}/{}]\tFlops:{:.2f}GFlops ({:.2f}%)'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, -1],
                        best[0][idx_data, -1],
                        best[1][idx_data, -1] + 1,
                        self.args.exit_threshold,
                        AVG_exit,
                        exit_len-1,
                        AVG_flops,
                        AVG_flops_percent
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

        self.ckp.save_exit_list(exit_list)

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def test_only_static(self):
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
                save_dict = {}
                psnr_list = []

                for lr, hr, filename in d:
                    lr, hr = self.prepare(lr, hr) # (B,C,H,W)
                    lr_list, num_h, num_w, new_h, new_w = utility.crop_parallel(lr, self.patch_size//scale, self.step//scale)
                    sr_list = torch.Tensor()

                    pbar = tqdm(range(len(lr_list)//self.args.n_parallel + 1), ncols=120)
                    for lr_patch_index in pbar:
                        sr_patches = self.model(lr_list[lr_patch_index*self.args.n_parallel:(lr_patch_index+1)*self.args.n_parallel], idx_scale)
                        sr_list = torch.cat([sr_list, sr_patches.cpu()])

                    sr = utility.combine(sr_list, num_h, num_w, new_h*scale, new_w*scale, self.patch_size, self.step)
                    save_dict['SR'] = sr
                    hr = hr[:, :, 0:new_h*scale, 0:new_w*scale].cpu()

                    item_psnr = utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=d)
                    self.ckp.log[-1, idx_data, idx_scale] += item_psnr.cpu()
                    self.ckp.write_log("{}\tPSNR:{:3f}".format(filename, item_psnr))
                    psnr_list.append(item_psnr)

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

                self.ckp.log[-1, idx_data, :] /= len(d)
                best = self.ckp.log.max(0)

                psnr_list = ["{}:{:.3f}".format(i, psnr) for i, psnr in enumerate(psnr_list)]
                psnr_list = ','.join(psnr_list)

                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {})'.format(
                        d.dataset.name,
                        scale,
                        psnr_list
                    )
                )
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, -1],
                        best[0][idx_data, -1],
                        best[1][idx_data, -1] + 1
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
            if self.args.model in ['EDSR', 'RCAN', 'FSRCNN']:
                self.test_only_static()
            elif self.args.model in ['EDSR_decision', 'RCAN_decision', 'FSRCNN_decision']:
                self.test_only_dynamic()
            return True

        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

