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
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.exit_list = self.model.model.exit_list
        self.exit_index = 0
        # start from the first exit
        self.model.apply(lambda m: setattr(m, 'exit_index', self.exit_index))

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
            loss = 0

            # increase mode
            # for i in range(0, len(self.exit_list)):
            #     self.model.apply(lambda m: setattr(m, 'exit_index', i))
            #     sr = self.model(lr, 0)
            #     loss += self.loss(sr, hr) * (i+1)
            # loss.backward()

            # sum mode
            # for i in range(0, len(self.exit_list)):
            #     self.model.apply(lambda m: setattr(m, 'exit_index', i))
            #     sr = self.model(lr, 0)
            #     loss += self.loss(sr, hr)
            # loss.backward()

            # forward every mode
            # for i in range(0, len(self.exit_list)):
            #     self.model.apply(lambda m: setattr(m, 'exit_index', i))
            #     sr = self.model(lr, 0)
            #     loss = self.loss(sr, hr)
            #     loss.backward()
            #     torch.cuda.empty_cache()
            
            # only last mode
            self.model.apply(lambda m: setattr(m, 'exit_index', len(self.exit_list)-1))
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            loss.backward()
            

            torch.cuda.empty_cache()
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
            torch.zeros(1, len(self.loader_test), len(self.exit_list))
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
                    for i  in range(len(self.exit_list)):
                        self.model.apply(lambda m: setattr(m, 'exit_index', i))
                        sr_i = self.model(lr, idx_scale)
                        sr_i = utility.quantize(sr_i, self.args.rgb_range)
                        save_dict['SR-{}'.format(i)] = sr_i
                        item_psnr = utility.calc_psnr(sr_i, hr, scale, self.args.rgb_range, dataset=d)
                        self.ckp.log[-1, idx_data, i] += item_psnr
                        torch.cuda.empty_cache()
                    
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

                psnr_list = ["{}:{:.3f}".format(i, self.ckp.log[-1, idx_data, i]) for i in range(len(self.exit_list))]
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

