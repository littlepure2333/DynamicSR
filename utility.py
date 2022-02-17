import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio
import pickle
import cv2

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

class timer():
    def __init__(self):
        self.acc = 0
        self.times = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        ''' accumulate (toc-tic) and hold times'''
        self.acc += self.toc()
        self.times += 1

    def release(self, avg=False, reset=True):
        ''' return all accumulated (toc-tic) in sum/avg mode, then reset'''
        ret = self.acc / self.count() if avg else self.acc
        if reset: self.reset()

        return ret

    def count(self):
        return self.times

    def reset(self):
        self.acc = 0
        self.times = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def save_exit_list(self, exit_list):
        with open(self.get_path('exit_list.pt'), 'wb') as _f:
            pickle.dump(exit_list, _f)

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=True, print_time=True):
        if print_time:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
            log = '[' + current_time + '] ' + log
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())
        
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))

    def save_results_dynamic(self, dataset, filename, save_dict, scale):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(self.args.data_test[0]),
                '{}_x{}_'.format(filename, scale)
            )

            # postfix = ('SR', 'LR', 'HR')
            # for v, p in zip(save_list, postfix):
            for key, value in save_dict.items():
                normalized = value[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, key), tensor_cpu))

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()
    # mse = diff.pow(2).mean()
    if mse <= 0:
        print(mse)
        print(sr)
        print(hr)
        raise ValueError
    
    # psnr = -10 * math.log10(mse)
    psnr = -10 * torch.log10(mse)

    return psnr

def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    if len(milestones) == 1:
        kwargs_scheduler = {'step_size': milestones[0], 'gamma': args.gamma}
        scheduler_class = lrs.StepLR
    else:
        kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
        scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

def print_params(model, checkpoint, args):
    if args.model == "SRVC":
        head_params = sum([l.nelement() for l in model.model.head.parameters()])
        kernel_params = sum([l.nelement() for l in model.model.kernel.parameters()])
        bias_params = sum([l.nelement() for l in model.model.bias.parameters()])
        
        checkpoint.write_log("head parameters: {}".format(head_params+kernel_params+bias_params))
        
        body_params = sum([l.nelement() for l in model.model.body.parameters()])
        
        checkpoint.write_log("body parameters: {}".format(body_params))

        model_params = sum([l.nelement() for l in model.model.parameters()])
        
        checkpoint.write_log("total parameters: {}".format(model_params))


def crop(img, crop_sz, step):
    b, c, h, w = img.shape
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    index = 0
    num_h = 0
    lr_list=[]
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            crop_img = img[:, :, x:x + crop_sz, y:y + crop_sz]
            lr_list.append(crop_img)
    new_h=x + crop_sz # new height after crop
    new_w=y + crop_sz # new width  after crop
    return lr_list, num_h, num_w, new_h, new_w

def combine(sr_list, num_h, num_w, h, w, patch_size, step):
    index=0
    sr_img = torch.zeros((1, 3, h, w)).to(sr_list[0].device)
    for i in range(num_h):
        for j in range(num_w):
            sr_img[:, :, i*step:i*step+patch_size, j*step:j*step+patch_size] += sr_list[index]
            index+=1

    # mean the overlap region
    for j in range(1,num_w):
        sr_img[:, :, :, j*step:j*step+(patch_size-step)]/=2
    for i in range(1,num_h):
        sr_img[:, :, i*step:i*step+(patch_size-step), :]/=2

    return sr_img

def crop_parallel(img, crop_sz, step):
    b, c, h, w = img.shape
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    index = 0
    num_h = 0
    lr_list=torch.Tensor().to(img.device)
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            crop_img = img[:, :, x:x + crop_sz, y:y + crop_sz]
            lr_list = torch.cat([lr_list, crop_img])
    new_h=x + crop_sz # new height after crop
    new_w=y + crop_sz # new width  after crop
    return lr_list, num_h, num_w, new_h, new_w

def combine_parallel(sr_list, num_h, num_w, h, w, patch_size, step):
    index=0
    sr_img = torch.zeros((1, 3, h, w)).to(sr_list.device)
    for i in range(num_h):
        for j in range(num_w):
            sr_img[:, :, i*step:i*step+patch_size, j*step:j*step+patch_size] += sr_list[index]
            index+=1

    # mean the overlap region
    for j in range(1,num_w):
        sr_img[:, :, :, j*step:j*step+(patch_size-step)]/=2
    for i in range(1,num_h):
        sr_img[:, :, i*step:i*step+(patch_size-step), :]/=2

    return sr_img


def add_mask(sr_img, scale, num_h, num_w, h, w, patch_size, step, exit_index, show_number=True):
    # white and 7-rainbow
    # color_list = [(255,255,255),(255,0,0),(255,165,0),(255,255,0),(0,255,0),(0,127,255),(0,0,255),(139,0,255)]
    color_list = [(255,255,255),(255,225,0),(255,165,0),(240,0,0),(0,255,0),(0,127,255),(0,0,255),(139,0,255)]

    idx = 0
    sr_img = sr_img.squeeze().permute(1,2,0).numpy() # (H,W,C)
    mask = np.zeros((sr_img.shape), 'float32')
    for i in range(num_h):
        for j in range(num_w):
            bbox = [j * step + 2*scale, 
                     i * step + 2*scale,
                     j * step + patch_size - (2*scale+1),
                     i * step + patch_size - (2*scale+1)]  # xl,yl,xr,yr

            color = color_list[int(exit_index[idx])]
            cv2.rectangle(mask, (bbox[0]+1, bbox[1]+1), (bbox[2]-1, bbox[3]-1), color=color, thickness=-1)
            cv2.putText(mask, '{}'.format(int(exit_index[idx]+1)), 
                        (bbox[0]+4*scale, bbox[3]-4*scale), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
            idx += 1

    # add_mask
    alpha = 0.7
    beta = 1 - alpha
    gamma = 0
    sr_mask = cv2.addWeighted(sr_img, alpha, mask, beta, gamma)
    sr_mask = torch.from_numpy(sr_mask).permute(2,0,1).unsqueeze(0)

    return sr_mask



def calc_avg_exit(exit_list):
    if exit_list.ndim == 2:
        exit_list = exit_list.sum(0)
    num = exit_list.sum()
    index = torch.arange(0,len(exit_list),1).float()
    avg = (index*exit_list).sum() / num

    return avg

def calc_flops(exit_list, model_name, scale, exit_interval):
    
    if exit_list.ndim == 2:
        exit_list = exit_list.sum(0)
    
    if model_name.find("EDSR") >= 0:
        if scale == 2:
            flops_list = torch.Tensor([4.27,5.47,6.68,7.89,9.10,10.31,11.52,12.72,13.93,15.14,16.35,17.56,18.77,19.98,21.18,22.39,23.60,24.81,26.02,27.23,28.44,29.64,30.85,32.06,33.27,34.48,35.69,36.89,38.10,39.31,40.52,41.73])
            flops_list = flops_list[exit_interval-1::exit_interval]
        elif scale == 3:
            flops_list = torch.Tensor([7.32,8.53,9.74,10.95,12.16,13.36,14.57,15.78,16.99,18.20,19.41,20.62,21.82,23.03,24.24,25.45,26.66,27.87,29.07,30.28,31.49,32.70,33.91,35.12,36.33,37.53,38.74,39.95,41.16,42.37,43.58,44.79])
            flops_list = flops_list[exit_interval-1::exit_interval]
        elif scale == 4:
            flops_list = torch.Tensor([14.02,15.23,16.44,17.64,18.85,20.06,21.27,22.48,23.69,24.89,26.10,27.31,28.52,29.73,30.94,32.15,33.35,34.56,35.77,36.98,38.19,39.40,40.61,41.81,43.02,44.23,45.44,46.65,47.86,49.06,50.27,51.48])
            flops_list = flops_list[exit_interval-1::exit_interval]
    elif model_name.find("RCAN") >= 0:
        if scale == 2:
            flops_list = torch.Tensor([1.75,3.30,4.85,6.41,7.96,9.51,11.06,12.61,14.16,15.72])
            flops_list = flops_list[exit_interval-1::exit_interval]
        elif scale == 3:
            flops_list = torch.Tensor([1.95,3.50,5.05,6.60,8.16,9.71,11.26,12.81,14.36,15.91])
            flops_list = flops_list[exit_interval-1::exit_interval]
        elif scale == 4:
            flops_list = torch.Tensor([2.38,3.93,5.48,7.03,8.58,10.13,11.69,13.24,14.79,16.34])
            flops_list = flops_list[exit_interval-1::exit_interval]

    num = exit_list.sum()
    flops = (flops_list*exit_list).sum() / num
    percent = flops / flops_list[-1] * 100.0

    return flops, percent


if __name__ == "__main__":
    import time
    tic = time.time()
    
    img1 = torch.ones(32,3,96,96)*101
    img2 = torch.ones(32,3,96,96)*100
    # print(img1)
    # print(img2)

    psnr = calc_psnr(img1, img2, 2, 255)
    print(psnr)

    toc = time.time()
    print(toc-tic)