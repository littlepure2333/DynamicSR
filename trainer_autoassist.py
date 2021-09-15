import time
import multiprocessing
from decimal import Decimal

import torch
import torch.nn.utils as utils
from tqdm import tqdm

import utility
from assistant import AssistantSampler, AssistantModelBinary


class thread_killer(object):
    """Boolean object for signaling a worker thread to terminate
    """

    def __init__(self):
        self.to_kill = False

    def __call__(self):
        return self.to_kill

    def set_tokill(self, tokill):
        self.to_kill = tokill


mp = torch.multiprocessing.get_context("spawn")

BQ_thread_killer = thread_killer()
BQ_thread_killer.set_tokill(False)
boss_queue = mp.Queue(maxsize=20)

AQ_thread_killer = thread_killer()
AQ_thread_killer.set_tokill(False)
assistant_queue = mp.Queue(maxsize=20)

global total_samples, total_success
total_samples = 1
total_success = 1

cuda_transfers_thread_killer = thread_killer()
cuda_transfers_thread_killer.set_tokill(False)
cuda_batches_queue = mp.Queue(maxsize=3)

device = torch.device("cuda")


def BQ_feeder(rank, args, train_data, sampler, total_samples, total_success):
    """Threaded worker for pre-processing input data.
    tokill is a thread_killer object that indicates whether a thread should be terminated
    dataset_generator is the training/validation dataset generator
    batches_queue is a limited size thread-safe Queue instance.
    """
    # initialize assistant
    lr, hr, _, _ = train_data[0]
    lr_dim = torch.numel(lr)
    hr_dim = torch.numel(hr)
    assistant = AssistantModelBinary(2074680, 8298720) # TODO fix
    
    while BQ_thread_killer() == False:
        # for (batch, (lr, hr, name, batch_indices)) in enumerate(train_loader):
        idx = -1
        lr_list = []
        hr_list = []
        while True:
            # generate batch
            idx = (idx + 1) % len(train_data)
            lr, hr, filename, index = train_data[idx]
            if assistant.evaluate(lr, hr):
                lr_list.append(lr)
                hr_list.append(hr)
            else:
                continue
            if len(lr_list) == args.batch_size:
                lr_batch = torch.stack(lr_list)
                hr_batch = torch.stack(hr_list)
                lr_list.clear()
                hr_list.clear()
            else:
                continue
            boss_queue.put((lr_batch, hr_batch))
            if boss_queue.qsize() >= 5 and not assistant_queue.empty():
                # print("Try to train Assistant with BQ.size =  %d"%( boss_queue.qsize()))
                (lr_cpu, hr_cpu), feed = assistant_queue.get(block=False)
                acc, prob = assistant.train_step(lr_cpu, hr_cpu, feed, args.sec_method)
                print("assist accuracy: {}".format(acc))
                # print("Succeeded")
            # We fill the queue with new fetched batch until we reach the max size.
            # boss_queue.put("queue_size=%d"%(boss_queue.qsize()), block=False)
            # print("I am rank %d, queue size = %d"%(rank, boss_queue.qsize()))
            if BQ_thread_killer() == True:
                print("BQ rank %d, exiting" % (rank))
                return
            # total_samples, total_success = sampler.get_assist_info()
            total_samples = assistant.total_samples
            total_success = assistant.total_success
            print("assist [pass/resist/total/rate] [{}/{}/{}/{:.5f}]".format(
                total_success, 
                total_samples-total_success, 
                total_samples,
                total_success/total_samples))
    # print("I am rank %d, exiting" % (rank))
    return


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.train_data = loader.loader_train.dataset.datasets[0]
        # self.train_data = loader.loader_train.dataset
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

        BQ_workers = 1
        AQ_workers = 1

        
        self.sampler = AssistantSampler(self.train_data, base_prob=args.base_prob)

        # Produce batches in BossQueue
        self.processes = []
        for i in range(BQ_workers):
            t = multiprocessing.Process(
                target=BQ_feeder, args=(i, args, self.train_data, self.sampler, total_samples, total_success)
            )
            t.start()
            self.processes.append(t)

        # Wait for the Assistants to init
        while boss_queue.qsize() < 10:
            time.sleep(1)
        self.t0 = time.time()

        # Init statistics
        self.batches_per_epoch = len(self.train_data) // args.batch_size + 1
        self.total_time = 0
        self.loss_sum = 0.0
        self.loss_square_sum = 0.0
        self.total_instances = 0
        self.psnr_mean = 0.0
        self.alpha = 0.75 # EMA (exponential moving average) decay factor

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        # self.loader_train.dataset.set_scale(0)
        for batch_idx in range(self.batches_per_epoch):
        # for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = boss_queue.get(block=True)
            lr_cpu = lr
            hr_cpu = hr
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            losses = self.loss(sr, hr)
            loss_sum = losses.sum()
            loss_sum.backward()

            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if self.args.sec_method == 'mean':
                # exponential moving average
                self.total_instances *= self.alpha
                self.total_instances += (1.0 - self.alpha) * lr.size(0)

                self.loss_sum *= self.alpha
                self.loss_sum += (1.0 - self.alpha) * loss_sum.item()

                self.loss_mean = self.loss_sum / self.total_instances

                feed = (losses.detach().cpu().numpy(), self.loss_mean)
                assistant_queue.put(((lr_cpu, hr_cpu), feed))
            # if self.args.sec_method == 'upsample_PSNR' or self.args.sec_method == 'downsample_PSNR':
            #     feed = sr.detach().cpu()
            #     assistant_queue.put(((lr_cpu, hr_cpu), feed))

            if (batch_idx + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch_idx + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch_idx),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()
            break

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
                for lr, hr, filename, index in tqdm(d, ncols=80):
                    lr_cpu = lr
                    hr_cpu = hr
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    psnr = utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=d)
                    self.ckp.log[-1, idx_data, idx_scale] += psnr

                    if self.args.sec_method == 'upsample_PSNR' or self.args.sec_method == 'downsample_PSNR':
                        self.psnr_mean *= self.alpha
                        self.psnr_mean += (1.0 - self.alpha) * psnr

                        feed = (self.psnr_mean, scale, self.args.rgb_range)

                        assistant_queue.put(((lr_cpu, hr_cpu), feed))

                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

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
            if epoch > self.args.epochs: # when should terminate training
                BQ_thread_killer.set_tokill(True)
                cuda_transfers_thread_killer.set_tokill(True)
                for p in self.processes:
                    self.ckp.write_log("Terminating process")
                    p.terminate()
                self.ckp.write_log(
                    "Training %d batches done in %f seconds"
                    % (self.batches_per_epoch * self.args.epochs, time.time() - self.t0)
                )
            return epoch > self.args.epochs

