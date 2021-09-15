import torch
import torch.nn as nn
from torch.utils.data.sampler import Sampler
import numpy as np
import utility
from torch.nn.functional import interpolate

# eps = 1e-10

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x)) # shift x
    return e_x / e_x.sum(axis=0)  # only difference


def sigmoid(x):
    """Compute softmax values for each sets of scores in x."""
    if x.all() >= 0:  # avoid value overflow
        return 1.0 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))

class AssistantSampler(Sampler):
    r""" Generate instances based on Assistant model.
    Arguments:
        data_source (Dataset): dataset to sample from
        base_prob (float): ground probability for an instance to pass
        num_classes (int): number of classes for classification
    """

    def __init__(self, data_source, base_prob=0.3):
        self.data_source = data_source

        lr, hr, _, _ = data_source[0]
        lr_dim = torch.numel(lr)
        hr_dim = torch.numel(hr)

        self.assistant = AssistantModelBinary(lr_dim, hr_dim)

        self.base_prob = base_prob

        self.total_samples = 1
        self.total_success = 1
        self.total_resist = 1
        # self.threadLock = threading.Lock()
        # self.sec_loss = 10.0 # TODO useless?

    def __iter__(self):
        # given idx i, return an instance
        index = -1
        while True:
            index = (index + 1) % len(self.data_source)
            coin = np.random.uniform()
            # coin2 = np.random.uniform()
            self.total_samples = self.total_samples + 1  # TODO not thread safe!!
            x, y, _, _ = self.data_source[index]
            # x = x.view(-1).numpy()
            if coin < self.base_prob:
                # self.total_success += 1  # not thread safe
                self.total_success = self.total_success + 1  # not thread safe
                yield index
            else:
                # continue
                coin = (coin - self.base_prob) / (
                    1.0 - self.base_prob
                )  # renormalize coin, still independent variable

                # compute importance
                keep_prob = self.assistant.get_importance(x, y)
                if coin < keep_prob:
                    self.total_success = self.total_success + 1  # not thread safe
                    yield index

    def __len__(self):
        return len(self.data_source)

    def get_assist_info(self):
        return self.total_samples, self.total_success
    
    def threshold(self): # TODO exponential moving average on accuracy
        pass

    def train_step(self, X, Y, feed, method="mean"):
        # self.sec_loss *= 0.0
        acc, probs = self.assistant.train_step(
            X, Y, feed, method=method
        )
        # self.sec_loss += 1 * acc
        return probs


class AssistantModelBinary(nn.Module):
    """
    predict p( not_trivial | x_i,  y_i) = sigmoid( W*x_i + U*y_i + B )
        where:
            not_trivial = ( loss_i > loss_mean)
    Arguments:
        lr_dim (int): input data vector dimension (LR)
        hr_dim (int): input data vector dimension (HR)
        num_classes (int): number of classes for classification
    """

    def __init__(self, lr_dim, hr_dim, lr=0.01, base_prob=0.3):
        super(AssistantModelBinary, self).__init__()
        self.lr_dim = lr_dim
        self.hr_dim = hr_dim
        self.lr = lr # learning rate
        self.lam = 1e-3
        self.fitted = 0
        self.base_prob = base_prob
        self.total_samples = 0
        self.total_success = 0
        self.total_resist = 0

        self.W = 0.001 * np.random.randn(lr_dim).astype(np.float32) # shape: (lr_dim,)
        self.U = 0.001 * np.random.randn(hr_dim).astype(np.float32) # shape: (hr_dim,)
        self.B = 0.001

    def get_importance(self, x, y): # apply on a single instance
        return sigmoid(self.W.dot(x.view(self.lr_dim)) + self.U.dot(y.view(self.hr_dim)) + self.B)

    def make_target_by_mean(self, loss, mean):
        return np.array(loss > mean, dtype=int)

    def make_target_by_PSNR(self, lr, hr, psnr_mean, scale, rgb_range, sample='upsample'):
        psnr = []
        for i in range(lr.shape[0]):
            if sample == 'upsample':
                img2 = hr[i]
                scale_factor = int(hr.shape[-1]/lr.shape[-1])
                img1 = interpolate(lr[i].unsqueeze(0), scale_factor=scale_factor, mode='bilinear').clamp(min=0, max=255)
            elif sample == 'downsample':
                img1 = lr[i]
                scale_factor = int(lr.shape[-1]/hr.shape[-1])
                img2 = interpolate(hr[i].unsqueeze(0), scale_factor=scale_factor, mode='bilinear').clamp(min=0, max=255)
            else:
                raise ValueError("specify upsample or downsample")
            psnr.append(utility.calc_psnr(img1, img2, scale, rgb_range))
        psnr = np.array(psnr)
        return np.array(psnr > psnr_mean, dtype=int)

    def train_step(self, X, Y, feed, method): # apply on a batch
        self.fitted += 1
        batch_size = Y.shape[0]
        lr = self.lr / Y.shape[0]

        if method == "mean":
            loss = feed[0]
            mean = feed[1]
            label = self.make_target_by_mean(loss, mean).reshape(
                (batch_size,)
            )
        elif method == 'upsample_PSNR':
            print("yes")
            psnr_mean, scale, rgb_range = feed
            label = self.make_target_by_PSNR(X, Y, psnr_mean, scale, rgb_range, sample='upsample')
        elif method == 'downsample_PSNR':
            psnr_mean, scale, rgb_range = feed
            label = self.make_target_by_PSNR(X, Y, psnr_mean, scale, rgb_range, sample='downsample')
        # elif method == "pred":
        #     label = self.make_target_by_pred(loss, mean, dev, Y, pred).reshape(
        #         (batch_size,)
        #     )
        else: 
            raise ValueError("the shrinking method {} is not supported!".format(method))

        X = X.reshape([batch_size, self.lr_dim])
        Y = Y.reshape([batch_size, self.hr_dim])

        prob = sigmoid(X.numpy().dot(self.W) + Y.numpy().dot(self.U) + self.B)  # shape = (batch_size,)
        predict = np.array(prob > 0.5, dtype=int)
        hit = np.sum(label == predict) * 1.0
        acc = hit / batch_size

        grad = prob - label
        # gradient update

        self.W -= lr * (grad.dot(X) + self.lam * self.W).squeeze()
        self.U -= lr * (grad.dot(Y) + self.lam * self.U).squeeze()
        self.B -= lr * (grad.mean() + self.lam * self.B)

        return acc, prob

    def evaluate(self, lr, hr):
        '''evluate whether the pair of (lr,hr) is nessary to train'''
        coin = np.random.uniform()
        self.total_samples = self.total_samples + 1  # TODO not thread safe!!
        if coin < self.base_prob:
            self.total_success = self.total_success + 1  # not thread safe
            return True
        else:
            # continue
            # renormalize coin, still independent variable
            coin = (coin - self.base_prob) / (1.0 - self.base_prob)  

            # compute importance
            keep_prob = self.get_importance(lr, hr)
            if coin < keep_prob:
                self.total_success = self.total_success + 1  # not thread safe
                return True
        
        return False