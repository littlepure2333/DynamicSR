import torch
from . import networks as N
import torch.nn as nn
import math
from . import losses as L



class base_SRModel(nn.Module):
    def __init__(self, opt):
        super(base_SRModel, self).__init__()

        self.opt = opt
        self.lambda_pred = opt.lambda_pred
        self.nc_adapter = opt.nc_adapter
        self.multi_adapter = opt.multi_adapter
        self.constrain = opt.constrain
        self.with_depth = opt.with_depth
        self.scale = opt.scale

        if self.nc_adapter > 0 and self.multi_adapter:
            assert self.n_blocks == self.nc_adapter

        n_feats = opt.n_feats
        n_upscale = int(math.log(opt.scale, 2))

        m_head = [N.MeanShift(),
                  N.conv(opt.input_nc, n_feats, mode='C')]
        self.head = N.seq(m_head)

        for i in range(self.n_blocks):
            setattr(self, '%s%d'%(self.block_name, i), self.block(
                n_feats, n_feats, res_scale=opt.res_scale, mode=opt.block_mode,
                clamp=self.clamp_wrapper(i) if self.nc_adapter != 0 else None,
                channel_attention=opt.channel_attention,
                sparse_conv=opt.sparse_conv,
                n_resblocks=opt.n_resblocks,
                clamp_wrapper=self.clamp_wrapper,
                side_ca=opt.side_ca))
        self.body_lastconv = N.conv(n_feats, n_feats, mode='C')

        if opt.scale == 3:
            m_up = N.upsample_pixelshuffle(n_feats, n_feats, mode='3')
        else:
            m_up = [N.upsample_pixelshuffle(n_feats, n_feats, mode='2') \
                    for _ in range(n_upscale)]
        self.up = N.seq(m_up)

        m_tail = [N.conv(n_feats, opt.output_nc, mode='C'),
                  N.MeanShift(sign=1)]
        self.tail = N.seq(m_tail)

        self.isTrain = opt.isTrain
        self.loss = opt.loss

    def forward_main_tail(self, x, pred):
        res = x
        for i in range(self.n_blocks):
            if self.nc_adapter <= 1 and not self.multi_adapter:
                res = getattr(self, '%s%d'%(self.block_name, i))(
                        res, pred)
            elif self.multi_adapter:
                setattr(self, 'pred%d'%i,
                        getattr(self, 'predictor%d'%i)(res,
                                depth if self.with_depth else None))
                res = getattr(self, '%s%d'%(self.block_name, i))(
                        res, getattr(self, 'pred%d'%i))
            else:
                res = getattr(self, '%s%d'%(self.block_name, i))(
                        res, pred[:, i:i+1, ...])
        res = self.body_lastconv(res)
        res += x

        res = self.up(res)
        res = self.tail(res)
        return res

    def forward_chop(self, x, pred, shave=10, min_size=160000):
        scale = self.scale
        n_GPUs = len(self.opt.gpu_ids)
        n, c, h, w = x.shape
        h_half, w_half = h//2, w//2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[..., 0:h_size, 0:w_size],
            x[..., 0:h_size, (w - w_size):w],
            x[..., (h - h_size):h, 0:w_size],
            x[..., (h - h_size):h, (w - w_size):w]
        ]
        pred_list = [
            pred[..., 0:h_size, 0:w_size],
            pred[..., 0:h_size, (w - w_size):w],
            pred[..., (h - h_size):h, 0:w_size],
            pred[..., (h - h_size):h, (w - w_size):w]
        ]
        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i+n_GPUs)], dim=0)
                pred_batch = torch.cat(pred_list[i:(i+n_GPUs)], dim=0)
                res = self.forward_main_tail(lr_batch, pred_batch)
                sr_list.extend(res.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self.forward_chop(lr_, pred_, shave, min_size) \
                    for lr_, pred_ in zip(lr_list, pred_list)]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale
        c = sr_list[0].shape[1]

        output = x.new(n, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size,
                               (w_size - w + w_half):w_size]
        return output



    def forward(self, x, hr=None, depth=None, FLOPs_only=False, chop=False):
        x = self.head(x)


        x = self.forward_main_tail(x, pred)

        if self.isTrain:
            criterion1 = getattr(self, 'criterion%s'%self.loss)
            loss1 = criterion1(x, hr)
            if self.nc_adapter != 0:
                if self.constrain == 'none':
                    loss_Pred = self.lambda_pred * pred.abs()
                    loss = loss1 + loss_Pred
                elif self.constrain == 'soft':
                    if self.multi_adapter:
                        pred = torch.cat([getattr(self, 'pred%d'%i) \
                             for i in range(self.nc_adapter)], dim=1)
                        loss_Pred = self.lambda_pred * \
                            (pred.mean((2,3)) - depth).clamp_min_(0).sum(dim=1)
                    else:
                        loss_Pred = self.lambda_pred * \
                            (pred.mean((2,3)) - depth).clamp_min_(0).mean(dim=1)
                            #(pred.mean((1,2,3)) - depth).clamp_min_(0)
                    # loss_Pred = self.lambda_pred * \
                    #             (pred.mean((1,2,3)) - depth).abs()
                    loss = loss1 + loss_Pred
                else:
                    loss = loss1
                    loss_Pred = torch.zeros_like(loss1)
                return x, pred, loss1, loss_Pred, loss
            return x, pred, loss1, loss1
        else:
            if self.multi_adapter:
                pred = torch.cat([getattr(self, 'pred%d'%i) \
                                  for i in range(self.nc_adapter)], dim=1)
        return x, pred