# for i in range(6,-1,-1):
#     print(i)

# for i in reversed(range(7)):
#     print(i)

# for i in range(1,8):
#     print(i)

# a = "1".rjust(4, '0')
# print(a)

# import pickle


# file = "/data/shizun/DIV2K/bin/DIV2K_train_LR_bicubic/X2/easy_x2_descending.pt"
# with open(file, 'rb') as _f:
#     a = pickle.load(_f)

# img, iy, ix = a[0]
# print(img)
# print(iy)
# print(ix)

# print(300//3)


# import torch

# import utility
# import data
# import model
# import loss
# from option import args
# # if args.assistant:
# #     from trainer_autoassist import Trainer
# # else:
# if args.dynamic:
#     from trainer_dynamic import Trainer
# if args.switchable:
#     from trainer_switchable import Trainer
# else:
#     from trainer import Trainer
# import os
# # if torch.cuda.is_available():
# os.environ["CUDA_VISIBLE_DEVICES"] = args.device
# torch.manual_seed(args.seed)
# checkpoint = utility.checkpoint(args)

# def main():
#     global model
#     if args.data_test == ['video']:
#         from videotester import VideoTester
#         model = model.Model(args, checkpoint)
#         t = VideoTester(args, model, checkpoint)
#         t.test()
#     else:
#         if checkpoint.ok:
#             _model = model.Model(args, checkpoint)
#             utility.print_params(_model,checkpoint,args)

#             if args.switchable:
#                 data_list = ('easy_x2_descending', 'midd_x2_descending', 'hard_x2_descending')
#                 # args.epochs = args.epochs // len(data_list)
#                 for data_part, cap_mult in zip(data_list, args.cap_mult_list):
#                     args.file_suffix = data_part
#                     loader = data.Data(args)
#                     _model.apply(lambda m: setattr(m, 'cap_mult', cap_mult))
#                     _loss = loss.Loss(args, checkpoint) if not args.test_only else None
#                     t = Trainer(args, loader, _model, _loss, checkpoint)
#                     while not t.terminate():
#                         t.train()
#                         t.test()
                    
#             else:
#                 loader = data.Data(args)
#                 _loss = loss.Loss(args, checkpoint) if not args.test_only else None
#                 t = Trainer(args, loader, _model, _loss, checkpoint)
#                 while not t.terminate():
#                     t.train()
#                     t.test()

#             checkpoint.done()

# if __name__ == '__main__':
#     main()


# print(101 // 100)

# def func():
#     a = []
#     a.append(1)
#     a.append(2)
#     # print(a)
#     return a


# for i in range(1,9):
#     print(i)

# a = list(range(3,32,4))
# print(a)

# import torch
# import torch.nn as nn

# body = [nn.Conv2d(3,3,3) for _ in range(9)]
# body = nn.Sequential(*body)

# x = torch.ones(2,3,24,24)
# print(x.shape)
# my_body = body[:4]
# print(my_body)
# y = my_body(x)
# print(y.shape)
# print(body)
# y = body(x)
# print(y.shape)




# from option import args
# import model
# import utility
# # import loss
# import torch
# import torch.nn as nn

# checkpoint = utility.checkpoint(args)
# _model = model.Model(args, checkpoint)
# # print(_model.model.head)
# # for m in _model.model.body[0].parameters():
# #     # print(m)
# #     m.requires_grad = False
# # for name, param in _model.named_parameters():
# #     print(name, param.size(), param.requires_grad)
# # _loss = loss.Loss(args, checkpoint) if not args.test_only else None
# x = torch.randn(2,3,96,96).to("cuda")
# g = torch.randn(2,3,192,192).to("cuda")
# y = _model(x, 0)
# f = nn.L1Loss()
# l = f(y,g)
# l.backward()


# a = 92 + 89 + 92 + 82 + 81 + 86 + 87 + 98 + 97 + 75 + 93 + 86 + 100
# a = 2*92 + 2*89 + 92 + 82 + 81 + 2*86 + 3*87 + 98 + 2*97 + 2*75 + 2*93 + 2*86

# a = a / (2+2+1+1+1+2+3+1+2+2+2+2)

# print(a)

# for i in range(32):
#     if i % 1 == 0:
#         print(i)

# import  socket
# hostname = socket.gethostname() # 获取当前主机名
# print(hostname)

# import numpy as np
# # 坐标向量
# a = np.array([1,2,3])
# # 坐标向量
# b = np.array([7,8])
# # 从坐标向量中返回坐标矩阵
# # 返回list,有两个元素,第一个元素是X轴的取值,第二个元素是Y轴的取值
# res = np.meshgrid(a,b)
# print(res)

# import glob
# from pprint import pprint

# a = glob.glob("/data/shizun/DIV2K/bin/DIV2K_train_LR_bicubic/X2/*_Canny_p192_s24.pt")
# pprint(a)

# from model.edsr_dynamic import make_model
# from option import args
# import torch

# edsr = make_model(args)

# edsr.eval()

# input = torch.ones((1,3,64,64))

# output = edsr(input)

# import torch.nn.functional as F
# import torch
# a = torch.tensor(3313.87)
# print(a)
# b = a.pow(2).mean()
# print(b)
# c = torch.tanh(b)
# print(c)
# d = torch.sigmoid(b)
# print(d)
# e = torch.log10(256*256/a)
# print(e)
# c = torch.tanh(e)
# print(c)
# d = torch.sigmoid(e)
# print(d)
# print(torch.sigmoid(torch.tensor(1)))
# print(torch.sigmoid(torch.tensor(2)))
# print(torch.sigmoid(torch.tensor(3)))
# print(torch.sigmoid(torch.tensor(4)))
# print(torch.sigmoid(torch.tensor(5)))

# import torch
# from torch.utils.tensorboard import SummaryWriter

# psnr_log_file = "/home/shizun/experiment/20211101_EDSR_dynamic_match1_x2_e300_ps192_lr0.0001_n32_i1_sTrue_b96_t1/psnr_log.pt"

# with open(psnr_log_file, 'rb') as _f:
#     psnr_log = torch.load(_f)
# new_psnr_log = psnr_log.squeeze()

# save_dir = "temp/{}".format(psnr_log_file.split('/')[-2])

# for bin_index in range(new_psnr_log.shape[0]):
#     writer = SummaryWriter('{}/bin{}'.format(save_dir, bin_index))
#     for i, psnr in enumerate(new_psnr_log[bin_index]):
#         writer.add_scalar('multi-exit PSNR', psnr, global_step=i)


# print((3-3)//4)
# print((7-3)//4)

# import numpy as np
# h = 1000
# crop_sz = 100
# step = 80

# h_space = np.arange(0, h - crop_sz + 1, step)
# print(h_space)

# import pickle
# import numpy as np
# statistics_file = "/data/shizun/DIV2K/bin/DIV2K_train_LR_bicubic/X2/statistics_Canny_p192_s24.pt"
# statistics_val_file = "/data/shizun/DIV2K/bin/DIV2K_train_LR_bicubic/X2/statistics_val_Canny_p192_s24.pt"

# with open(statistics_file, 'rb') as _f:
#     statistics = pickle.load(_f) # all_id_iy_ix_metric

# with open(statistics_val_file, 'rb') as _f:
#     statistics_val = pickle.load(_f) # all_id_iy_ix_metric

# level = []
# for i in range(10):
#     level.append(statistics[len(statistics)//10*i][-1])
# print(level) # [0, 0, 53040, 131835, 212160, 291465, 370770, 453390, 541110, 656115]


# metric = statistics_val[:,-1]
# indexs = []
# for i in level:
#     index = np.argwhere(metric >= i)[0][0]
#     indexs.append(index)
# print(indexs) # [0, 0, 1665, 2846, 4227, 5174, 5979, 6724, 7534, 8537]
# indexs.append(len(metric))
# print(indexs)

# hist = []
# for i in range(len(indexs)-1):
#     hist.append(indexs[i+1]-indexs[i])
# print(hist) # [0, 1665, 1181, 1381, 947, 805, 745, 810, 1003, 1369]

# print(np.sum(np.array(hist)))

# a = "RCAN_decision"

# print(a.find("decision"))


# import numpy as np
# import torch
# a = np.zeros((1,8,8,8))
# print(len(a))
# # print(a.shape)
# b = np.concatenate((a,a))
# print(len(b))
# print(b.shape)

# a = torch.ones(6)
# print(a.shape)
# print(a[0:5])
# print(a[5:10])

# print(6//5 + 1)

# a = "c"

# if a in ["a", "b"]:
#     print(1)

import torch
# import utility

# index = torch.arange(0,32,1).unsqueeze(0)
# index = torch.ones(32).unsqueeze(0)
# a = torch.cat((index,index))
# # print(a.shape)
# # b = a.sum(dim=0)
# # print(b)

# print(utility.calc_avg_exit(a))


# flops_list = torch.Tensor([4.27,5.47,6.68,7.89,9.10,10.31,11.52,12.72,13.93,15.14,16.35,17.56,18.77,19.98,21.18,22.39,23.60,24.81,26.02,27.23,28.44,29.64,30.85,32.06,33.27,34.48,35.69,36.89,38.10,39.31,40.52,41.73])
# print(flops_list)
# flops_list = flops_list[3::4]
# print(flops_list)

# b = 4
# depth = torch.empty([b, 1]).uniform_(0, 32)
# x = torch.ones((b,1,6,6))
# x = x * depth.view(-1, 1, 1, 1)
# # print(x)

# print(depth.mean(dim=0))


# a = torch.arange(0,10)*1.
# b = torch.ones(10)*5.
# print(a)
# print(b)
# if len(torch.where(a>b)[0]) >0:
#     print(len(torch.where(a>b)[0]))
#     # print(torch.nonzero(a>b))
# a[[1,2,3]] = b[[1,2,3]]
# print(a)

# a = torch.arange(0,80)
# print(a)


# a = torch.ones(2)
# b = torch.cat((a,a))
# print(b)
import os

# exp_path = "/data/shizun/experiment/20220223_RCAN_my_x2_e300_ps192_lr0.0001/"
# exp_path = "/data/shizun/experiment/20220224_RCAN_my_x3_e300_ps192_lr0.0001/"
exp_path = "/data/shizun/experiment/20220222_RCAN_my_x4_e300_ps192_lr0.0001/"
model_path = os.path.join(exp_path, "model/model_best.pt")
save_path = os.path.join(exp_path, "model/model_best_old.pt")
stat_dict = torch.load(model_path)
# print(stat_dict)
torch.save(stat_dict, save_path, _use_new_zipfile_serialization=False)

# import torch

# a = torch.arange(16).reshape(4,4)
# print(a)

# b = torch.argmax(a,dim=1)
# # b = [3,2,2,3]
# print(b)


# # for i in b

# print(a[:,b])

# a = [1,2,3,4,5]

# print(a[:3])
# print(a[-3:])

# import torch

# a = torch.linspace(1, 0, 5).unsqueeze(0).transpose(1,0).unsqueeze(0)
# print(a)
# # print(a.T)
# b = torch.ones(3,5,8)
# print(b)

# # b = b*a
# b = b[None,:,:] * a

# a = torch.ones((3,192,192))

# print(a[:, :, :8].shape)


# print(b)


# a = torch.ones((10,3,32,32))
# mse = a.mean((-1,-2,-3))
# print(mse)
# print(mse.shape)
# print(mse.squeeze().shape)

# a = torch.arange(1,10,1).unsqueeze(0).T.float()
# a = torch.arange(1,10,1).float()
# print(a)
# b = torch.arange(2,11,1).float()

# f = torch.nn.MSELoss()

# c = f(a,b)
# print(c)