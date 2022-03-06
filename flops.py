from torchstat import stat
from option import args # modify default template in option.py
from mmcv.cnn import get_model_complexity_info
from thop import profile
import torch
import time

# edsr_decision
# from model.edsr_decision import make_model
# args.model = 'EDSR_decision'
# args.n_resblocks = 32
# args.n_feats = 256
# args.res_scale = 0.1

# edsr
# from model.edsr import make_model
# args.model = 'EDSR'
# args.n_resblocks = 32
# args.n_feats = 256
# args.res_scale = 0.1

# rcan_decision
# from model.rcan_decision import make_model
# args.model = 'RCAN_decision'
# args.n_resgroups = 10
# args.n_resblocks = 20
# args.n_feats = 64

# rcan
from model.rcan import make_model
args.model = 'RCAN'
args.n_resgroups = 10
args.n_resblocks = 20
args.n_feats = 64

# fsrcnn_decision
# from model.fsrcnn_decision import make_model
# args.model = 'fsrcnn_decision'

# fsrcnn
# from model.fsrcnn import make_model
# args.model = 'fsrcnn'

# vdsr_decision
# from model.vdsr_decision import make_model
# args.model = 'VDSR_decision'
# args.n_resblocks = 20
# args.n_feats = 64

# vdsr
# from model.vdsr import make_model
# args.model = 'VDSR'
# args.n_resblocks = 20
# args.n_feats = 64

# ecbsr
# from model.ecbsr_rep import make_model
# args.model = 'ECBSR_rep'
# args.m_ecbsr = 16
# args.c_ecbsr = 64
# args.dm_ecbsr = 2
# args.act = 'prelu'


# print flops
args.scale = [2]
m = make_model(args)
stat(m, (3, 48, 48))
# print(m)

# mmcv for RCAN
# input_shape = (64,32,32)
# m = m.body[0].body[0].body[3]
# # print(m)
# flops, params = get_model_complexity_info(m, input_shape)
# split_line = '=' * 30
# print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
# split_line, input_shape, flops, params))


# thop
# input = torch.randn(1, 3, 32, 32)
# macs, params = profile(m, inputs=(input, ))
# from thop import clever_format
# macs, params = clever_format([macs, params], "%.3f")
# print(params)


# input = torch.randn(1, 3, 32, 32)
# output = m(input)


'''
EDSR x2: (3,32,32)
    head: 7,401,472 (0.02%)
    body: 38,671,745,024 = 32 * 1,208,483,840 (92.6%)
    tail: 3,049,533,440 (7.3%)
    eedm: 256 （0.00%）
    total:41,728,679,936 (41.73GFlops)
    Total params: 40,729,627
(4.27,5.47,6.68,7.89,9.10,10.31,11.52,12.72,13.93,15.14,16.35,17.56,18.77,19.98,21.18,22.39,23.60,24.81,26.02,27.23,28.44,29.64,30.85,32.06,33.27,34.48,35.69,36.89,38.10,39.31,40.52,41.73)

EDSR x3: (3,32,32)
    head: 7,462,912 (0.02%)
    body: 38,671,745,024 = 32 * 1,208,483,840 （86.3%）
    tail: 6,106,147,840（13.6%）
    eedm: 256
    total:44,785,355,776 (44.79GFlops) 
    Total params: 43,680,027
(7.32,8.53,9.74,10.95,12.16,13.36,14.57,15.78,16.99,18.20,19.41,20.62,21.82,23.03,24.24,25.45,26.66,27.87,29.07,30.28,31.49,32.70,33.91,35.12,36.33,37.53,38.74,39.95,41.16,42.37,43.58,44.79)

EDSR x4: (3,32,32)
    head: 7,548,928 (0.01%)
    body: 38,671,745,024 = 32 * 1,208,483,840 (75.1%)
    tail: 12,802,375,680 (24.9%)
    eedm: 256
    total:51,481,669,888 (51.48GFlops)
    Total params: 43,089,947
(14.02,15.23,16.44,17.64,18.85,20.06,21.27,22.48,23.69,24.89,26.10,27.31,28.52,29.73,30.94,32.15,33.35,34.56,35.77,36.98,38.19,39.40,40.61,41.81,43.02,44.23,45.44,46.65,47.86,49.06,50.27,51.48)

EDSR x4: (3,40,40)
    head: 11,795,200
    body: 60,424,601,600 = 32 * 1,888,268,800
    tail: 20,003,712,000
    eedm: 256
    total:80,440,108,800 (80.44GFlops)
'''

'''
EDSR x2: (3,48,48)
    head: 17,243,136
    body: 87,010,836,480 = 32 * 2,719,088,640
    tail: 6,861,450,240
    eedm: 
    total:93,889,529,856 (93.89GFlops)
    list: 9.60,12.32,15.04,17.76,20.47,23.19,25.91,28.63,31.35,34.07,36.79,39.51,42.23,44.95,47.67,50.38,53.10,55.82,58.54,61.26,63.98,66.70,69.42,72.14,74.86,77.57,80.29,83.01,85.73,88.45,91.17,93.89

EDSR x3: (3,48,48)
    head: 17,381,376
    body: 87,010,836,480 = 32 * 2,719,088,640
    tail: 13,738,832,640
    eedm: 
    total:100,767,050,496 (100.77GFlops)
    list: 16.48,19.19,21.91,24.63,27.35,30.07,32.79,35.51,38.23,40.95,43.67,46.39,49.10,51.82,54.54,57.26,59.98,62.70,65.42,68.14,70.86,73.58,76.30,79.01,81.73,84.45,87.17,89.89,92.61,95.33,98.05,100.77


EDSR x4: (3,48,48)
    head: 17,574,912
    body: 87,010,836,480 = 32 * 2,719,088,640
    tail: 28,805,345,280
    eedm: 256
    total:115,833,756,672 (115.83GFlops)
    list: 31.54,34.26,36.98,39.70,42.42,45.14,47.86,50.58,53.29,56.01,58.73,61.45,64.17,66.89,69.61,72.33,75.05,77.77,80.49,83.20,85.92,88.64,91.36,94.08,96.80,99.52,102.24,104.96,107.68,110.40,113.11,115.83
'''


'''
RCAN x2: (3,32,32)
    head: 3,631,104 
    body: 15,515,340,864 = 10 * 1,551,534,086 
    tail: 196,161,536
    eedm: 64
    total:15,715,133,504 (15.72GFlops)
[1.75,3.30,4.85,6.41,7.96,9.51,11.06,12.61,14.16,15.72]

RCAN x3: (3,32,32)
    head: 3,723,264
    body: 15,515,340,864 = 10 * 1,551,534,086
    tail: 394,095,616
    eedm: 64
    total:15,913,129,024 (15.91GFlops)
[1.95,3.50,5.05,6.60,8.16,9.71,11.26,12.81,14.36,15.91]

RCAN x4: (3,32,32)
    head: 2,043,904
    body: 15,515,340,864 = 10 * 1,551,534,086
    tail: 822,460,416
    eedm: 64
    total:16,341,579,840 (16.34GFlops)
[2.38,3.93,5.48,7.03,8.58,10.13,11.69,13.24,14.79,16.34]
'''

'''
RCAN x2: (3,48,48)
    head: 4,267,008
    body: 34,913,273,920 = 10 * 3,491,327,392
    tail: 441,363,456
    eedm: 
    total:35,358,904,384 (35.36GFlops)
    list: 3.94,7.43,10.92,14.41,17.90,21.39,24.88,28.38,31.87,35.36


RCAN x3: (3,48,48)
    head: 4,181,248
    body: 34,913,273,920 = 10 * 3,491,327,392
    tail: 886,715,136
    eedm: 
    total:35,804,394,304 (35.8GFlops)
    list: 4.38,7.87,11.36,14.86,18.35,21.84,25.33,28.82,32.31,35.80


RCAN x4: (3,48,48)
    head: 4,598,784
    body: 34,913,273,920 = 10 * 3,491,327,392
    tail: 1,850,535,936
    eedm: 
    total:36,768,408,640 (36.77GFlops)
    list: 5.35,8.84,12.33,15.82,19.31,22.80,26.29,29.79,33.28,36.77

'''

'''
VDSR x2: (3,32,32)
    head: 7,704,576
    body: 5,357,568 = 18 * 151,519,232
    tail: 7,090,176
    eedm: 
    total:2,742,140,928 (2.74GFlops)


VDSR x3: (3,32,32)
    head: 17,335,296
    body: 6,136,528,896 = 18 * 340,918,272
    tail: 15,952,896
    eedm: 
    total:6,169,817,088 (6.17GFlops)


VDSR x4: (3,32,32)
    head: 30,818,304
    body: 10,909,384,704 = 18 * 606,076,928
    tail: 28,360,704
    eedm: 
    total:10,968,563,712.0 (10.97GFlops)
'''

'''
VDSR x2: (3,48,48)
    head: 17,335,296
    body: 6,136,528,896 = 18 * 340,918,272
    tail: 15,952,896
    eedm:
    total:6,169,817,088 (6.17GFlops)
    list: 0.37,0.72,1.06,1.40,1.74,2.08,2.42,2.76,3.10,3.44,3.78,4.12,4.47,4.81,5.15,5.49,5.83,6.17

VDSR x3: (3,48,48)
    head: 39,004,416
    body: 13,807,190,016 = 18 * 767,066,112
    tail: 35,894,016
    eedm:
    total:13,882,088,448 (13.88GFlops)
    list: 0.84,1.61,2.38,3.14,3.91,4.68,5.44,6.21,6.98,7.75,8.51,9.28,10.05,10.81,11.58,12.35,13.12,13.88

VDSR x4: (3,48,48)
    head: 69,341,184
    body: 24,546,115,584 = 18 * 1,363,673,088
    tail: 63,811,584
    eedm:
    total:24,679,268,352 (24.68GFlops)
    list: 1.50,2.86,4.22,5.59,6.95,8.32,9.68,11.04,12.41,13.77,15.13,16.50,17.86,19.22,20.59,21.95,23.32,24.68
'''

'''
ECBSR x2: (3,32,32)
    head: 
    body: 
    tail: 
    eedm:
    total:


ECBSR x3: (3,32,32)
    head: 
    body: 
    tail: 
    eedm:
    total:


ECBSR x4: (3,32,32)
    head: 1,851,392
    body: 605,028,352 = 16 * 37,814,272
    tail: 28,360,704
    eedm:
    total:635,240,448 (635.24MFlops)
'''

'''
ECBSR x2: (3,48,48)
    head: 4,137,984
    body: 1,361,313,792 = 16 * 85,082,112
    tail: 15,952,896
    eedm:
    total:1,381,404,672 (1.38GFlops)
    list: 0.11,0.19,0.28,0.36,0.45,0.53,0.62,0.70,0.79,0.87,0.96,1.04,1.13,1.21,1.30,1.38

ECBSR x3: (3,48,48)
    head: 4,149,504
    body: 1,361,313,792 = 16 * 85,082,112
    tail: 35,894,016
    eedm:
    total:1,401,357,312 (1.4GFlops)
    list: 0.13,0.21,0.30,0.38,0.47,0.55,0.64,0.72,0.81,0.89,0.98,1.06,1.15,1.23,1.32,1.40

ECBSR x4: (3,48,48)
    head: 4,165,632
    body: 1,361,313,792 = 16 * 85,082,112
    tail: 63,811,584
    eedm:
    total: 1,429,291,008 (1.43GFlops)
    list: 0.15,0.24,0.32,0.41,0.49,0.58,0.66,0.75,0.83,0.92,1.00,1.09,1.17,1.26,1.34,1.43
'''


'''
FSRCNN x2: (3,32,32)
    head: 5,058,560
    body: 5,357,568 = 4 * 1,339,392
    tail: 745,472
    eedm: 12
    total:11,161,600 (11.16MFlops)
'''

    # head: 4,267,008
    # body: 34,913,273,920 = 10 * 3,491,327,392
    # tail: 441,363,456

# for i in range(10):
#     b = 4267008+441363456+(i+1)*3491327392
#     print("{:.2f}".format(b/1000000000.0))

