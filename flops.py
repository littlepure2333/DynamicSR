from torchstat import stat
from option import args # modify default template in option.py

# edsr_decision
# from model.edsr_decision import make_model
# args.model = 'EDSR_decision'
# args.n_resblocks = 32
# args.n_feats = 256
# args.res_scale = 0.1
# args.scale = [3]

# edsr
# from model.edsr import make_model
# args.model = 'EDSR'
# args.n_resblocks = 32
# args.n_feats = 256
# args.res_scale = 0.1
# args.scale = [4]

# rcan_decision
# from model.rcan_decision import make_model
# args.model = 'RCAN_decision'
# args.n_resgroups = 10
# args.n_resblocks = 20
# args.n_feats = 64
# args.scale = [2]

# rcan
# from model.rcan import make_model
# args.model = 'RCAN'
# args.n_resgroups = 10
# args.n_resblocks = 20
# args.n_feats = 64
# args.scale = [3]

# fsrcnn_decision
from model.fsrcnn_decision import make_model
args.model = 'fsrcnn_decision'
args.scale = [3]

# fsrcnn
# from model.fsrcnn import make_model
# args.model = 'fsrcnn'
# args.scale = [2]


m = make_model(args)
stat(m, (3, 32, 32))
# print(m)


'''
EDSR x2: (3,32,32)
    head: 7,401,472
    body: 38,671,745,024 = 32 * 1,208,483,840
    tail: 3,049,533,440
    eedm: 256
    total:41,728,679,936 (41.73GFlops)
(4.27,5.47,6.68,7.89,9.10,10.31,11.52,12.72,13.93,15.14,16.35,17.56,18.77,19.98,21.18,22.39,23.60,24.81,26.02,27.23,28.44,29.64,30.85,32.06,33.27,34.48,35.69,36.89,38.10,39.31,40.52,41.73)

EDSR x3: (3,32,32)
    head: 7,462,912
    body: 38,671,745,024 = 32 * 1,208,483,840
    tail: 6,106,147,840
    eedm: 256
    total:44,785,355,776 (44.79GFlops)
(7.32,8.53,9.74,10.95,12.16,13.36,14.57,15.78,16.99,18.20,19.41,20.62,21.82,23.03,24.24,25.45,26.66,27.87,29.07,30.28,31.49,32.70,33.91,35.12,36.33,37.53,38.74,39.95,41.16,42.37,43.58,44.79)

EDSR x4: (3,32,32)
    head: 7,548,928
    body: 38,671,745,024 = 32 * 1,208,483,840
    tail: 12,802,375,680
    eedm: 256
    total:51,481,669,888.0 (51.48GFlops)
(14.02,15.23,16.44,17.64,18.85,20.06,21.27,22.48,23.69,24.89,26.10,27.31,28.52,29.73,30.94,32.15,33.35,34.56,35.77,36.98,38.19,39.40,40.61,41.81,43.02,44.23,45.44,46.65,47.86,49.06,50.27,51.48)
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
FSRCNN x2: (3,32,32)
    head: 5,058,560
    body: 5,357,568 = 4 * 1,339,392
    tail: 745,472
    eedm: 12
    total:11,161,600 (11.16MFlops)


FSRCNN x3: (3,32,32)
    head: 
    body: 
    tail: 
    eedm: 
    total:


FSRCNN x4: (3,32,32)
    head: 
    body: 
    tail: 
    eedm: 
    total:


'''



print(4300800+57344+688128+12288)
print(688128+57344)


# for i in range(10):
#     b = 2043904+822460416+64+(i+1)*1551534086
#     print("{:.2f}".format(b/1000000000.0))

