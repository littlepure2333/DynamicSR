machine  = {"A3":"/data/shizun/dataset/", 
            "B3":"/data/shizun/dataset/", 
            "C4":"/data/shizun/dataset/",
            "4gpu-2":"/home/shizun/datasets/image_process/",
            "4gpu-4":"/data/shizun/",
            "4gpu-5":"/data/shizun/"}

import datetime
import  socket
hostname = socket.gethostname() # 获取当前主机名
dir_data = machine[hostname]
today = datetime.datetime.now().strftime('%Y%m%d')

def set_template(args):
    # Set the templates here
    if args.template == 'EDSR':
        # model
        args.model = 'EDSR'   ################
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1
        
        # data
        args.scale = "2"   #############
        args.dir_data = dir_data
        args.ext = "sep"
        # args.data_range = '1-16/801-802'
        args.patch_size = 192

        # device
        args.device = "0,1,2,3"  #############
        args.n_GPUs = 4

        # pipeline
        args.epochs = 300
        args.lr = 1e-4
        args.batch_size = 16
        args.print_every = 10
        args.decision = True
        args.exit_interval = 2    ############

        # experiemnt
        args.reset = True
        # args.save_results = True  ############
        # args.save_gt = True      ############
        args.pre_train = "/home/shizun/experiment/20210929_EDSR_paper_x2_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211023_EDSR_paper_x3_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211023_EDSR_paper_x4_e300_ps192_lr0.0001/model/model_best.pt"
        args.save = "{}_{}_x{}_e{}_ps{}_lr{}_n{}_i{}_s{}_sum_pretrain_de3".format(today, args.model, args.scale, args.epochs, args.patch_size, args.lr, args.n_resblocks, args.exit_interval, args.shared_tail)

        # resume
        args.load = "20211103_EDSR_paper_x4_e600_ps192_lr0.0001"
        # args.load = "20211103_EDSR_paper_x3_e600_ps192_lr0.0001"
        # args.load = "20211103_EDSR_paper_x2_e600_ps192_lr0.0001"
        args.resume = -1

    elif args.template == 'EDSR_decision':
        # model
        args.model = 'EDSR_decision'   ################
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1
        
        # data
        args.scale = "2"   #############
        args.dir_data = dir_data
        args.ext = "sep"
        # args.data_range = '1-16/801-802'
        args.patch_size = 192

        # device
        args.device = "0,1,2,3"  #############
        args.n_GPUs = 4

        # pipeline
        args.epochs = 300
        args.lr = 1e-4
        args.batch_size = 16
        args.print_every = 10
        args.decision = True
        args.exit_interval = 2    ############

        # experiemnt
        args.reset = True
        # args.save_results = True  ############
        # args.save_gt = True      ############
        args.pre_train = "/home/shizun/experiment/20210929_EDSR_paper_x2_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211023_EDSR_paper_x3_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211023_EDSR_paper_x4_e300_ps192_lr0.0001/model/model_best.pt"
        args.save = "{}_{}_x{}_e{}_ps{}_lr{}_n{}_i{}_s{}_sum_pretrain_de3".format(today, args.model, args.scale, args.epochs, args.patch_size, args.lr, args.n_resblocks, args.exit_interval, args.shared_tail)

    elif args.template == 'EDSR_dytest':
        # model
        args.model = 'EDSR_decision'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

        # data
        # args.scale = "2"   #############
        args.dir_data = dir_data
        # args.data_test = 'TEST8K'  #############
        # args.data_test = 'DIV2K'  #############
        if args.data_test == 'DIV2K':
            args.data_range = '801-900'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        args.ext = "sep"
        args.patch_size = 32*int(args.scale)
        args.step = 30*int(args.scale)

        # device
        # args.device = "1"  #############
        args.n_GPUs = 1

        # pipeline
        args.dynamic = True
        args.test_only = True     ############
        # args.exit_interval = 1    ############
        # args.exit_threshold = 0.8 ############

        # experiment
        args.reset = True
        # args.save_results = True  ############
        # args.save_gt = True     ############
        # args.pre_train = "/home/shizun/experiment/20211104_EDSR_decision_x2_e300_ps192_lr0.0001_n32_i1_sTrue_sum_pretrain_de3/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211108_EDSR_decision_x2_e300_ps192_lr0.0001_n32_i2_sTrue_sum_pretrain_de3/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211102_EDSR_decision_x2_e300_ps192_lr0.0001_n32_i4_sTrue_sum_pretrain_de3/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211104_EDSR_decision_x3_e300_ps192_lr0.0001_n32_i1_sTrue_sum_pretrain_de3/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211108_EDSR_decision_x3_e300_ps192_lr0.0001_n32_i2_sTrue_sum_pretrain_de3/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211108_EDSR_decision_x3_e300_ps192_lr0.0001_n32_i4_sTrue_sum_pretrain_de3/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211104_EDSR_decision_x4_e300_ps192_lr0.0001_n32_i1_sTrue_sum_pretrain_de3/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211108_EDSR_decision_x4_e300_ps192_lr0.0001_n32_i2_sTrue_sum_pretrain_de3/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211108_EDSR_decision_x4_e300_ps192_lr0.0001_n32_i4_sTrue_sum_pretrain_de3/model/model_best.pt"
        args.save = "{}_{}_x{}_e{}_ps{}_st{}_n{}_i{}_{}_th{}_dynamic_test".format(today, args.model, args.scale, args.epochs, args.patch_size, args.step, args.n_resgroups, args.exit_interval, args.data_test, args.exit_threshold)

    elif args.template == 'EDSR_test':
        # model
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

        # data
        # args.scale = "2"   #############
        args.dir_data = dir_data
        # args.data_test = 'TEST8K'  #############
        # args.data_test = 'DIV2K'  #############
        if args.data_test == 'DIV2K':
            args.data_range = '801-900'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        args.ext = "sep"
        args.patch_size = 32*int(args.scale)
        args.step = 30*int(args.scale)

        # device
        # args.device = "3"  #############
        args.n_GPUs = 1

        # pipeline
        args.dynamic = True
        args.test_only = True     ############
        args.n_parallel = 80         ############

        # experiment
        args.reset = True
        # args.save_results = True  ############
        # args.save_gt = True       ############
        # args.pre_train = "/home/shizun/experiment/20210929_EDSR_paper_x2_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211023_EDSR_paper_x3_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211023_EDSR_paper_x4_e300_ps192_lr0.0001/model/model_best.pt"
        args.save = "{}_{}_x{}_e{}_ps{}_st{}_n{}_i{}_{}_th{}_static_test".format(today, args.model, args.scale, args.epochs, args.patch_size, args.step, args.n_resgroups, args.exit_interval, args.data_test, args.exit_threshold)

    elif args.template == 'EDSR_ada':
        # model
        args.model = 'EDSR_ada'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

        # data
        args.scale = "4"   #############
        args.dir_data = dir_data
        args.ext = "sep"
        args.patch_size = 192

        # device
        args.device = "2"  #############
        args.n_GPUs = 1

        # pipeline
        args.epochs = 300
        args.lr = 1e-4
        args.batch_size = 16
        args.print_every = 10
        args.ada = True

        # experiemnt
        args.reset = True
        # args.save_results = True  ############
        # args.save_gt = True      ############
        # args.pre_train = "/home/shizun/experiment/20210929_EDSR_paper_x2_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211023_EDSR_paper_x3_e300_ps192_lr0.0001/model/model_best.pt"
        args.pre_train = "/home/shizun/experiment/20211023_EDSR_paper_x4_e300_ps192_lr0.0001/model/model_best.pt"
        args.save = "{}_{}_x{}_e{}_ps{}_lr{}_n{}_pretrain".format(today, args.model, args.scale, args.epochs, args.patch_size, args.lr, args.n_resblocks)

        # resume
        # args.load = "20211103_EDSR_paper_x4_e600_ps192_lr0.0001"
        # args.resume = -1

    elif args.template == 'RCAN':
        args.model = 'RCAN'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64

        args.patch_size = 192
        args.epochs = 600
        args.dir_data = dir_data
        args.data_train = 'DIV2K'
        args.data_test = 'DIV2K'
        args.scale = "2"
        args.device = "2,3"
        args.n_GPUs = 2
        args.batch_size = 16
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        # args.chop = True

        # args.save = "{}_{}_paper_x{}_e{}_ps{}_lr{}".format(today, args.model, args.scale, args.epochs, args.patch_size, args.lr)

        # resume
        args.reset = False
        args.load = "/home/shizun/experiment/20211108_RCAN_paper_x2_e600_ps192_lr0.0001/"
        args.resume = -1

    elif args.template == 'RCAN_decision':
        args.model = 'RCAN_decision'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64

        args.patch_size = 192
        args.epochs = 300
        args.dir_data = dir_data
        args.data_train = 'DIV2K'
        args.data_test = 'DIV2K'
        args.scale = "4"
        args.device = "0,1"
        args.n_GPUs = 2
        args.batch_size = 16
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        # args.chop = True
        args.decision = True
        # args.save_results = True  ############
        # args.save_gt = True      ############
        args.exit_interval = 1    ############
        # args.pre_train = "/home/shizun/experiment/20211105_RCAN_paper_x2_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211105_RCAN_paper_x3_e300_ps192_lr0.0001/model/model_best.pt"
        args.pre_train = "/home/shizun/experiment/20211105_RCAN_paper_x4_e300_ps192_lr0.0001/model/model_best.pt"
        args.save = "{}_{}_x{}_e{}_ps{}_lr{}_n{}_i{}_sum_pretrain_de3".format(today, args.model, args.scale, args.epochs, args.patch_size, args.lr, args.n_resgroups, args.exit_interval)

    elif args.template == 'RCAN_dytest':
        # model
        args.model = 'RCAN_decision'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64

        # data
        # args.scale = "2"   #############
        args.dir_data = dir_data
        # args.data_test = 'TEST8K'  #############
        # args.data_test = 'DIV2K'  #############
        if args.data_test == 'DIV2K':
            args.data_range = '801-900'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        args.ext = "sep"
        args.patch_size = 32*int(args.scale)
        args.step = 30*int(args.scale)

        # device
        # args.device = "1"  #############
        args.n_GPUs = 1

        # pipeline
        args.dynamic = True
        args.test_only = True     ############
        # args.exit_interval = 1    ############
        # args.exit_threshold = 0.8 ############

        # experiment
        args.reset = True
        # args.save_results = True  ############
        # args.save_gt = True     ############
        # args.pre_train = "/home/shizun/experiment/20211108_RCAN_decision_x2_e300_ps192_lr0.0001_n10_i1_sum_pretrain_de3/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211108_RCAN_decision_x3_e300_ps192_lr0.0001_n10_i1_sum_pretrain_de3/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211108_RCAN_decision_x4_e300_ps192_lr0.0001_n10_i1_sum_pretrain_de3/model/model_best.pt"
        args.save = "{}_{}_x{}_e{}_ps{}_st{}_n{}_i{}_{}_th{}_dynamic_test".format(today, args.model, args.scale, args.epochs, args.patch_size, args.step, args.n_resgroups, args.exit_interval, args.data_test, args.exit_threshold)

    elif args.template == 'RCAN_test':
        # model
        args.model = 'RCAN'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64

        # data
        # args.scale = "2"   #############
        args.dir_data = dir_data
        # args.data_test = 'TEST8K'  #############
        # args.data_test = 'DIV2K'  #############
        if args.data_test == 'DIV2K':
            args.data_range = '801-900'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        args.ext = "sep"
        args.patch_size = 32*int(args.scale)
        args.step = 30*int(args.scale)

        # device
        # args.device = "3"  #############
        args.n_GPUs = 1

        # pipeline
        args.dynamic = True
        args.test_only = True     ############
        args.n_parallel = 80         ############

        # experiment
        args.reset = True
        # args.save_results = True  ############
        # args.save_gt = True       ############
        # args.pre_train = "/home/shizun/experiment/20211105_RCAN_paper_x2_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211105_RCAN_paper_x3_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211105_RCAN_paper_x4_e300_ps192_lr0.0001/model/model_best.pt"
        args.save = "{}_{}_x{}_e{}_ps{}_st{}_n{}_i{}_{}_static_test".format(today, args.model, args.scale, args.epochs, args.patch_size, args.step, args.n_resgroups, args.exit_interval, args.data_test)

    elif args.template == 'FSRCNN':
        args.model = 'FSRCNN'

        args.patch_size = 192
        args.epochs = 300
        args.dir_data = dir_data
        args.data_train = 'DIV2K'
        args.data_test = 'DIV2K'
        args.scale = "4"
        args.device = "3"
        args.n_GPUs = 1
        args.batch_size = 16
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        # args.chop = True
        args.save = "{}_{}_paper_x{}_e{}_ps{}_lr{}".format(today, args.model, args.scale, args.epochs, args.patch_size, args.lr)

    elif args.template == 'FSRCNN_decision':
        args.model = 'FSRCNN_decision'

        args.patch_size = 192
        args.epochs = 600
        args.dir_data = dir_data
        args.data_train = 'DIV2K'
        args.data_test = 'DIV2K'
        args.scale = "4"
        args.device = "1"
        args.n_GPUs = 1
        args.batch_size = 16
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        # args.chop = True
        args.decision = True
        # args.save_results = True  ############
        # args.save_gt = True      ############
        args.exit_interval = 1    ############
        # args.pre_train = "/home/shizun/experiment/20211108_FSRCNN_paper_x2_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211108_FSRCNN_paper_x3_e300_ps192_lr0.0001/model/model_best.pt"
        args.pre_train = "/home/shizun/experiment/20211108_FSRCNN_paper_x4_e300_ps192_lr0.0001/model/model_best.pt"
        args.save = "{}_{}_x{}_e{}_ps{}_lr{}_n{}_i{}_sum_pretrain_de3".format(today, args.model, args.scale, args.epochs, args.patch_size, args.lr, args.n_resgroups, args.exit_interval)

    elif args.template == 'RDN':
        args.model = 'RDN'
        args.lr = 1e-4
        args.G0 = 64
        args.RDNconfig = 'B'

        args.patch_size = 192
        args.epochs = 300
        args.dir_data = dir_data
        args.data_train = 'DIV2K'
        args.data_test = 'DIV2K'
        args.scale = "4"
        args.device = "3"
        args.n_GPUs = 1
        args.batch_size = 16
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        # args.chop = True
        args.save = "{}_{}_paper_x{}_e{}_ps{}_lr{}".format(today, args.model, args.scale, args.epochs, args.patch_size, args.lr)

    elif args.template == 'VDSR':
        args.model = 'VDSR'
        args.n_resblocks = 20
        args.n_feats = 64
        args.patch_size = 41
        args.lr = 1e-1

    elif args.template == 'SRVC':
        args.model = 'SRVC'
        args.f = 16
        args.F = 64
        args.n_feats = 32
        args.patch_h = 7
        args.patch_w = 7

        args.lr = 1e-3
        args.loss = '1*L1'
        args.patch_size = 200
        args.scale = "2"
        args.batch_size = 16
        args.epochs = 5000

        args.device = "3"
        args.n_GPUs = 1
        args.dir_data = dir_data
        args.ext = "sep"
        args.print_every = 5

        # args.save = "{}_{}_x{}_ps{}_lr{}_e{}_{}_f{}_F{}_nf{}_ps{}".format(today, args.model, args.scale, args.patch_size, args.lr, args.epochs, args.loss, args.f, args.F, args.n_feats, args.patch_h)
        # args.save = "20210526_SRVC_x2_ps200_lr0.0001_e5000_1*L1_f16_F64_nf32_ps7_test"
        # args.load = "20210524_SRVC_x2_ps200_lr0.0001_e5000_1*L1_f8_F64_nf32_ps7"
        # args.resume = -1
        # args.reset = True ######!!!!!!!!!
        
        # args.test_only = 'True'
        # args.data_test = 'Set5+Set14+B100+Urban100'
        # args.save_results = 'True'
        # args.save_gt = True

        # args.pre_train = "/home/shizun/experiment/20210524_SRVC_x2_ps200_lr0.0001_e5000_1*L1_f16_F64_nf32_ps7/model/model_best.pt"
        # args.data_range = '1-10'

    elif args.template == 'SAN':
        args.model = 'SAN'
        args.n_feats = 64
        args.n_resgroups = 10
        args.n_resblocks = 5

        args.patch_size = 192
        args.epochs = 300
        args.dir_data = dir_data
        args.data_train = 'DIV2K'
        args.data_test = 'DIV2K'
        args.scale = "3"
        args.device = "2,3"
        args.n_GPUs = 2
        args.batch_size = 16
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        args.chop = True
        args.save = "{}_{}_paper_x{}_e{}_ps{}_lr{}".format(today, args.model, args.scale, args.epochs, args.patch_size, args.lr)

    elif args.template == 'HAN':
        args.model = 'HAN'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64

        args.patch_size = 192
        args.epochs = 300
        args.dir_data = dir_data
        args.data_train = 'DIV2K'
        args.data_test = 'DIV2K'
        args.scale = "4"
        args.device = "0,1"
        args.n_GPUs = 2
        args.batch_size = 16
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        # args.chop = True
        args.save = "{}_{}_paper_x{}_e{}_ps{}_lr{}".format(today, args.model, args.scale, args.epochs, args.patch_size, args.lr)


        # for test
        # args.save = "benchmark/{}_{}_x{}_base".format(today, args.model, args.scale)
        # args.save = "benchmark/{}_{}_x{}_ours".format(today, args.model, args.scale)
        # args.pre_train = "/home/shizun/experiment/20210622_HAN_x2_ps192_lr0.0001_e300_1*L1/model/model_best.pt"
        # args.test_only = 'True'
        # args.data_test = 'Set5+Set14+B100+Urban100'

        # args.data_test = 'DIV2K'
        # args.data_range = '1-320/801-810'
        # args.save_results = 'True'
        # args.save_gt = True

    elif args.template == 'MDSR':
        args.model = 'MDSR'
        args.patch_size = 48
        args.epochs = 650

    elif args.template == 'DDBPN':
        args.model = 'DDBPN'
        args.decay = '500'
        args.gamma = 0.1
        args.weight_decay = 1e-4
        args.lr = 1e-4
        args.loss = '1*MSE'
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = '4'
        # args.save = "20210306_ddbpn_x4_lr1e-4_p0.01"
        args.load = "20210306_ddbpn_x4_lr1e-4_p0.01" #
        args.device = "2,3"
        args.n_GPUs = 2
        args.epochs = 1000 #
        args.batch_size = 40
        args.print_every = 10
        args.ext = "sep"
        # args.reset = True
        args.reset = False #
        args.resume = 299 #
        args.data_train = 'DIV2K_PSNR'
        args.data_test = 'DIV2K'
        args.data_partion = 0.01
        args.file_suffix = "_psnr_up_new.pt"

    elif args.template == 'GAN':
        args.epochs = 200
        args.lr = 5e-5
        args.decay = '150'
