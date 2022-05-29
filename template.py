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
        args.scale = "4"   #############
        args.dir_data = dir_data
        args.ext = "sep"
        # args.data_test = 'DIV2K'  #############
        args.data_test = 'B100'  #############
        if args.data_test == 'DIV2K':
            args.data_range = '801-900'
            # args.data_range = '801-810'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        # args.data_range = '1-16/801-900'
        args.patch_size = 192

        # device
        args.device = "0"  #############
        args.n_GPUs = 1

        # pipeline
        args.epochs = 300
        args.lr = 1e-4
        args.batch_size = 16
        args.print_every = 10
        # args.decision = True
        # args.exit_interval = 2    ############

        # experiemnt
        # args.reset = True
        # args.pre_train = "pretrained/EDSR_x2.pt"
        # args.pre_train = "pretrained/EDSR_x3.pt"
        # args.pre_train = "pretrained/EDSR_x4.pt"
        # args.pre_train = "/home/shizun/experiment/20210929_EDSR_paper_x2_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211023_EDSR_paper_x3_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211023_EDSR_paper_x4_e300_ps192_lr0.0001/model/model_best.pt"
        # args.save = "{}_{}_x{}_e{}_ps{}_lr{}_n{}_i{}_s{}_sum_pretrain_de3".format(today, args.model, args.scale, args.epochs, args.patch_size, args.lr, args.n_resblocks, args.exit_interval, args.shared_tail)

        # test
        args.test_only = True
        # args.chop = True
        args.ssim = True
        args.save_results = True  ############
        args.save_gt = True      ############
        # args.pre_train = "/home/shizun/experiment/20210929_EDSR_paper_x2_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211023_EDSR_paper_x3_e300_ps192_lr0.0001/model/model_best.pt"
        args.pre_train = "/home/shizun/experiment/20211023_EDSR_paper_x4_e300_ps192_lr0.0001/model/model_best.pt"
        args.save = "{}_{}_x{}_{}_test_pic".format(today, args.model, args.scale, args.data_test)

        # resume
        # args.load = "20211103_EDSR_paper_x4_e600_ps192_lr0.0001"
        # args.load = "20211103_EDSR_paper_x3_e600_ps192_lr0.0001"
        # args.load = "20211103_EDSR_paper_x2_e600_ps192_lr0.0001"
        # args.resume = -1

    if args.template == 'EDSR_pix':
        # model
        args.model = 'EDSR_pix'   ################
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1
        
        # data
        args.scale = "2"   #############
        args.dir_data = dir_data
        args.ext = "sep"
        # args.data_range = '1-16/801-900'
        args.patch_size = 192

        # device
        args.device = "3"  #############
        args.n_GPUs = 1

        # pipeline
        args.epochs = 300
        args.lr = 1e-4
        args.batch_size = 16
        args.print_every = 10
        # args.decision = True
        # args.exit_interval = 2    ############
        # args.test_only = True
        # args.save_results = True  ############
        # args.save_gt = True      ############

        # experiemnt
        args.reset = True
        # args.pre_train = "pretrained/EDSR_x2.pt"
        # args.pre_train = "/home/shizun/experiment/20210929_EDSR_paper_x2_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211023_EDSR_paper_x3_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211023_EDSR_paper_x4_e300_ps192_lr0.0001/model/model_best.pt"
        args.save = "{}_{}_x{}_e{}_ps{}_lr{}_n{}_i{}".format(today, args.model, args.scale, args.epochs, args.patch_size, args.lr, args.n_resblocks, args.exit_interval)
        # args.save = "{}_{}_x{}_e{}_ps{}_lr{}_n{}_i{}_s{}_official_test".format(today, args.model, args.scale, args.epochs, args.patch_size, args.lr, args.n_resblocks, args.exit_interval, args.shared_tail)

        # resume
        # args.load = "20211103_EDSR_paper_x4_e600_ps192_lr0.0001"
        # args.load = "20211103_EDSR_paper_x3_e600_ps192_lr0.0001"
        # args.load = "20211103_EDSR_paper_x2_e600_ps192_lr0.0001"
        # args.resume = -1

    elif args.template == 'EDSR_pix_decision':
        # model
        args.model = 'EDSR_pix_decision'   ################
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1
        
        # data
        args.scale = "4"   #############
        args.dir_data = dir_data
        args.ext = "sep"
        # args.data_range = '1-16/801-802'
        args.patch_size = 192

        # device
        args.device = "2"  #############
        args.n_GPUs = 1

        # pipeline
        args.epochs = 300
        args.lr = 1e-4
        args.batch_size = 16
        args.print_every = 10
        args.decision = True
        args.exit_interval = 4    ############
        # args.save_results = True  ############
        # args.save_gt = True      ############

        # experiemnt
        args.reset = True
        # args.pre_train = "pretrained/EDSR_x2.pt"
        # args.pre_train = "pretrained/EDSR_x3.pt"
        # args.pre_train = "pretrained/EDSR_x4.pt"
        # args.pre_train = "/home/shizun/experiment/20220122_EDSR_pix_x2_e300_ps192_lr0.0001_n32_i1_sFalse/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20220122_EDSR_pix_x3_e300_ps192_lr0.0001_n32_i1_sFalse/model/model_best.pt"
        args.pre_train = "/home/shizun/experiment/20220122_EDSR_pix_x4_e300_ps192_lr0.0001_n32_i1_sFalse/model/model_best.pt"
        args.save = "{}_{}_x{}_e{}_ps{}_lr{}_n{}_i{}".format(today, args.model, args.scale, args.epochs, args.patch_size, args.lr, args.n_resblocks, args.exit_interval)

    elif args.template == 'EDSR_decision':
        # model
        args.model = 'EDSR_decision'   ################
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1
        
        # data
        args.scale = "4"   #############
        args.dir_data = dir_data
        args.ext = "sep"
        # args.data_range = '1-16/801-802'
        args.patch_size = 192

        # device
        args.device = "1"  #############
        args.n_GPUs = 1

        # pipeline
        args.epochs = 500
        args.lr = 1e-4
        args.decay = "200-400"
        args.batch_size = 16
        args.print_every = 10
        args.decision = True
        args.strategy = "de3"
        args.eedm = 3
        args.exit_interval = 4    ############
        # args.save_results = True  ############
        # args.save_gt = True      ############

        # experiemnt
        # args.reset = True
        # # args.pre_train = "pretrained/EDSR_x2.pt"
        # # args.pre_train = "pretrained/EDSR_x3.pt"
        # # args.pre_train = "pretrained/EDSR_x4.pt"
        # # args.pre_train = "/home/shizun/experiment/20210929_EDSR_paper_x2_e300_ps192_lr0.0001/model/model_best.pt"
        # # args.pre_train = "/home/shizun/experiment/20211023_EDSR_paper_x3_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211023_EDSR_paper_x4_e300_ps192_lr0.0001/model/model_best.pt"
        # args.save = "{}_{}_{}_x{}_e{}_ps{}_lr{}_n{}_i{}_eedm{}".format(today, args.model, args.strategy, args.scale, args.epochs, args.patch_size, args.lr, args.n_resblocks, args.exit_interval, args.eedm)
        
        # resume
        args.reset = False
        # args.load = "/data/shizun/experiment/20220529_EDSR_decision_de3_x4_e500_ps192_lr0.0001_n32_i4_eedm2/"
        args.load = "/data/shizun/experiment/20220529_EDSR_decision_de3_x4_e500_ps192_lr0.0001_n32_i4_eedm3/"
        args.resume = -1

    elif args.template == 'EDSR_multi':
        # model
        args.model = 'EDSR_decision'   ################
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1
        
        # data
        args.scale = "3"   #############
        args.dir_data = dir_data
        args.ext = "sep"
        # args.data_range = '1-16/801-802'
        args.patch_size = 192

        # device
        args.device = "2,3"  #############
        args.n_GPUs = 2

        # pipeline
        args.epochs = 300
        args.lr = 1e-4
        args.batch_size = 16
        args.print_every = 10
        args.decision = True
        args.strategy = "multi"
        args.exit_interval = 1    ############
        # args.save_results = True  ############
        # args.save_gt = True      ############

        # experiemnt
        args.reset = True
        # args.pre_train = "pretrained/EDSR_x2.pt"
        # args.pre_train = "pretrained/EDSR_x3.pt"
        # args.pre_train = "pretrained/EDSR_x4.pt"
        # args.pre_train = "/home/shizun/experiment/20210929_EDSR_paper_x2_e300_ps192_lr0.0001/model/model_best.pt"
        args.pre_train = "/home/shizun/experiment/20211023_EDSR_paper_x3_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211023_EDSR_paper_x4_e300_ps192_lr0.0001/model/model_best.pt"
        args.save = "{}_{}_{}_x{}_e{}_ps{}_lr{}_n{}_i{}".format(today, args.model, args.strategy, args.scale, args.epochs, args.patch_size, args.lr, args.n_resblocks, args.exit_interval)

    elif args.template == 'EDSR_eedm':
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
        args.device = "2"  #############
        args.n_GPUs = 1

        # pipeline
        args.epochs = 300
        args.loss = '1*MSE'
        args.lr = 1e-4
        args.batch_size = 16
        args.print_every = 10
        args.decision = True
        args.strategy = "eedm"
        args.exit_interval = 1    ############
        # args.save_results = True  ############
        # args.save_gt = True      ############

        # experiemnt
        args.reset = True
        # args.pre_train = "pretrained/EDSR_x2.pt"
        # args.pre_train = "pretrained/EDSR_x3.pt"
        # args.pre_train = "pretrained/EDSR_x4.pt"
        args.pre_train = "/home/shizun/experiment/20220224_EDSR_decision_multi_x2_e300_ps192_lr0.0001_n32_i1/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20220225_EDSR_decision_multi_x3_e300_ps192_lr0.0001_n32_i1/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20220222_EDSR_decision_multi_x4_e300_ps192_lr0.0001_n32_i1/model/model_best.pt"
        args.save = "{}_{}_{}_x{}_e{}_ps{}_lr{}_n{}_i{}".format(today, args.model, args.strategy, args.scale, args.epochs, args.patch_size, args.lr, args.n_resblocks, args.exit_interval)

    elif args.template == 'EDSR_check':
        # model
        args.model = 'EDSR_check'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

        # data
        args.scale = "4"   #############
        args.dir_data = dir_data
        # args.data_test = 'TEST8K'  #############
        args.data_test = 'DIV2K'  #############
        if args.data_test == 'DIV2K':
            args.data_range = '801-900'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        args.ext = "sep"
        args.patch_size = 48*int(args.scale)
        args.step = 46*int(args.scale)

        # device
        args.device = "3"  #############
        args.n_GPUs = 1

        # pipeline
        args.check = True
        args.test_only = True     ############
        args.exit_interval = 4    ############
        args.exit_threshold = 0.9 ############
        args.n_parallel = 500         ############
        # args.save_results = True  ############
        # args.save_gt = True     ############
        # args.add_mask = True

        # experiment
        args.reset = True
        args.eedm = 1
        # args.pre_train = "/data/shizun/experiment/20220224_EDSR_decision_multi_x2_e300_ps192_lr0.0001_n32_i1/model/model_best.pt"
        # args.pre_train = "/data/shizun/experiment/20220225_EDSR_decision_multi_x3_e300_ps192_lr0.0001_n32_i1/model/model_best.pt"
        # args.pre_train = "/data/shizun/experiment/20220222_EDSR_decision_multi_x4_e300_ps192_lr0.0001_n32_i1/model/model_best.pt"
        args.pre_train = "/home/shizun/experiment/20220301_EDSR_decision_de3_x4_e300_ps192_lr0.0001_n32_i4/model/model_best.pt"
        args.save = "{}_{}_x{}_e{}_ps{}_st{}_n{}_i{}_{}_eedm{}_th{}_check".format(today, args.model, args.scale, args.epochs, args.patch_size, args.step, args.n_resgroups, args.exit_interval, args.data_test, args.eedm, args.exit_threshold)

    elif args.template == 'EDSR_dytest':
        # model
        args.model = 'EDSR_decision'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

        # data
        args.scale = "4"   #############
        args.dir_data = dir_data
        # args.data_test = 'TEST8K'  #############
        args.data_test = 'DIV2K'  #############
        # args.data_test = 'Urban100'  #############
        # args.data_test = 'B100'  #############
        if args.data_test == 'DIV2K':
            args.data_range = '801-900'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        args.ext = "sep"
        args.patch_size = 48*int(args.scale)
        args.step = 46*int(args.scale)

        # device
        args.device = "2"  #############
        args.n_GPUs = 1

        # pipeline
        args.dynamic = True
        args.test_only = True     ############
        args.ssim = True
        args.exit_interval = 4    ############
        args.exit_threshold = 0 ############
        args.n_parallel = 500         ############
        # args.save_results = True  ############
        # args.save_gt = True     ############
        # args.add_mask = True

        # experiment
        args.reset = True
        args.strategy = "de3"
        args.eedm = 1
        # args.pre_train = "/home/shizun/experiment/20220301_EDSR_decision_de3_x2_e300_ps192_lr0.0001_n32_i1/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20220305_EDSR_decision_de3_x2_e300_ps192_lr0.0001_n32_i2/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20220301_EDSR_decision_de3_x2_e300_ps192_lr0.0001_n32_i4/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20220301_EDSR_decision_de3_x3_e300_ps192_lr0.0001_n32_i4/model/model_best.pt"
        args.pre_train = "/home/shizun/experiment/20220301_EDSR_decision_de3_x4_e300_ps192_lr0.0001_n32_i4/model/model_best.pt"
        # args.pre_train = "/data/shizun/experiment/20220526_EDSR_decision_de3_x4_e300_ps192_lr0.0001_n32_i4/model/model_best.pt"

        # args.pre_train = "/data/shizun/experiment/20220528_EDSR_decision_de3_x4_e300_ps192_lr0.0001_n32_i4_eedm2/model/model_best.pt"
        # args.pre_train = "/data/shizun/experiment/20220528_EDSR_decision_de3_x4_e300_ps192_lr0.0001_n32_i4_eedm3/model/model_best.pt"
        
        args.save = "{}_{}_x{}_{}_e{}_ps{}_st{}_n{}_i{}_{}_rand_th{}_dynamic_test".format(today, args.model, args.scale, args.strategy, args.epochs, args.patch_size, args.step, args.n_resgroups, args.exit_interval, args.data_test, args.eedm, args.exit_threshold)

    elif args.template == 'EDSR_test':
        # model
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

        # data
        args.scale = "2"   #############
        args.dir_data = dir_data
        # args.data_test = 'TEST8K'  #############
        args.data_test = 'DIV2K'  #############
        if args.data_test == 'DIV2K':
            args.data_range = '801-810'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        args.ext = "sep"
        args.patch_size = 96*int(args.scale)
        args.step = 94*int(args.scale)

        # device
        args.device = "0"  #############
        args.n_GPUs = 1

        # pipeline
        args.dynamic = True
        args.test_only = True     ############
        args.n_parallel = 400         ############
        args.ssim = True
        # args.save_results = True  ############
        # args.save_gt = True       ############

        # experiment
        args.reset = True
        args.pre_train = "pretrained/EDSR_x2.pt"
        # args.pre_train = "pretrained/EDSR_x3.pt"
        # args.pre_train = "pretrained/EDSR_x4.pt"
        # args.pre_train = "/home/shizun/experiment/20210929_EDSR_paper_x2_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211023_EDSR_paper_x3_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211023_EDSR_paper_x4_e300_ps192_lr0.0001/model/model_best.pt"
        args.save = "{}_{}_x{}_e{}_ps{}_st{}_n{}_i{}_{}_static_test_801-810".format(today, args.model, args.scale, args.epochs, args.patch_size, args.step, args.n_resgroups, args.exit_interval, args.data_test)

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
        args.device = "1"  #############
        args.n_GPUs = 1

        # pipeline
        args.epochs = 300
        args.lr = 5e-5
        args.batch_size = 16
        args.print_every = 10
        args.ada = True
        args.lambda_pred = 0.01

        # experiemnt
        args.reset = True
        # args.save_results = True  ############
        # args.save_gt = True      ############
        # args.pre_train = "/home/shizun/experiment/20210929_EDSR_paper_x2_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20211023_EDSR_paper_x3_e300_ps192_lr0.0001/model/model_best.pt"
        args.pre_train = "/home/shizun/experiment/20211023_EDSR_paper_x4_e300_ps192_lr0.0001/model/model_best.pt"
        args.save = "{}_{}_x{}_e{}_ps{}_lr{}_n{}_l{}_pretrain_new".format(today, args.model, args.scale, args.epochs, args.patch_size, args.lr, args.n_resblocks, args.lambda_pred)

        # resume
        # args.load = "20211103_EDSR_paper_x4_e600_ps192_lr0.0001"
        # args.resume = -1

    elif args.template == 'EDSR_adatest':
        # model
        args.model = 'EDSR_ada'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

        # data
        args.scale = "4"   #############
        args.dir_data = dir_data
        args.ext = "sep"
        # args.data_test = 'TEST8K'  #############
        args.data_test = 'DIV2K'  #############
        if args.data_test == 'DIV2K':
            args.data_range = '801-900'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        args.ext = "sep"

        # device
        args.device = "3"  #############
        args.n_GPUs = 1

        # pipeline
        args.ada = True
        args.test_only = True
        args.ada_depth = 16

        # experiemnt
        args.reset = True
        # args.save_results = True  ############
        # args.save_gt = True      ############
        args.pre_train = "/home/shizun/experiment/20211112_EDSR_ada_x4_e300_ps192_lr5e-05_n32_l0.01_pretrain_new/model/model_best.pt"
        args.save = "{}_{}_x{}_n{}_{}_d{}_ada_test".format(today, args.model, args.scale, args.n_resblocks, args.data_test, args.ada_depth)

    if args.template.find('EDSR_match') >= 0:
        args.model = 'EDSR_dynamic'   ################
        # args.model = 'EDSR'
        args.lr = 1e-4
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1
        args.patch_size = 192
        args.epochs = 300
        args.dir_data = dir_data
        args.data_train = 'DIV2K_DYNAMIC'
        args.data_test = 'DIV2K_DYNAMIC'
        args.scale = "2"   #############
        args.device = "1"  #############
        args.n_GPUs = 1
        # args.data_range = '1-16/801-802'
        args.batch_size = 16
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        args.match = True
        args.shared_tail = True
        args.exit_interval = 1    ############
        args.test_only = True
        args.bins = 96           ############
        args.n_test_samples = 1  ############
        args.bin_index = 80
        args.save_results = True
        args.save_gt = True
        args.statistics_file = "/data/shizun/DIV2K/bin/DIV2K_train_LR_bicubic/X2/statistics_Canny_p192_s24.pt"
        # args.statistics_file = "/data/shizun/DIV2K/bin/DIV2K_train_LR_bicubic/X3/statistics_Canny_p192_s24.pt"
        # args.statistics_file = "/data/shizun/DIV2K/bin/DIV2K_train_LR_bicubic/X4/statistics_Canny_p192_s24.pt"
        args.pre_train = "/home/shizun/experiment/20211021_EDSR_dynamic_x2_e300_ps192_lr0.0001_n32_i1_sTrue_sum_pretrain/model/model_best.pt"
        args.save = "{}_{}_match_x{}_e{}_ps{}_lr{}_n{}_i{}_s{}_b{}_t{}".format(today, args.model, args.scale, args.epochs, args.patch_size, args.lr, args.n_resblocks, args.exit_interval, args.shared_tail, args.bins, args.n_test_samples)


    elif args.template == 'RCAN':
        # model
        args.model = 'RCAN'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64

        # data
        args.scale = "4"
        args.ext = "sep"
        args.dir_data = dir_data
        args.data_train = 'DIV2K'
        # args.data_test = 'DIV2K'
        # args.data_test = 'Urban100'
        args.data_test = 'B100'
        args.patch_size = 192

        # device
        args.device = "1"
        args.n_GPUs = 1

        # pipeline
        args.epochs = 200
        args.batch_size = 16
        args.print_every = 10

        # experiment
        # args.reset = True
        # args.save = "{}_{}_my_x{}_e{}_ps{}_lr{}".format(today, args.model, args.scale, args.epochs, args.patch_size, args.lr)

        # resume
        # args.reset = False
        # args.load = "/home/shizun/experiment/20211108_RCAN_paper_x2_e600_ps192_lr0.0001/"
        # args.resume = -1

        # test
        args.test_only = True
        # args.chop = True
        args.ssim = True
        args.save_results = True  ############
        args.save_gt = True      ############
        if args.data_test == 'DIV2K':
            args.data_range = '801-900'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        # args.pre_train = "/data/shizun/experiment/20220223_RCAN_my_x2_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/data/shizun/experiment/20220224_RCAN_my_x3_e300_ps192_lr0.0001/model/model_best.pt"
        args.pre_train = "/data/shizun/experiment/20220222_RCAN_my_x4_e300_ps192_lr0.0001/model/model_best.pt"
        args.save = "{}_{}_x{}_{}_test_pic".format(today, args.model, args.scale, args.data_test)


    elif args.template == 'RCAN_multi':
        # model
        args.model = 'RCAN_decision'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64
        
        # data
        args.scale = "3"   #############
        args.dir_data = dir_data
        args.ext = "sep"
        args.patch_size = 192

        # device
        args.device = "1"  #############
        args.n_GPUs = 1

        # pipeline
        args.epochs = 300
        args.lr = 1e-4
        args.batch_size = 16
        args.print_every = 10
        args.decision = True
        args.strategy = "multi"
        args.exit_interval = 1    ############
        # args.save_results = True  ############
        # args.save_gt = True      ############

        # experiemnt
        args.reset = True
        # args.pre_train = "/home/shizun/experiment/20220223_RCAN_my_x2_e300_ps192_lr0.0001/model/model_best.pt"
        args.pre_train = "/home/shizun/experiment/20220224_RCAN_my_x3_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20220222_RCAN_my_x4_e300_ps192_lr0.0001/model/model_best.pt"
        args.save = "{}_{}_{}_x{}_e{}_ps{}_lr{}_n{}_i{}".format(today, args.model, args.strategy, args.scale, args.epochs, args.patch_size, args.lr, args.n_resblocks, args.exit_interval)

    elif args.template == 'RCAN_check':
        # model
        args.model = 'RCAN_check'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64

        # data
        args.scale = "2"   #############
        args.dir_data = dir_data
        # args.data_test = 'TEST8K'  #############
        args.data_test = 'DIV2K'  #############
        if args.data_test == 'DIV2K':
            args.data_range = '801-900'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        args.ext = "sep"
        args.patch_size = 48*int(args.scale)
        args.step = 46*int(args.scale)

        # device
        args.device = "3"  #############
        args.n_GPUs = 1

        # pipeline
        args.check = True
        args.test_only = True     ############
        args.exit_interval = 1    ############
        args.exit_threshold = 1 ############
        args.n_parallel = 500         ############
        # args.save_results = True  ############
        # args.save_gt = True     ############
        # args.add_mask = True

        # experiment
        args.reset = True
        args.pre_train = "/data/shizun/experiment/20220225_RCAN_decision_multi_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt"
        # args.pre_train = "/data/shizun/experiment/20220225_RCAN_decision_multi_x3_e300_ps192_lr0.0001_n20_i1/model/model_best.pt"
        # args.pre_train = "/data/shizun/experiment/20220224_RCAN_decision_multi_x4_e300_ps192_lr0.0001_n20_i1/model/model_best.pt"
        args.save = "{}_{}_x{}_e{}_ps{}_st{}_n{}_i{}_{}_th{}_check".format(today, args.model, args.scale, args.epochs, args.patch_size, args.step, args.n_resgroups, args.exit_interval, args.data_test, args.exit_threshold)

    elif args.template == 'RCAN_decision':
        # model
        args.model = 'RCAN_decision'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64

        # data
        args.scale = "4"   #############
        args.dir_data = dir_data
        args.ext = "sep"
        # args.data_range = '1-16/801-802'
        args.patch_size = 192

        # device
        args.device = "0"  #############
        args.n_GPUs = 1

        # pipeline
        args.epochs = 300
        args.lr = 1e-4
        args.batch_size = 16
        args.print_every = 10
        args.decision = True
        args.strategy = "de3"
        args.exit_interval = 1    ############
        # args.save_results = True  ############
        # args.save_gt = True      ############

        # experiemnt
        args.reset = True
        # args.pre_train = "/home/shizun/experiment/20220223_RCAN_my_x2_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20220224_RCAN_my_x3_e300_ps192_lr0.0001/model/model_best.pt"
        args.pre_train = "/home/shizun/experiment/20220222_RCAN_my_x4_e300_ps192_lr0.0001/model/model_best.pt"
        args.save = "{}_{}_{}_x{}_e{}_ps{}_lr{}_n{}_i{}".format(today, args.model, args.strategy, args.scale, args.epochs, args.patch_size, args.lr, args.n_resblocks, args.exit_interval)

    elif args.template == 'RCAN_dytest':
        # model
        args.model = 'RCAN_decision'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64

        # data
        args.scale = "4"   #############
        args.dir_data = dir_data
        # args.data_test = 'TEST8K'  #############
        # args.data_test = 'DIV2K'  #############
        args.data_test = 'Urban100'  #############
        # args.data_test = 'B100'  #############
        if args.data_test == 'DIV2K':
            args.data_range = '801-900'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        args.ext = "sep"
        args.patch_size = 48*int(args.scale)
        args.step = 46*int(args.scale)

        # device
        args.device = "3"  #############
        args.n_GPUs = 1

        # pipeline
        args.dynamic = True
        args.test_only = True     ############
        args.ssim = True
        args.exit_interval = 1    ############
        args.exit_threshold = 0.87 ############
        args.n_parallel = 500

        # experiment
        args.reset = True
        # args.save_results = True  ############
        # args.save_gt = True     ############
        # args.pre_train = "/home/shizun/experiment/20220301_RCAN_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20220301_RCAN_decision_de3_x3_e300_ps192_lr0.0001_n20_i1/model/model_best.pt"
        args.pre_train = "/home/shizun/experiment/20220301_RCAN_decision_de3_x4_e300_ps192_lr0.0001_n20_i1/model/model_best.pt"
        args.save = "{}_{}_x{}_e{}_ps{}_st{}_n{}_i{}_{}_th{}_dynamic_test".format(today, args.model, args.scale, args.epochs, args.patch_size, args.step, args.n_resgroups, args.exit_interval, args.data_test, args.exit_threshold)

    elif args.template == 'RCAN_test':
        # model
        args.model = 'RCAN'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64

        # data
        args.scale = "2"   #############
        args.dir_data = dir_data
        # args.data_test = 'TEST8K'  #############
        args.data_test = 'DIV2K'  #############
        if args.data_test == 'DIV2K':
            args.data_range = '801-900'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        args.ext = "sep"
        args.patch_size = 32*int(args.scale)
        args.step = 30*int(args.scale)

        # device
        args.device = "3"  #############
        args.n_GPUs = 1

        # pipeline
        args.dynamic = True
        args.test_only = True     ############
        args.n_parallel = 200         ############

        # experiment
        args.reset = True
        # args.save_results = True  ############
        # args.save_gt = True       ############
        args.pre_train = "/home/shizun/experiment/20211108_RCAN_paper_x2_e600_ps192_lr0.0001/model/model_best.pt"
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
        # model
        args.model = 'VDSR'
        args.n_resblocks = 20
        args.n_feats = 64

        # data
        args.scale = "4"
        args.ext = "sep"
        args.dir_data = dir_data
        args.data_train = 'DIV2K'
        args.data_test = 'DIV2K'
        args.patch_size = 192

        # device
        args.device = "1"
        args.n_GPUs = 1

        # pipeline
        args.epochs = 300
        args.lr = 1e-4
        args.batch_size = 16
        args.print_every = 10

        # experiment
        # args.reset = True
        # args.save = "{}_{}_my_x{}_e{}_ps{}_lr{}".format(today, args.model, args.scale, args.epochs, args.patch_size, args.lr)

        # resume
        # args.reset = False
        # args.load = "/home/shizun/experiment/20211108_RCAN_paper_x2_e600_ps192_lr0.0001/"
        # args.resume = -1

        # test
        args.test_only = True
        # args.chop = True
        args.ssim = True
        args.save_results = True  ############
        args.save_gt = True      ############
        if args.data_test == 'DIV2K':
            args.data_range = '801-900'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        # args.pre_train = "/data/shizun/experiment/20220223_VDSR_my_x2_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/data/shizun/experiment/20220223_VDSR_my_x3_e300_ps192_lr0.0001/model/model_best.pt"
        args.pre_train = "/data/shizun/experiment/20220222_VDSR_my_x4_e300_ps192_lr0.0001/model/model_best.pt"
        args.save = "{}_{}_x{}_{}_test_pic".format(today, args.model, args.scale, args.data_test)


    elif args.template == 'VDSR_multi':
        # model
        args.model = 'VDSR_decision'
        args.n_resblocks = 20
        args.n_feats = 64

        # data
        args.scale = "3"
        args.ext = "sep"
        args.dir_data = dir_data
        args.data_train = 'DIV2K'
        args.data_test = 'DIV2K'
        # if args.data_test == 'DIV2K':
        #     args.data_range = '801-900'
        # elif args.data_test == 'TEST8K':
        #     args.data_range = '1-100'
        args.patch_size = 192

        # device
        args.device = "1"
        args.n_GPUs = 1

        # pipeline
        args.epochs = 300
        args.lr = 1e-4
        args.batch_size = 16
        args.print_every = 10
        # args.chop = True
        args.decision = True
        args.strategy = "multi"
        args.exit_interval = 1    ############
        args.reset = True
        # args.test_only = True
        # args.save_results = True  ############
        # args.save_gt = True      ############


        # experiment
        # args.pre_train = "/home/shizun/experiment/20220223_VDSR_my_x2_e300_ps192_lr0.0001/model/model_best.pt"
        args.pre_train = "/home/shizun/experiment/20220223_VDSR_my_x3_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20220222_VDSR_my_x4_e300_ps192_lr0.0001/model/model_best.pt"
        args.save = "{}_{}_{}_x{}_e{}_ps{}_lr{}_n{}_i{}".format(today, args.model, args.strategy, args.scale, args.epochs, args.patch_size, args.lr, args.n_resblocks, args.exit_interval)

        # resume
        # args.reset = False
        # args.load = "/home/shizun/experiment/20211108_RCAN_paper_x2_e600_ps192_lr0.0001/"
        # args.resume = -1

        # test
        # args.pre_train = "/data/shizun/experiment/20220223_VDSR_my_x2_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/data/shizun/experiment/20220223_VDSR_my_x3_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/data/shizun/experiment/20220222_VDSR_my_x4_e300_ps192_lr0.0001/model/model_best.pt"
        # args.save = "{}_{}_my_x{}_e{}_test".format(today, args.model, args.scale, args.epochs)

    elif args.template == 'VDSR_check':
        # model
        args.model = 'VDSR_check'
        args.n_resblocks = 20
        args.n_feats = 64

        # data
        args.scale = "3"
        args.ext = "sep"
        args.dir_data = dir_data
        args.data_train = 'DIV2K'
        args.data_test = 'DIV2K'
        if args.data_test == 'DIV2K':
            args.data_range = '801-900'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        args.patch_size = 192

        # device
        args.device = "1"
        args.n_GPUs = 1

        # pipeline
        args.check = True
        args.test_only = True     ############
        args.exit_interval = 3    ############
        args.exit_threshold = 1 ############
        args.n_parallel = 500         ############
        # args.save_results = True  ############
        # args.save_gt = True     ############
        # args.add_mask = True


        # experiment
        # args.pre_train = "/home/shizun/experiment/20220223_VDSR_decision_multi_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt"
        args.pre_train = "/home/shizun/experiment/20220224_VDSR_decision_multi_x3_e300_ps192_lr0.0001_n20_i1/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20220223_VDSR_decision_multi_x4_e300_ps192_lr0.0001_n20_i1/model/model_best.pt"
        args.save = "{}_{}_x{}_e{}_ps{}_st{}_n{}_i{}_{}_th{}_check".format(today, args.model, args.scale, args.epochs, args.patch_size, args.step, args.n_resgroups, args.exit_interval, args.data_test, args.exit_threshold)

    elif args.template == 'VDSR_decision':
        # model
        args.model = 'VDSR_decision'   ################
        args.n_resblocks = 20
        args.n_feats = 64
        
        # data
        args.scale = "2"   #############
        args.dir_data = dir_data
        args.ext = "sep"
        # args.data_range = '1-16/801-802'
        args.patch_size = 192

        # device
        args.device = "0"  #############
        args.n_GPUs = 1

        # pipeline
        args.epochs = 300
        args.lr = 1e-4
        args.batch_size = 16
        args.print_every = 10
        args.decision = True
        args.strategy = "de3"
        args.exit_interval = 2    ############
        # args.save_results = True  ############
        # args.save_gt = True      ############

        # experiemnt
        args.reset = True
        args.pre_train = "/data/shizun/experiment/20220223_VDSR_my_x2_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/data/shizun/experiment/20220223_VDSR_my_x3_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/data/shizun/experiment/20220222_VDSR_my_x4_e300_ps192_lr0.0001/model/model_best.pt"        
        args.save = "{}_{}_{}_x{}_e{}_ps{}_lr{}_n{}_i{}".format(today, args.model, args.strategy, args.scale, args.epochs, args.patch_size, args.lr, args.n_resblocks, args.exit_interval)

    elif args.template == 'VDSR_dytest':
        # model
        args.model = 'VDSR_decision'
        args.n_resblocks = 20
        args.n_feats = 64

        # data
        args.scale = "4"   #############
        args.dir_data = dir_data
        # args.data_test = 'TEST8K'  #############
        args.data_test = 'DIV2K'  #############
        if args.data_test == 'DIV2K':
            args.data_range = '801-900'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        args.ext = "sep"
        args.patch_size = 48*int(args.scale)
        args.step = 46*int(args.scale)

        # device
        args.device = "1"  #############
        args.n_GPUs = 1

        # pipeline
        args.dynamic = True
        args.test_only = True     ############
        args.ssim = True
        args.exit_interval = 2    ############
        args.exit_threshold = 1 ############
        args.n_parallel = 500         ############
        args.save_results = True  ############
        args.save_gt = True     ############
        # args.add_mask = True

        # experiment
        args.reset = True
        # args.pre_train = "/data/shizun/experiment/20220228_VDSR_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt"
        # args.pre_train = "/data/shizun/experiment/20220228_VDSR_decision_de3_x3_e300_ps192_lr0.0001_n20_i1/model/model_best.pt"
        # args.pre_train = "/data/shizun/experiment/20220228_VDSR_decision_de3_x4_e300_ps192_lr0.0001_n20_i1/model/model_best.pt"
        # args.pre_train = "/data/shizun/experiment/20220301_VDSR_decision_de3_x2_e300_ps192_lr0.0001_n20_i2/model/model_best.pt"
        # args.pre_train = "/data/shizun/experiment/20220228_VDSR_decision_de3_x3_e300_ps192_lr0.0001_n20_i2/model/model_best.pt"
        args.pre_train = "/data/shizun/experiment/20220228_VDSR_decision_de3_x4_e300_ps192_lr0.0001_n20_i2/model/model_best.pt"
        args.save = "{}_{}_x{}_e{}_ps{}_st{}_n{}_i{}_{}_th{}_dynamic_test_pic".format(today, args.model, args.scale, args.epochs, args.patch_size, args.step, args.n_resgroups, args.exit_interval, args.data_test, args.exit_threshold)

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

    elif args.template == 'ECBSR':
        # model
        args.model = 'ECBSR'
        args.m_ecbsr = 16
        args.c_ecbsr = 64
        args.dm_ecbsr = 2
        args.act = 'prelu'

        # data
        args.scale = "4"
        args.ext = "sep"
        args.dir_data = dir_data
        args.data_train = 'DIV2K'
        args.data_test = 'DIV2K'
        args.patch_size = 192

        # device
        args.device = "1"
        args.n_GPUs = 1

        # pipeline
        args.epochs = 300
        args.batch_size = 16
        args.print_every = 10

        # experiment
        # args.reset = True
        # args.save = "{}_{}_my_x{}_e{}_ps{}_lr{}".format(today, args.model, args.scale, args.epochs, args.patch_size, args.lr)

        # resume
        # args.reset = False
        # args.load = "/home/shizun/experiment/20211108_RCAN_paper_x2_e600_ps192_lr0.0001/"
        # args.resume = -1

        # test
        args.test_only = True
        # args.chop = True
        args.ssim = True
        args.save_results = True  ############
        args.save_gt = True      ############
        if args.data_test == 'DIV2K':
            args.data_range = '801-900'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        # args.pre_train = "/data/shizun/experiment/20220225_ECBSR_my_x2_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/data/shizun/experiment/20220225_ECBSR_my_x3_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/data/shizun/experiment/20220225_ECBSR_my_x4_e300_ps192_lr0.0001/model/model_best.pt"

        # args.pre_train = "/data/shizun/experiment/20220228_ECBSR_my_x2_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/data/shizun/experiment/20220228_ECBSR_my_x3_e300_ps192_lr0.0001/model/model_best.pt"
        args.pre_train = "/data/shizun/experiment/20220228_ECBSR_my_x4_e300_ps192_lr0.0001/model/model_best.pt"
        args.save = "{}_{}_x{}_{}_test_pic".format(today, args.model, args.scale, args.data_test)

    elif args.template == 'ECBSR_multi':
        # model
        args.model = 'ECBSR_decision'
        args.m_ecbsr = 16
        args.c_ecbsr = 64
        args.dm_ecbsr = 2
        args.act = 'prelu'
        
        # data
        args.scale = "2"   #############
        args.dir_data = dir_data
        args.ext = "sep"
        # args.data_range = '1-16/801-802'
        args.patch_size = 192

        # device
        args.device = "2"  #############
        args.n_GPUs = 1

        # pipeline
        args.epochs = 300
        args.lr = 1e-4
        args.batch_size = 16
        args.print_every = 10
        args.decision = True
        args.strategy = "multi"
        args.exit_interval = 1    ############
        # args.save_results = True  ############
        # args.save_gt = True      ############

        # experiemnt
        args.reset = True
        # args.pre_train = "/home/shizun/experiment/20220228_ECBSR_my_x2_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20220228_ECBSR_my_x3_e300_ps192_lr0.0001/model/model_best.pt"
        args.pre_train = "/home/shizun/experiment/20220228_ECBSR_my_x4_e300_ps192_lr0.0001/model/model_best.pt"
        args.save = "{}_{}_{}_x{}_e{}_ps{}_lr{}_n{}_i{}".format(today, args.model, args.strategy, args.scale, args.epochs, args.patch_size, args.lr, args.n_resblocks, args.exit_interval)

    elif args.template == 'ECBSR_check':
        # model
        args.model = 'ECBSR_check'
        args.m_ecbsr = 16
        args.c_ecbsr = 64
        args.dm_ecbsr = 2
        args.act = 'prelu'

        # data
        args.scale = "4"   #############
        args.dir_data = dir_data
        # args.data_test = 'TEST8K'  #############
        args.data_test = 'DIV2K'  #############
        if args.data_test == 'DIV2K':
            args.data_range = '801-900'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        args.ext = "sep"
        args.patch_size = 48*int(args.scale)
        args.step = 46*int(args.scale)

        # device
        args.device = "3"  #############
        args.n_GPUs = 1

        # pipeline
        args.check = True
        args.test_only = True     ############
        args.exit_interval = 4    ############
        args.exit_threshold = 1 ############
        args.n_parallel = 500         ############
        # args.save_results = True  ############
        # args.save_gt = True     ############
        # args.add_mask = True

        # experiment
        args.reset = True
        args.pre_train = "/data/shizun/experiment/20220228_ECBSR_decision_multi_x4_e300_ps192_lr0.0001_n16_i1/model/model_best.pt"
        args.save = "{}_{}_x{}_e{}_ps{}_st{}_n{}_i{}_{}_th{}_check".format(today, args.model, args.scale, args.epochs, args.patch_size, args.step, args.n_resgroups, args.exit_interval, args.data_test, args.exit_threshold)

    elif args.template == 'ECBSR_decision':
        # model
        args.model = 'ECBSR_decision'   ################
        args.m_ecbsr = 16
        args.c_ecbsr = 64
        args.dm_ecbsr = 2
        args.act = 'prelu'
        
        # data
        args.scale = "2"   #############
        args.dir_data = dir_data
        args.ext = "sep"
        # args.data_range = '1-16/801-802'
        args.patch_size = 192

        # device
        args.device = "2"  #############
        args.n_GPUs = 1

        # pipeline
        args.epochs = 300
        args.lr = 1e-4
        args.batch_size = 16
        args.print_every = 10
        args.decision = True
        args.strategy = "de3"
        args.exit_interval = 2    ############
        # args.save_results = True  ############
        # args.save_gt = True      ############

        # experiemnt
        args.reset = True
        args.pre_train = "/home/shizun/experiment/20220228_ECBSR_my_x2_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20220228_ECBSR_my_x3_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20220228_ECBSR_my_x4_e300_ps192_lr0.0001/model/model_best.pt"
        args.save = "{}_{}_{}_x{}_e{}_ps{}_lr{}_n{}_i{}".format(today, args.model, args.strategy, args.scale, args.epochs, args.patch_size, args.lr, args.n_resblocks, args.exit_interval)

    elif args.template == 'ECBSR_dytest':
        # model
        args.model = 'ECBSR_decision'
        args.m_ecbsr = 16
        args.c_ecbsr = 64
        args.dm_ecbsr = 2
        args.act = 'prelu'

        # data
        args.scale = "4"   #############
        args.dir_data = dir_data
        # args.data_test = 'TEST8K'  #############
        args.data_test = 'DIV2K'  #############
        if args.data_test == 'DIV2K':
            args.data_range = '801-900'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        args.ext = "sep"
        args.patch_size = 48*int(args.scale)
        args.step = 46*int(args.scale)

        # device
        args.device = "1"  #############
        args.n_GPUs = 1

        # pipeline
        args.dynamic = True
        args.test_only = True     ############
        args.ssim = True
        args.exit_interval = 1    ############
        args.exit_threshold = 1 ############
        args.n_parallel = 500         ############
        args.save_results = True  ############
        args.save_gt = True     ############
        # args.add_mask = True

        # experiment
        args.reset = True
        # args.pre_train = "/data/shizun/experiment/20220228_ECBSR_decision_de3_x2_e300_ps192_lr0.0001_n16_i1/model/model_best.pt"
        # args.pre_train = "/data/shizun/experiment/20220228_ECBSR_decision_de3_x3_e300_ps192_lr0.0001_n16_i1/model/model_best.pt"
        args.pre_train = "/data/shizun/experiment/20220228_ECBSR_decision_de3_x4_e300_ps192_lr0.0001_n16_i1/model/model_best.pt"
        args.save = "{}_{}_x{}_e{}_ps{}_st{}_n{}_i{}_{}_th{}_dynamic_test_pic".format(today, args.model, args.scale, args.epochs, args.patch_size, args.step, args.n_resgroups, args.exit_interval, args.data_test, args.exit_threshold)

    elif args.template == 'EDSR_demo':
        # model
        args.model = 'EDSR'   ################
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1
        
        # data
        args.scale = "2"   #############
        args.data_test = 'Demo'  #############
        args.dir_demo = "sample/"

        # device
        args.device = "0,1,2,3"  #############
        args.n_GPUs = 4

        # pipeline
        args.test_only = True
        args.save_results = True 
        # args.chop = True 

        # experiemnt
        # args.reset = True
        args.pre_train = "pretrained/EDSR_x2.pt"
        # args.pre_train = "pretrained/EDSR_x3.pt"
        # args.pre_train = "pretrained/EDSR_x4.pt"

    if args.template == 'ESPCN':
        # model
        args.model = 'ESPCN'   ################
        
        # data
        args.scale = "2"   #############
        args.dir_data = dir_data
        args.ext = "sep"
        # args.data_range = '801-810/801-810'
        args.patch_size = 192

        # device
        args.device = "0"  #############
        args.n_GPUs = 1

        # pipeline
        args.epochs = 300
        args.lr = 1e-4
        args.batch_size = 16
        args.print_every = 10

        # experiemnt
        # args.reset = True
        # args.pre_train = "/home/shizun/experiment/20220321_ESPCN_x2_e300_ps192_lr0.0001/model/model_best.pt"
        # args.save = "{}_{}_x{}_e{}_ps{}_lr{}".format(today, args.model, args.scale, args.epochs, args.patch_size, args.lr)
        # args.save = "{}_{}_x{}_e{}_ps{}_lr{}_overfit_wo_pretrain".format(today, args.model, args.scale, args.epochs, args.patch_size, args.lr)

        # test
        args.data_test = 'DIV2K'  #############
        if args.data_test == 'DIV2K':
            # args.data_range = '801-900'
            args.data_range = '801-810'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        args.test_only = True
        # args.chop = True
        args.ssim = True
        # args.save_results = True  ############
        # args.save_gt = True      ############
        args.pre_train = "/home/shizun/experiment/20220322_ESPCN_x2_e300_ps192_lr0.0001_overfit_w_pretrain/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20220322_ESPCN_x2_e300_ps192_lr0.0001_overfit_wo_pretrain/model/model_best.pt"
        args.save = "{}_{}_x{}_{}_overfit_w_pretrain_test_801-810".format(today, args.model, args.scale, args.data_test)

        # resume
        # args.load = "20211103_EDSR_paper_x4_e600_ps192_lr0.0001"
        # args.resume = -1


    elif args.template == 'ESPCN_test':
        # model
        args.model = 'ESPCN'

        # data
        args.scale = "2"   #############
        args.dir_data = dir_data
        # args.data_test = 'TEST8K'  #############
        args.data_test = 'DIV2K'  #############
        if args.data_test == 'DIV2K':
            args.data_range = '801-810'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        args.ext = "sep"
        args.patch_size = 96*int(args.scale)
        args.step = 94*int(args.scale)

        # device
        args.device = "0"  #############
        args.n_GPUs = 1

        # pipeline
        args.dynamic = True
        args.test_only = True     ############
        args.n_parallel = 400         ############
        args.ssim = True
        # args.save_results = True  ############
        # args.save_gt = True       ############

        # experiment
        args.reset = True
        args.pre_train = "/home/shizun/experiment/20220321_ESPCN_x2_e300_ps192_lr0.0001/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20220322_ESPCN_x2_e300_ps192_lr0.0001_overfit_w_pretrain/model/model_best.pt"
        # args.pre_train = "/home/shizun/experiment/20220322_ESPCN_x2_e300_ps192_lr0.0001_overfit_wo_pretrain/model/model_best.pt"
        args.save = "{}_{}_x{}_e{}_ps{}_st{}_n{}_i{}_{}_static_test_801-810".format(today, args.model, args.scale, args.epochs, args.patch_size, args.step, args.n_resgroups, args.exit_interval, args.data_test)


    elif args.template == 'RRDB':
        # model
        args.model = 'RRDB'
        args.n_resblocks = 23

        # data
        args.scale = "4"
        args.ext = "sep"
        args.dir_data = dir_data
        args.data_train = 'DIV2K'
        args.data_test = 'DIV2K'
        # args.data_test = 'TEST8K'
        args.patch_size = 192

        # device
        args.device = "2"
        args.n_GPUs = 1

        # pipeline
        args.epochs = 300
        args.batch_size = 16
        args.print_every = 10

        # experiment
        # args.reset = True
        # args.save = "{}_{}_x{}_e{}_ps{}_lr{}_n{}_my_data".format(today, args.model, args.scale, args.epochs, args.patch_size, args.lr, args.n_resblocks)

        # resume
        # args.reset = False
        # args.load = "/home/shizun/experiment/20211108_RCAN_paper_x2_e600_ps192_lr0.0001/"
        # args.resume = -1

        # test
        args.test_only = True
        # args.chop = True
        args.ssim = True
        # args.save_results = True  ############
        # args.save_gt = True      ############
        if args.data_test == 'DIV2K':
            args.data_range = '801-900'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        # args.pre_train = "/data/shizun/experiment/20220525_RRDB_x4_e300_ps192_lr0.0001_my_data/model/model_best.pt"
        args.pre_train = "/data/shizun/experiment/20220525_RRDB_x4_e300_ps192_lr0.0001_ori_data/model/model_best.pt"
        args.save = "{}_{}_n23_ori_data_x{}_{}_test".format(today, args.model, args.scale, args.data_test)

    elif args.template == 'RRDB_decision':
        # model
        args.model = 'RRDB_decision'
        args.n_resblocks = 20

        # data
        args.scale = "4"   #############
        args.dir_data = dir_data
        args.ext = "sep"
        # args.data_range = '1-16/801-802'
        args.patch_size = 192

        # device
        args.device = "0"  #############
        args.n_GPUs = 1

        # pipeline
        args.epochs = 300
        args.lr = 1e-4
        args.decay = "200-400"
        args.batch_size = 16
        args.print_every = 10
        args.decision = True
        args.strategy = "de4"
        args.exit_interval = 4    ############
        # args.save_results = True  ############
        # args.save_gt = True      ############

        # experiemnt
        args.reset = True
        args.pre_train = "/data/shizun/experiment/20220525_RRDB_x4_e300_ps192_lr0.0001_n20_my_data/model/model_best.pt"
        args.save = "{}_{}_{}_x{}_e{}_ps{}_lr{}_n{}_i{}".format(today, args.model, args.strategy, args.scale, args.epochs, args.patch_size, args.lr, args.n_resblocks, args.exit_interval)
        
        # resume
        # args.reset = False
        # # args.load = "/data/shizun/experiment/20220526_RRDB_decision_de3_x4_e500_ps192_lr0.0001_n20_i4/"
        # args.load = "/data/shizun/experiment/20220527_RRDB_decision_de3_x4_e500_ps192_lr0.0001_n20_i2/"
        # args.resume = -1

    elif args.template == 'RRDB_dytest':
        # model
        args.model = 'RRDB_decision'
        args.n_resblocks = 20

        # data
        args.scale = "4"   #############
        args.dir_data = dir_data
        # args.data_test = 'TEST8K'  #############
        args.data_test = 'DIV2K'  #############
        if args.data_test == 'DIV2K':
            args.data_range = '801-900'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        args.ext = "sep"
        args.patch_size = 48*int(args.scale)
        args.step = 46*int(args.scale)

        # device
        args.device = "1"  #############
        args.n_GPUs = 1

        # pipeline
        args.epochs = 300
        args.dynamic = True
        args.test_only = True     ############
        args.ssim = True
        args.exit_interval = 4    ############
        args.exit_threshold = 1.0 ############
        args.n_parallel = 500

        # experiment
        args.reset = True
        # args.save_results = True  ############
        # args.save_gt = True     ############
        # args.pre_train = "/data/shizun/experiment/20220525_RRDB_decision_de3_x4_e300_ps192_lr0.0001_n20_i4/model/model_best.pt"
        # args.pre_train = "/data/shizun/experiment/20220526_RRDB_decision_de3_x4_e500_ps192_lr0.0001_n20_i4/model/model_best.pt"
        # args.pre_train = "/data/shizun/experiment/20220526_RRDB_decision_de3_x4_e300_ps192_lr0.0001_n20_i2/model/model_best.pt"
        args.pre_train = "/data/shizun/experiment/20220527_RRDB_decision_eedm_x4_e300_ps192_lr0.0001_n20_i4/model/model_best.pt"
        args.save = "{}_{}_x{}_e{}_ps{}_st{}_n{}_i{}_{}_th{}_dynamic_test".format(today, args.model, args.scale, args.epochs, args.patch_size, args.step, args.n_resblocks, args.exit_interval, args.data_test, args.exit_threshold)

    elif args.template == 'RRDB_check':
        # model
        args.model = 'RRDB_check'
        args.n_resblocks = 20

        # data
        args.scale = "4"   #############
        args.dir_data = dir_data
        # args.data_test = 'TEST8K'  #############
        args.data_test = 'DIV2K'  #############
        if args.data_test == 'DIV2K':
            args.data_range = '801-900'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        args.ext = "sep"
        args.patch_size = 48*int(args.scale)
        args.step = 46*int(args.scale)

        # device
        args.device = "2"  #############
        args.n_GPUs = 1

        # pipeline
        args.check = True
        args.test_only = True     ############
        args.exit_interval = 4    ############
        args.exit_threshold = 1 ############
        args.n_parallel = 500         ############
        # args.save_results = True  ############
        # args.save_gt = True     ############
        # args.add_mask = True

        # experiment
        args.reset = True
        args.pre_train = "/data/shizun/experiment/20220525_RRDB_decision_de3_x4_e300_ps192_lr0.0001_n20_i4/model/model_best.pt"
        args.save = "{}_{}_x{}_e{}_ps{}_st{}_n{}_i{}_{}_th{}_check".format(today, args.model, args.scale, args.epochs, args.patch_size, args.step, args.n_resblocks, args.exit_interval, args.data_test, args.exit_threshold)

    elif args.template == 'RRDB_eedm':
        # model
        args.model = 'RRDB_decision'   ################
        args.n_resblocks = 20

        
        # data
        args.scale = "4"   #############
        args.dir_data = dir_data
        args.ext = "sep"
        # args.data_range = '1-16/801-802'
        args.patch_size = 192

        # device
        args.device = "1"  #############
        args.n_GPUs = 1

        # pipeline
        args.epochs = 300
        args.loss = '1*MSE'
        args.lr = 1e-4
        args.batch_size = 16
        args.print_every = 10
        args.decision = True
        args.strategy = "eedm1"
        args.exit_interval = 4    ############
        # args.save_results = True  ############
        # args.save_gt = True      ############

        # experiemnt
        args.reset = True
        args.pre_train = "/data/shizun/experiment/20220525_RRDB_decision_de3_x4_e300_ps192_lr0.0001_n20_i4/model/model_best.pt"
        args.save = "{}_{}_{}_x{}_e{}_ps{}_lr{}_n{}_i{}".format(today, args.model, args.strategy, args.scale, args.epochs, args.patch_size, args.lr, args.n_resblocks, args.exit_interval)


    elif args.template == 'SWINIR':
        # model
        args.model = 'SWINIR'
        args.n_resblocks = 6

        # data
        args.scale = "4"
        args.ext = "sep"
        args.dir_data = dir_data
        args.data_train = 'DIV2K'
        args.data_test = 'DIV2K'
        # args.data_test = 'TEST8K'
        args.patch_size = 192

        # device
        args.device = "2"
        args.n_GPUs = 1

        # pipeline
        args.epochs = 500
        args.lr = 1e-4
        args.decay = "200-400"
        args.batch_size = 16
        args.print_every = 10

        # experiment
        # args.reset = True
        # args.save = "{}_{}_x{}_e{}_ps{}_lr{}_n{}_my_data".format(today, args.model, args.scale, args.epochs, args.patch_size, args.lr, args.n_resblocks)

        # resume
        # args.reset = False
        # args.load = "/home/shizun/experiment/20220528_SWINIR_x4_e500_ps192_lr0.0001_n6_my_data/"
        # args.resume = -1

        # test
        args.test_only = True
        # args.chop = True
        args.ssim = True
        # args.save_results = True  ############
        # args.save_gt = True      ############
        if args.data_test == 'DIV2K':
            args.data_range = '801-900'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        # args.pre_train = "/data/shizun/experiment/20220526_SWINIR_x4_e300_ps192_lr0.0001_n6_my_data/model/model_best.pt"
        args.pre_train = "/data/shizun/experiment/20220528_SWINIR_x4_e500_ps192_lr0.0001_n6_my_data/model/model_best.pt"
        args.save = "{}_{}_ori_data_x{}_{}_e500_test".format(today, args.model, args.scale, args.data_test)


    elif args.template == 'SWINIR_decision':
        # model
        args.model = 'SWINIR_decision'

        # data
        args.scale = "4"   #############
        args.dir_data = dir_data
        args.ext = "sep"
        # args.data_range = '1-16/801-802'
        args.patch_size = 192

        # device
        args.device = "3"  #############
        args.n_GPUs = 1

        # pipeline
        args.epochs = 300
        args.lr = 1e-4
        args.decay = "200-400"
        args.batch_size = 16
        args.print_every = 10
        args.decision = True
        args.strategy = "de3"
        args.exit_interval = 1    ############
        # args.save_results = True  ############
        # args.save_gt = True      ############

        # experiemnt
        # args.reset = True
        # args.pre_train = "/data/shizun/experiment/20220526_SWINIR_x4_e300_ps192_lr0.0001_n6_my_data/model/model_best.pt"
        # args.save = "{}_{}_{}_x{}_e{}_ps{}_lr{}_n6_i{}".format(today, args.model, args.strategy, args.scale, args.epochs, args.patch_size, args.lr, args.exit_interval)
        
        # resume
        args.reset = False
        args.load = "/data/shizun/experiment/20220527_SWINIR_decision_de3_x4_e300_ps192_lr0.0001_n6_i1/"
        args.resume = -1


    elif args.template == 'SWINIR_dytest':
        # model
        args.model = 'SWINIR_decision'

        # data
        args.scale = "4"   #############
        args.dir_data = dir_data
        # args.data_test = 'TEST8K'  #############
        args.data_test = 'DIV2K'  #############
        if args.data_test == 'DIV2K':
            args.data_range = '801-900'
        elif args.data_test == 'TEST8K':
            args.data_range = '1-100'
        args.ext = "sep"
        args.patch_size = 48*int(args.scale)
        args.step = 46*int(args.scale)

        # device
        args.device = "3"  #############
        args.n_GPUs = 1

        # pipeline
        args.epochs = 300
        args.dynamic = True
        args.test_only = True     ############
        args.ssim = True
        args.exit_interval = 1    ############
        args.exit_threshold = 0.8 ############
        args.n_parallel = 500

        # experiment
        args.reset = True
        # args.save_results = True  ############
        # args.save_gt = True     ############
        args.pre_train = "/data/shizun/experiment/20220527_SWINIR_decision_de3_x4_e300_ps192_lr0.0001_n6_i1/model/model_best.pt"
        args.save = "{}_{}_x{}_e{}_ps{}_st{}_n{}_i{}_{}_th{}_dynamic_test".format(today, args.model, args.scale, args.epochs, args.patch_size, args.step, args.n_resblocks, args.exit_interval, args.data_test, args.exit_threshold)
