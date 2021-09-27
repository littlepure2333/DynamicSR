machine  = {"A3":"/data/shizun/dataset/", 
            "B3":"/data/shizun/dataset/", 
            "C4":"/data/shizun/dataset/",
            "4gpu-2":"/home/shizun/datasets/image_process/",
            "4gpu-5":"/data/shizun/"}
dir_data = machine["4gpu-5"]

import datetime
today = datetime.datetime.now().strftime('%Y%m%d')

def set_template(args):
    # Set the templates here
    if args.template.find('jpeg') >= 0:
        args.data_train = 'DIV2K_jpeg'
        args.data_test = 'DIV2K_jpeg'
        args.epochs = 200
        args.decay = '100'

    if args.template.find('EDSR_paper') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

        args.dir_data = dir_data
        args.scale = "2"
        args.device = "0,1"
        args.n_GPUs = 2
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        # args.reset = False
        args.epochs = 5000
        # args.data_range = '1-320/801-810'

        # args.load = "20210609_EDSR_x3_ps192_lr0.0001_e5000_1*L1_nf256_p0.1"
        # args.resume = -1

        # args.data_train = 'DIV2K_PSNR'
        # args.data_test = 'DIV2K'
        # args.data_partion = 0.1
        # args.file_suffix = "_psnr_up_new.pt" #

        # args.save = "{}_{}_x{}_ps{}_lr{}_e{}_{}_nf{}_p{}".format(today, args.model, args.scale, args.patch_size, args.lr, args.epochs, args.loss, args.n_feats, args.data_partion)
        # args.seed = 6
        args.chop = True
        args.ssim = True
        
        args.save = "20210619_EDSR_x2_ps192_lr0.0001_e5000_1*L1_nf256_p1_test"
        
        args.test_only = 'True'
        args.data_test = 'Set5+Set14+B100+Urban100'

        args.pre_train = "/home/shizun/experiment/20210609_EDSR_x2_ps192_lr0.0001_e5000_1*L1_nf256_p1/model/model_best.pt"


    if args.template.find('MDSR') >= 0:
        args.model = 'MDSR'
        args.patch_size = 48
        args.epochs = 650

    if args.template.find('DDBPN') >= 0:
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

    if args.template.find('GAN') >= 0:
        args.epochs = 200
        args.lr = 5e-5
        args.decay = '150'

    if args.template.find('RCAN') >= 0:
        args.model = 'RCAN'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64
        args.chop = True

    if args.template.find('VDSR') >= 0:
        args.model = 'VDSR'
        args.n_resblocks = 20
        args.n_feats = 64
        args.patch_size = 41
        args.lr = 1e-1

    if args.template.find('VDSR_psnr') >= 0:
        args.model = 'VDSR'
        args.n_resblocks = 20
        args.n_feats = 64
        args.lr = 1e-1
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "4"
        args.save = "20210226_vdsr_x4_r20_f64_lr1e-1_baseline"
        args.device = "1"
        args.n_GPUs = 1
        args.batch_size = 32
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        # args.data_train = 'DIV2K_PSNR'
        # args.data_test = 'DIV2K'
        # args.data_partion = 0.5
        # args.file_suffix = "_psnr_up_new.pt"
        ### inference all images and save results
        # args.test_only = 'True'
        # args.save_results = 'True'
        # args.data_range = '1-810'
        # args.pre_train = "/home/shizun/experiment/20210219_vdsr_x2_r20_f64_lr1e-1_p0.1/model/model_best.pt"

    if args.template.find('EDSR_baseline') >= 0:
        args.model = 'EDSR'
        args.lr = 1e-5
        args.n_resblocks = 24
        args.n_feats = 224
        args.res_scale = 1
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "2"
        args.device = "2,3"
        args.n_GPUs = 2
        args.batch_size = 16
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        # args.save = "20210317_edsr_x4_r24_f224_lr1e-5_cutblur0.5_inferenece"
        args.save = "{}_{}_x{}_ps{}_lr{}".format(today, args.model, args.scale, args.patch_size, args.lr)

        ### inference all images and save results
        # args.cutblur = 0
        # args.data_test = 'Set5+Set14+B100+Urban100'
        # args.test_only = 'True'
        # args.save_results = 'True'
        args.data_range = '1-10/11-12'
        # args.pre_train = "/home/shizun/experiment/20210314_edsr_x4_r24_f224_lr1e-5_cutblur0.5_new/model/model_best.pt"
        
        # args.load = "20201104_edsr_x2_r24_f224_lr1e-5_baseline"
        # args.resume = -1
        # args.test_only = 'True'
        # args.save_results = 'True'
        # args.save_gt = True

        # args.save = "20210526_edsr_x2_r24_f224_lr1e-5_baseline_test"
        
        # args.test_only = 'True'
        # args.data_test = 'Set5+Set14+B100+Urban100'

        # args.pre_train = "/home/shizun/experiment/edsr/20201104_edsr_x2_r24_f224_lr1e-5_baseline/model/model_best.pt"


    if args.template.find('EDSR_psnr') >= 0:
        args.model = 'EDSR'
        args.lr = 1e-5
        args.n_resblocks = 24
        args.n_feats = 224
        args.res_scale = 1
        args.patch_size = 256
        args.dir_data = dir_data
        args.scale = "4" #
        # args.save = "20210317_edsr_x4_r24_f224_lr1e-5_p0.5_pretrain"
        # args.pre_train = "/home/shizun/experiment/edsr/20201124_edsr_x4_r24_f224_lr1e-5_baseline/model/model_best.pt"
        args.device = "0,1"
        args.n_GPUs = 2
        args.batch_size = 32
        args.print_every = 4
        args.ext = "sep"
        args.reset = True
        # args.reset = False #
        # args.resume = -1 #
        # args.epochs = 900 #
        args.data_train = 'DIV2K_PSNR'
        args.data_test = 'DIV2K'
        args.data_partion = 0.01
        args.file_suffix = "_psnr_up_p256.pt" #

        args.save = "{}_{}_x{}_ps{}_lr{}_e{}_{}_nf{}_p{}".format(today, args.model, args.scale, args.patch_size, args.lr, args.epochs, args.loss, args.n_feats, args.data_partion)

        # args.save = "20210526_edsr_x2_r24_f224_lr1e-5_baseline_test"
        
        # args.test_only = 'True'
        # args.data_test = 'Set5+Set14+B100+Urban100'

        # args.pre_train = "/home/shizun/experiment/edsr/20201104_edsr_x2_r24_f224_lr1e-5_baseline/model/model_best.pt"



    if args.template.find('EDSR_cutblur') >= 0:
        args.model = 'EDSR'
        args.lr = 1e-5
        args.n_resblocks = 24
        args.n_feats = 224
        args.res_scale = 1
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "4"
        args.save = "20210317_edsr_x4_r24_f224_lr1e-5_cutblur0.5_pretrain"
        args.pre_train = "/home/shizun/experiment/edsr/20201124_edsr_x4_r24_f224_lr1e-5_baseline/model/model_best.pt"
        args.device = "0,1"
        args.n_GPUs = 2
        args.batch_size = 32
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        args.cutblur = 0.5
        args.data_train = 'DIV2K_PSNR'
        args.data_test = 'DIV2K'
        args.data_partion = 0.1
        args.file_suffix = "_psnr_up_new.pt"

    if args.template.find('EDSR_dynamic') >= 0:
        args.model = 'EDSR_dynamic'
        args.lr = 1e-4
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "2"
        args.device = "0,1"
        args.n_GPUs = 2
        args.batch_size = 16
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        args.save_results = 'True'
        args.save_gt = True
        args.dynamic = True
        # args.save = "20210317_edsr_x4_r24_f224_lr1e-5_cutblur0.5_inferenece"
        args.save = "{}_{}_forward_every_x{}_ps{}_lr{}".format(today, args.model, args.scale, args.patch_size, args.lr)

        ### inference all images and save results
        # args.cutblur = 0
        # args.data_test = 'Set5+Set14+B100+Urban100'
        # args.test_only = 'True'
        # args.save_results = 'True'
        # args.data_range = '1-10/11-12'
        # args.model = 'EDSR'
        # args.lr = 1e-5
        # args.n_resblocks = 24
        # args.n_feats = 224
        # args.res_scale = 1
        # args.patch_size = 192
        # args.dir_data = dir_data
        # args.scale = "4" #
        # args.save = "20210315_edsr_x4_r24_f224_lr1e-5_dynamic_1e-6~0.3"
        # args.device = "0,1"
        # args.n_GPUs = 2
        # args.batch_size = 32
        # args.print_every = 4
        # args.ext = "sep"
        # args.reset = True
        # args.data_train = 'DIV2K_PSNR'
        # args.data_test = 'DIV2K'
        # args.data_partion = 1e-6
        # args.final_data_partion = 0.3
        # args.file_suffix = "_psnr_up_new.pt" #
        

    if args.template.find('EDSR_switchable') >= 0:
        args.model = 'EDSR_switchable'
        args.lr = 1e-4
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1
        args.patch_size = 192
        args.epochs = 300
        args.dir_data = dir_data
        args.scale = "2"
        args.device = "3"
        args.n_GPUs = 1
        args.batch_size = 16
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        args.save_results = 'True'
        args.save_gt = True
        args.data_train = 'DIV2K_SWITCHABLE'
        args.data_test = 'DIV2K'
        args.switchable = True
        args.data_part_list = ('easy_x2_descending', 'midd_x2_descending', 'hard_x2_descending')
        args.width_mult_list = (0.33, 0.67, 1.0)
        args.save = "{}_{}_x{}_e{}_ps{}_lr{}_recurrent_easier".format(today, args.model, args.scale, args.epochs, args.patch_size, args.lr)

    if args.template.find('EDSR_32') >= 0:
        args.model = 'EDSR'
        args.lr = 1e-5
        args.n_resblocks = 24
        args.n_feats = 224
        args.res_scale = 1
        args.patch_size = 32 #
        args.dir_data = dir_data
        args.scale = "4" #
        args.save = "20210308_edsr_x4_r24_f224_lr1e-5_ps32_p0.1" #
        args.device = "2,3" #
        args.n_GPUs = 2
        args.batch_size = 64
        args.print_every = 4
        args.ext = "sep"
        args.reset = True
        args.data_train = 'DIV2K_PSNR'
        args.data_test = 'DIV2K'
        args.data_partion = 0.1 #
        args.file_suffix = "_psnr_up_p32.pt" #

    if args.template.find('EDSR_ohem') >= 0:
        args.model = 'EDSR'
        args.lr = 1e-5
        args.n_resblocks = 24
        args.n_feats = 224
        args.res_scale = 1
        args.patch_size = 192 #
        args.dir_data = dir_data
        args.scale = "2" #
        args.device = "2,3" #
        args.n_GPUs = 2
        args.batch_size = 16
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        args.ohem = True
        # args.cpu = True
        # args.data_train = 'DIV2K_PSNR'
        # args.data_test = 'DIV2K'
        args.data_partion = 0.5 #
        # args.file_suffix = "_psnr_up_p96.pt" #
        args.save = "{}_{}_x{}_ps{}_ohem{}".format(today, args.model, args.scale, args.patch_size, args.data_partion)

    if args.template.find('EDSR_patchnet') >= 0:
        args.model = 'EDSR'
        args.lr = 1e-5
        args.n_resblocks = 24
        args.n_feats = 224
        args.res_scale = 1
        args.patch_size = 192 #
        args.dir_data = dir_data
        args.scale = "2" #
        args.device = "2,3" #
        args.n_GPUs = 2
        args.batch_size = 16
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        args.patchnet = True
        # args.cpu = True
        # args.data_train = 'DIV2K_PSNR'
        # args.data_test = 'DIV2K'
        args.save = "{}_{}_x{}_ps{}_patchnet".format(today, args.model, args.scale, args.patch_size)

    if args.template.find('EDSR_psnr_nms') >= 0:
        args.model = 'EDSR'
        args.lr = 1e-5
        args.n_resblocks = 24
        args.n_feats = 224
        args.res_scale = 1
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "2"
        args.save = "20210208_edsr_x2_r24_f224_lr1e-5_psnr_nms_1000*0.1"
        args.device = "0,1"
        args.n_GPUs = 2
        args.batch_size = 32
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        args.data_train = 'DIV2K_PSNR'
        args.data_test = 'DIV2K_PSNR'
        args.data_partion = 0.1
        args.file_suffix = "_psnr_nms_1000.pt"
        # args.data_range = '1-200/201-210'

    if args.template.find('EDSR_psnr_darts') >= 0:
        args.model = 'EDSR'
        args.lr = 1e-5
        args.n_resblocks = 24
        args.n_feats = 224
        args.res_scale = 1
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "2"
        args.save = "20210208_edsr_x2_r24_f224_lr1e-5_psnr_darts_1e4*0.1"
        args.device = "2,3"
        args.n_GPUs = 2
        args.batch_size = 32
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        args.data_train = 'DIV2K_PSNR'
        args.data_test = 'DIV2K_PSNR'
        args.data_partion = 0.1
        args.file_suffix = "_psnr_darts.pt"
        # args.data_range = '1-200/201-210'

    if args.template.find('EDSR_std') >= 0:
        args.model = 'EDSR'
        args.lr = 1e-5
        args.n_resblocks = 24
        args.n_feats = 224
        args.res_scale = 1
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "2"
        args.save = "202103014_edsr_x2_r24_f224_lr1e-5_std0_rgb_p192_s0.01_new"
        args.device = "0,1"
        args.n_GPUs = 2
        args.batch_size = 32
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        args.data_train = 'DIV2K_PSNR'
        args.data_test = 'DIV2K'
        args.data_partion = 0.01
        args.file_suffix = "_std0_rgb_p192_new.pt"
        # args.data_range = '1-200/201-210'
    
    if args.template.find('EDSR_auto') >= 0:
        args.model = 'EDSR'
        args.lr = 1e-5
        args.n_resblocks = 24
        args.n_feats = 224
        args.res_scale = 1
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "2"
        args.save = "20201111_edsr_x2_r24_f224_lr1e-5_auto_debug"
        args.device = "2,3"
        args.n_GPUs = 2
        args.batch_size = 32
        args.print_every = 5
        args.ext = "sep"
        args.reset = True
        args.assistant = True
        args.base_prob = 0.3
        args.sec_method = 'upsample_PSNR'
        # args.epochs = 3

    if args.template.find('RCAN_psnr') >= 0:
        args.model = 'RCAN'
        args.lr = 1e-5
        args.n_resgroups = 6
        args.n_resblocks = 20
        args.n_feats = 64
        args.patch_size = 192
        args.chop = False
        args.dir_data = dir_data
        args.scale = "4"
        args.save = "20210218_rcan_x4_g6_r20_f64_lr1e-5_p0.00001_new"
        args.device = "2,3"
        args.n_GPUs = 2
        args.batch_size = 16
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        args.data_train = 'DIV2K_PSNR'
        args.data_test = 'DIV2K_PSNR'
        args.data_partion = 0.00001
        args.file_suffix = "_psnr_up_new.pt"

    if args.template.find('RCAN_dynamic') >= 0:
        args.model = 'RCAN'
        args.lr = 1e-5
        args.n_resgroups = 6
        args.n_resblocks = 20
        args.n_feats = 64
        args.patch_size = 192
        args.chop = False
        args.dir_data = dir_data
        args.scale = "4"
        args.save = "20210313_rcan_x4_g6_r20_f64_lr1e-5_dynamic_1e-6~0.3"
        args.device = "2,3"
        args.n_GPUs = 2
        args.batch_size = 16
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        args.data_train = 'DIV2K_PSNR'
        args.data_test = 'DIV2K'
        args.data_partion = 1e-6
        args.final_data_partion = 0.3
        args.file_suffix = "_psnr_up_new.pt"

    if args.template.find('RCAN_psnr_nms') >= 0:
        args.model = 'RCAN'
        args.lr = 1e-5
        args.n_resgroups = 6
        args.n_resblocks = 20
        args.n_feats = 64
        args.patch_size = 192
        args.chop = False
        args.dir_data = dir_data
        args.scale = "2"
        args.save = "20210302_rcan_x2_g6_r20_f64_lr1e-5_psnr_nms_1e3*0.1"
        args.device = "2,3"
        args.n_GPUs = 2
        args.batch_size = 16
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        args.data_train = 'DIV2K_PSNR'
        args.data_test = 'DIV2K'
        args.data_partion = 0.1
        args.file_suffix = "_psnr_nms_1000.pt"

    if args.template.find('RCAN_psnr_darts') >= 0:
        args.model = 'RCAN'
        args.lr = 1e-5
        args.n_resgroups = 6
        args.n_resblocks = 20
        args.n_feats = 64
        args.patch_size = 192
        args.chop = False
        args.dir_data = dir_data
        args.scale = "2"
        args.save = "20210302_rcan_x2_g6_r20_f64_lr1e-5_psnr_darts_1e3*0.01"
        args.device = "0,1"
        args.n_GPUs = 2
        args.batch_size = 16
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        args.data_train = 'DIV2K_PSNR'
        args.data_test = 'DIV2K'
        args.data_partion = 0.01
        args.file_suffix = "_psnr_up_new.pt"

    if args.template.find('ESPCN_psnr') >= 0:
        args.model = 'ESPCN'
        args.lr = 1e-3
        args.patch_size = 192
        args.epochs = 1000
        args.dir_data = dir_data
        args.scale = "2"
        # args.save = "20210225_espcn_x2_b64_lr1e-5_p0.5"
        # args.load = "espcn/20210302_espcn_x2_b64_lr1e-5_psnr_nms_1e3*1" #
        args.device = "0"
        args.n_GPUs = 1
        args.batch_size = 16
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        # args.save = "{}_{}_x{}_ps{}_lr{}_e{}_{}_bs{}".format(today, args.model, args.scale, args.patch_size, args.lr, args.epochs, args.loss, args.batch_size)
        # args.reset = False #
        # args.resume = -1 #
        # args.epochs = 900 #
        # args.data_train = 'DIV2K_PSNR'
        # args.data_test = 'DIV2K_PSNR'
        # args.data_partion = 1
        # args.file_suffix = "_psnr_nms_1000.pt" #
        
        args.save = "20210530_ESPCN_x2_ps192_lr0.001_e5000_1*L1_bs16_test"
        
        args.test_only = 'True'
        args.data_test = 'Set5+Set14+B100+Urban100'

        args.pre_train = "/home/shizun/experiment/20210530_ESPCN_x2_ps192_lr0.001_e5000_1*L1_bs16/model/model_best.pt"

        
    if args.template.find('SRCNN_psnr') >= 0:
        args.model = 'SRCNN'
        args.lr = 1e-4
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "4"
        args.epochs = 1000
        # args.save = "20210225_SRCNN_x4_b32_lr1e-5_baseline"
        args.device = "3"
        args.n_GPUs = 1
        args.batch_size = 16
        args.print_every = 10
        args.ext = "sep"
        args.reset = True # !!!!!!!!!

        # args.data_train = 'DIV2K_PSNR'
        # args.data_test = 'DIV2K_PSNR'
        # args.data_partion = 0.1
        # args.file_suffix = "_psnr_up_new.pt"

        # args.save = "{}_{}_x{}_ps{}_lr{}_e{}_{}_bs{}_p{}".format(today, args.model, args.scale, args.patch_size, args.lr, args.epochs, args.loss, args.batch_size, args.data_partion)
        
        args.save = "20210605_SRCNN_x4_ps192_lr0.0001_e1000_1*L1_bs16_p0.1_test"
        args.test_only = 'True'
        args.data_test = 'Set5+Set14+B100+Urban100'
        args.pre_train = "/home/shizun/experiment/20210605_SRCNN_x4_ps192_lr0.0001_e1000_1*L1_bs16_p0.1/model/model_best.pt"

    if args.template.find('benchmark') >= 0:
        args.model = 'EDSR'
        args.lr = 1e-5
        args.n_resblocks = 24
        args.n_feats = 224
        args.res_scale = 1
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "3"
        args.save = "20210228_edsr_x3_r24_f224_lr1e-5_benchmark_baseline"
        args.device = "0,1"
        args.n_GPUs = 2
        args.batch_size = 32
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        args.data_train = 'Set5+Set14+B100+Urban100'
        args.data_test = 'Set5+Set14+B100+Urban100'

    if args.template.find('benchmark_psnr') >= 0:
        args.model = 'EDSR'
        args.lr = 1e-5
        args.n_resblocks = 24
        args.n_feats = 224
        args.res_scale = 1
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "2"
        args.save = "20210218_edsr_x2_r24_f224_lr1e-5_benchmark_p0.001"
        args.device = "0,1"
        args.n_GPUs = 2
        args.batch_size = 32
        args.print_every = 10
        args.ext = "sep"
        args.reset = True
        args.data_train = 'Set5_PSNR+Set14_PSNR+B100_PSNR+Urban100_PSNR'
        args.data_test = 'Set5_PSNR+Set14_PSNR+B100_PSNR+Urban100_PSNR'
        args.data_partion = 0.001
        args.file_suffix = "_psnr_up_new.pt"
        # args.data_range = '1-200/201-210'

    if args.template.find('benchmark_test') >= 0:
        # args.model = 'SRCNN'
        
        # args.model = 'ESPCN'

        # args.model = 'EDSR'
        # args.n_resblocks = 24
        # args.n_feats = 224
        # args.res_scale = 1
        
        # args.model = 'RCAN'
        # args.lr = 1e-5
        # args.n_resgroups = 6
        # args.n_resblocks = 20
        # args.n_feats = 64

        # args.model = 'RDN'
        # args.G0 = 64
        # args.RDNconfig = 'B'

        args.model = 'HAN'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64

        # rename
        load_folder = ""
        # load_name = "20210225_SRCNN_x4_b32_lr1e-5_baseline"
        # load_name = "20210222_SRCNN_x4_b32_lr1e-5_p0.000001_new"
        # load_name = "20201124_edsr_x4_r24_f224_lr1e-5_baseline"
        # load_name = "20210207_edsr_x4_r24_f224_lr1e-5_p0.1_new"
        # load_name = "20210622_HAN_x4_ps192_lr0.0001_e300_1*L1"
        # load_name = "20210622_HAN_x4_ps192_lr0.0001_e300_1*L1_p0.1"

        load_name = "20210624_HAN_x2_ps192_lr0.0001_e300_1*L1_p1e-06"
        clean_name = load_name.split('_',1)[-1].split('_new')[0]
        save_folder = "benchmark/"
        args.scale = load_name.split('x', 1)[-1][0]
        # rename
        args.save = "{}_benchmark".format(save_folder+clean_name)
        args.device = "0,1"
        args.n_GPUs = 2
        args.chop = True
        # args.batch_size = 32
        # args.print_every = 10
        # args.ext = "sep"
        args.reset = 'False'
        args.data_test = 'Set5+Set14+B100+Urban100'
        # args.data_test = 'Urban100'
        args.test_only = 'True'
        args.pre_train = "/home/shizun/experiment/{}/model/model_best.pt".format(load_folder+load_name)

        # DIV2k
        # args.save_results = 'True'
        # args.data_test = 'DIV2K'
        # args.data_range = '1-9'
        # args.save_psnr_list = True

    if args.template.find('RDN_psnr') >= 0:
        args.model = 'RDN'
        args.lr = 1e-5
        args.G0 = 64
        args.RDNconfig = 'B'
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "2"
        # args.save = "20210309_RDN_x4_B_G064_b8_lr1e-5_p0.5"
        args.device = "2,3"
        args.n_GPUs = 2
        args.batch_size = 8
        args.print_every = 4
        args.ext = "sep"
        args.reset = True
        # args.data_train = 'DIV2K_PSNR'
        # args.data_test = 'DIV2K'
        # args.data_partion = 0.3
        # args.file_suffix = "_psnr_up_new.pt"
        args.save = "{}_{}_x{}_ps{}_lr{}_e{}_{}_p{}".format(today, args.model, args.scale, args.patch_size, args.lr, args.epochs, args.loss, args.data_partion)

    if args.template.find('RDN_dynamic') >= 0:
        args.model = 'RDN'
        args.lr = 1e-5
        args.G0 = 64
        args.RDNconfig = 'B'
        args.patch_size = 192
        args.dir_data = dir_data
        args.scale = "2"
        args.save = "20210313_RDN_x2_B_G064_b8_lr1e-5_dynamic_1e-6~0.3"
        args.device = "0,1"
        args.n_GPUs = 2
        args.batch_size = 8
        args.print_every = 4
        args.ext = "sep"
        args.reset = True
        args.data_train = 'DIV2K_PSNR'
        args.data_test = 'DIV2K'
        args.data_partion = 1e-6
        args.final_data_partion = 0.3
        args.file_suffix = "_psnr_up_new.pt"

    if args.template.find('SRVC') >= 0:
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

    if args.template.find('SAN') >= 0:
        args.model = 'SAN'
        args.n_feats = 64
        args.n_resgroups = 10
        args.n_resblocks = 5

        args.lr = 1e-4
        args.loss = '1*L1'
        args.patch_size = 192
        args.scale = "4"
        args.batch_size = 16
        args.epochs = 300

        args.device = "2,3"
        args.n_GPUs = 2
        args.dir_data = dir_data
        args.ext = "sep"
        args.print_every = 5
        args.chop = True

        args.save = "{}_{}_x{}_ps{}_lr{}_e{}_{}".format(today, args.model, args.scale, args.patch_size, args.lr, args.epochs, args.loss)
        # args.load = "20210421_SRVC_x2_ps200_lr0.0001_e5000_1*L1"
        # args.resume = -1
        args.reset = True ######!!!!!!!!!

    if args.template.find('HAN') >= 0:
        args.model = 'HAN'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64

        args.lr = 1e-4
        args.loss = '1*L1'
        args.patch_size = 192
        args.scale = "2"
        args.batch_size = 16
        args.epochs = 300

        args.device = "0,1,2,3"
        args.n_GPUs = 4
        args.dir_data = dir_data
        args.ext = "sep"
        args.print_every = 5
        args.chop = True

        # args.data_train = 'DIV2K_PSNR'
        # args.data_test = 'DIV2K'
        # args.data_partion = 0.00001
        # args.file_suffix = "_psnr_up_new.pt"

        # args.save = "{}_{}_x{}_ps{}_lr{}_e{}_{}_p{}".format(today, args.model, args.scale, args.patch_size, args.lr, args.epochs, args.loss, args.data_partion)
        # args.reset = True ######!!!!!!!!!

        # for test
        # args.save = "benchmark/{}_{}_x{}_base".format(today, args.model, args.scale)
        args.save = "benchmark/{}_{}_x{}_ours".format(today, args.model, args.scale)
        args.pre_train = "/home/shizun/experiment/20210622_HAN_x2_ps192_lr0.0001_e300_1*L1/model/model_best.pt"
        args.test_only = 'True'
        args.data_test = 'Set5+Set14+B100+Urban100'

        # args.data_test = 'DIV2K'
        # args.data_range = '1-320/801-810'
        # args.save_results = 'True'
        # args.save_gt = True

