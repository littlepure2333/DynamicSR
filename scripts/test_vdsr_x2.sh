device=2


# python main.py --template VDSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 2 --exit_threshold 1 \
# --pre_train /data/shizun/experiment/20220301_VDSR_decision_de3_x2_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# # [DIV2K x2] PSNR: 33.576	SSIM: 0.9287	Threshold: 1.0	Average exits:[7.95/8]	Flops:6.13GFlops (99.414%) avg pass time: 0.59s
# # [DIV2K x2] PSNR: 33.621	SSIM: 0.9292	Threshold: 1.0	Average exits:[7.95/8]	Flops:6.14GFlops (99.499%) avg pass time: 0.41s

# python main.py --template VDSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 2 --exit_threshold 0.99 \
# --pre_train /data/shizun/experiment/20220301_VDSR_decision_de3_x2_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# # [DIV2K x2] PSNR: 33.573	SSIM: 0.9287	Threshold: 0.99	Average exits:[6.32/8]	Flops:5.02GFlops (81.408%) avg pass time: 0.64s
# # [DIV2K x2] PSNR: 33.618	SSIM: 0.9293	Threshold: 0.99	Average exits:[6.22/8]	Flops:4.96GFlops (80.317%) avg pass time: 0.32s

# python main.py --template VDSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 2 --exit_threshold 0.98 \
# --pre_train /data/shizun/experiment/20220301_VDSR_decision_de3_x2_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# # [DIV2K x2] PSNR: 33.571	SSIM: 0.9287	Threshold: 0.98	Average exits:[5.90/8]	Flops:4.74GFlops (76.764%) avg pass time: 0.63s
# # [DIV2K x2] PSNR: 33.615	SSIM: 0.9292	Threshold: 0.98	Average exits:[5.76/8]	Flops:4.64GFlops (75.212%) avg pass time: 0.31s

# python main.py --template VDSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 2 --exit_threshold 0.97 \
# --pre_train /data/shizun/experiment/20220301_VDSR_decision_de3_x2_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# # [DIV2K x2] PSNR: 33.568	SSIM: 0.9286	Threshold: 0.97	Average exits:[5.53/8]	Flops:4.48GFlops (72.690%) avg pass time: 0.59s
# # [DIV2K x2] PSNR: 33.612	SSIM: 0.9292	Threshold: 0.97	Average exits:[5.39/8]	Flops:4.39GFlops (71.117%) avg pass time: 0.31s

# python main.py --template VDSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 2 --exit_threshold 0.96 \
# --pre_train /data/shizun/experiment/20220301_VDSR_decision_de3_x2_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# # [DIV2K x2] PSNR: 33.565	SSIM: 0.9286	Threshold: 0.96	Average exits:[5.22/8]	Flops:4.27GFlops (69.270%) avg pass time: 0.57s
# # [DIV2K x2] PSNR: 33.609	SSIM: 0.9292	Threshold: 0.96	Average exits:[5.06/8]	Flops:4.16GFlops (67.468%) avg pass time: 0.29s








# python main.py --template VDSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 1 \
# --pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [DIV2K x2] PSNR: 33.563	SSIM: 0.9282	Threshold: 1.0	Average exits:[16.04/17]	Flops:5.84GFlops (94.682%) avg pass time: 0.64s
# # [DIV2K x2] PSNR: 33.608	SSIM: 0.9288	Threshold: 1.0	Average exits:[16.04/17]	Flops:5.84GFlops (94.725%) avg pass time: 0.35s

# python main.py --template VDSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.99 \
# --pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [DIV2K x2] PSNR: 33.546	SSIM: 0.9282	Threshold: 0.99	Average exits:[11.13/17]	Flops:4.17GFlops (67.513%) avg pass time: 0.78s

# python main.py --template VDSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.98 \
# --pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [DIV2K x2] PSNR: 33.539	SSIM: 0.9282	Threshold: 0.98	Average exits:[10.14/17]	Flops:3.83GFlops (62.061%) avg pass time: 0.71s

# python main.py --template VDSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.97 \
# --pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [DIV2K x2] PSNR: 33.529	SSIM: 0.9281	Threshold: 0.97	Average exits:[9.45/17]	Flops:3.59GFlops (58.240%) avg pass time: 0.61s

# python main.py --template VDSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.96 \
# --pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [DIV2K x2] PSNR: 33.516	SSIM: 0.9280	Threshold: 0.96	Average exits:[8.90/17]	Flops:3.41GFlops (55.214%) avg pass time: 0.51s







# python main.py --template VDSR_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 400 --exit_interval 2 --exit_threshold 1 \
# --pre_train /data/shizun/experiment/20220301_VDSR_decision_de3_x2_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# # [TEST8K x2] PSNR: 39.536	SSIM: 0.9636	Threshold: 1.0	Average exits:[7.97/8]	Flops:6.15GFlops (99.695%) avg pass time: 6.80s

# python main.py --template VDSR_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 400 --exit_interval 2 --exit_threshold 0.98 \
# --pre_train /data/shizun/experiment/20220301_VDSR_decision_de3_x2_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# # [TEST8K x2] PSNR: 39.513	SSIM: 0.9636	Threshold: 0.98	Average exits:[6.00/8]	Flops:4.81GFlops (77.961%) avg pass time: 6.64s

# python main.py --template VDSR_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 500 --exit_interval 2 --exit_threshold 0.96 \
# --pre_train /data/shizun/experiment/20220301_VDSR_decision_de3_x2_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# # [TEST8K x2] PSNR: 39.501	SSIM: 0.9635	Threshold: 0.96	Average exits:[5.10/8]	Flops:4.19GFlops (67.928%) avg pass time: 6.10s

# python main.py --template VDSR_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 500 --exit_interval 2 --exit_threshold 0.94 \
# --pre_train /data/shizun/experiment/20220301_VDSR_decision_de3_x2_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# # [TEST8K x2] PSNR: 39.492	SSIM: 0.9635	Threshold: 0.94	Average exits:[4.60/8]	Flops:3.85GFlops (62.319%) avg pass time: 5.95s

# python main.py --template VDSR_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 500 --exit_interval 2 --exit_threshold 0.92 \
# --pre_train /data/shizun/experiment/20220301_VDSR_decision_de3_x2_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# # [TEST8K x2] PSNR: 39.482	SSIM: 0.9634	Threshold: 0.92	Average exits:[4.18/8]	Flops:3.56GFlops (57.774%) avg pass time: 5.87s

# python main.py --template VDSR_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 500 --exit_interval 2 --exit_threshold 0.9 \
# --pre_train /data/shizun/experiment/20220301_VDSR_decision_de3_x2_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# # [TEST8K x2] PSNR: 39.472	SSIM: 0.9634	Threshold: 0.9	Average exits:[3.88/8]	Flops:3.36GFlops (54.379%) avg pass time: 6.44s