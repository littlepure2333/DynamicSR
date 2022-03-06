device=3


# python main.py --template EDSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 1 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x2_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x2] PSNR: 34.461	SSIM: 0.9366	Threshold: 1.0	Average exits:[7.00/7]	Flops:93.89GFlops (100.000%) avg pass time: 2.05s

# python main.py --template EDSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.99 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x2_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x2] PSNR: 34.425	SSIM: 0.9363	Threshold: 0.99	Average exits:[5.92/7]	Flops:82.16GFlops (87.511%) avg pass time: 1.84s

# python main.py --template EDSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.98 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x2_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x2] PSNR: 34.406	SSIM: 0.9362	Threshold: 0.98	Average exits:[5.30/7]	Flops:75.35GFlops (80.257%) avg pass time: 1.73s

# python main.py --template EDSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.97 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x2_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x2] PSNR: 34.392	SSIM: 0.9360	Threshold: 0.97	Average exits:[4.86/7]	Flops:70.57GFlops (75.167%) avg pass time: 1.68s

# python main.py --template EDSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.96 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x2_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x2] PSNR: 34.380	SSIM: 0.9359	Threshold: 0.96	Average exits:[4.52/7]	Flops:66.92GFlops (71.279%) avg pass time: 1.59s

# python main.py --template EDSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.9 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x2_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x2] PSNR: 34.305	SSIM: 0.9351	Threshold: 0.9	Average exits:[3.09/7]	Flops:51.39GFlops (54.740%) avg pass time: 1.27s

# python main.py --template EDSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.85 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x2_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x2] PSNR: 34.246	SSIM: 0.9345	Threshold: 0.85	Average exits:[2.43/7]	Flops:44.15GFlops (47.024%) avg pass time: 1.06s

# python main.py --template EDSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.84 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x2_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x2] PSNR: 34.235	SSIM: 0.9344	Threshold: 0.84	Average exits:[2.33/7]	Flops:43.06GFlops (45.863%) avg pass time: 1.05s

# python main.py --template EDSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.83 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x2_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x2] PSNR: 34.224	SSIM: 0.9343	Threshold: 0.83	Average exits:[2.22/7]	Flops:41.95GFlops (44.682%) avg pass time: 1.07s

# python main.py --template EDSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.82 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x2_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x2] PSNR: 34.214	SSIM: 0.9342	Threshold: 0.82	Average exits:[2.13/7]	Flops:40.95GFlops (43.612%) avg pass time: 1.08s

# python main.py --template EDSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.81 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x2_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x2] PSNR: 34.204	SSIM: 0.9341	Threshold: 0.81	Average exits:[2.05/7]	Flops:40.04GFlops (42.645%) avg pass time: 1.09s

# python main.py --template EDSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.8 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x2_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x2] PSNR: 34.195	SSIM: 0.9340	Threshold: 0.8	Average exits:[1.96/7]	Flops:39.09GFlops (41.630%) avg pass time: 1.08s





# python main.py --template EDSR_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 400 --exit_interval 4 --exit_threshold 1 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x2_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [TEST8K x2] PSNR: 40.162	SSIM: 0.9662	Threshold: 1.0	Average exits:[7.00/7]	Flops:93.89GFlops (100.000%) avg pass time: 33.23s


# python main.py --template EDSR_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.95 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x2_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [TEST8K x2] PSNR: 40.087	SSIM: 0.9659	Threshold: 0.95	Average exits:[4.35/7]	Flops:65.06GFlops (69.297%) avg pass time: 23.39s

# python main.py --template EDSR_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.9 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x2_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [TEST8K x2] PSNR: 40.025	SSIM: 0.9655	Threshold: 0.9	Average exits:[3.01/7]	Flops:50.47GFlops (53.751%) avg pass time: 19.64s

# python main.py --template EDSR_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.88 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x2_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [TEST8K x2] PSNR: 39.999	SSIM: 0.9654	Threshold: 0.88	Average exits:[2.64/7]	Flops:46.42GFlops (49.445%) avg pass time: 20.21s

# python main.py --template EDSR_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.86 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x2_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [TEST8K x2] PSNR: 39.976	SSIM: 0.9653	Threshold: 0.86	Average exits:[2.36/7]	Flops:43.38GFlops (46.206%) avg pass time: 18.20s

# python main.py --template EDSR_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.85 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x2_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [TEST8K x2] PSNR: 39.966	SSIM: 0.9653	Threshold: 0.85	Average exits:[2.24/7]	Flops:42.07GFlops (44.811%) avg pass time: 15.91s

# python main.py --template EDSR_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.83 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x2_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [TEST8K x2] PSNR: 39.946	SSIM: 0.9652	Threshold: 0.83	Average exits:[2.00/7]	Flops:39.55GFlops (42.123%) avg pass time: 15.04s

# python main.py --template EDSR_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.81 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x2_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [TEST8K x2] PSNR: 39.928	SSIM: 0.9651	Threshold: 0.81	Average exits:[1.82/7]	Flops:37.53GFlops (39.973%) avg pass time: 13.85s

# python main.py --template EDSR_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.8 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x2_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [TEST8K x2] PSNR: 39.919	SSIM: 0.9651	Threshold: 0.8	Average exits:[1.74/7]	Flops:36.68GFlops (39.069%) avg pass time: 13.63s