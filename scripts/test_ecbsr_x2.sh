device=3


# python main.py --template ECBSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 1 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x2_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [DIV2K x2] PSNR: 33.871	SSIM: 0.9316	Threshold: 1.0	Average exits:[14.89/15]	Flops:1.37GFlops (99.345%) avg pass time: 0.22s

# python main.py --template ECBSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.99 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x2_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [DIV2K x2] PSNR: 33.841	SSIM: 0.9313	Threshold: 0.99	Average exits:[12.03/15]	Flops:1.13GFlops (81.877%) avg pass time: 0.24s

# python main.py --template ECBSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.98 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x2_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [DIV2K x2] PSNR: 33.818	SSIM: 0.9310	Threshold: 0.98	Average exits:[10.63/15]	Flops:1.01GFlops (73.246%) avg pass time: 0.28s

# python main.py --template ECBSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.97 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x2_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [DIV2K x2] PSNR: 33.788	SSIM: 0.9307	Threshold: 0.97	Average exits:[9.39/15]	Flops:0.91GFlops (65.650%) avg pass time: 0.30s

# python main.py --template ECBSR_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.96 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x2_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [DIV2K x2] PSNR: 33.750	SSIM: 0.9303	Threshold: 0.96	Average exits:[7.63/15]	Flops:0.76GFlops (54.877%) avg pass time: 0.24s







# python main.py --template ECBSR_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 400 --exit_interval 1 --exit_threshold 1 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x2_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# [TEST8K x2] PSNR: 39.732	SSIM: 0.9646	Threshold: 1.0	Average exits:[14.94/15]	Flops:1.37GFlops (99.625%) avg pass time: 6.83s

# python main.py --template ECBSR_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.98 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x2_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [TEST8K x2] PSNR: 39.699	SSIM: 0.9643	Threshold: 0.98	Average exits:[11.21/15]	Flops:1.06GFlops (76.824%) avg pass time: 7.92s

# python main.py --template ECBSR_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.96 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x2_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [TEST8K x2] PSNR: 39.632	SSIM: 0.9639	Threshold: 0.96	Average exits:[7.40/15]	Flops:0.74GFlops (53.485%) avg pass time: 5.07s

# python main.py --template ECBSR_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.94 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x2_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [TEST8K x2] PSNR: 39.566	SSIM: 0.9636	Threshold: 0.94	Average exits:[4.91/15]	Flops:0.52GFlops (38.019%) avg pass time: 5.45s

# python main.py --template ECBSR_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.92 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x2_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [TEST8K x2] PSNR: 39.530	SSIM: 0.9634	Threshold: 0.92	Average exits:[3.94/15]	Flops:0.44GFlops (32.074%) avg pass time: 5.37s

# python main.py --template ECBSR_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.9 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x2_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [TEST8K x2] PSNR: 39.506	SSIM: 0.9633	Threshold: 0.9	Average exits:[3.20/15]	Flops:0.38GFlops (27.425%) avg pass time: 5.06s