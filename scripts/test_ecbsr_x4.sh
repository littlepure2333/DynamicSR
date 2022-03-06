device=3


# python main.py --template ECBSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 1 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x4_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# [DIV2K x4] PSNR: 28.311	SSIM: 0.8036	Threshold: 1.0	Average exits:[14.99/15]	Flops:1.43GFlops (99.956%) avg pass time: 0.14s

# python main.py --template ECBSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.99 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x4_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [DIV2K x4] PSNR: 28.255	SSIM: 0.8020	Threshold: 0.99	Average exits:[10.67/15]	Flops:1.06GFlops (74.158%) avg pass time: 0.15s

# python main.py --template ECBSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.98 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x4_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [DIV2K x4] PSNR: 28.224	SSIM: 0.8011	Threshold: 0.98	Average exits:[9.04/15]	Flops:0.92GFlops (64.380%) avg pass time: 0.14s

# python main.py --template ECBSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.97 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x4_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [DIV2K x4] PSNR: 28.197	SSIM: 0.8003	Threshold: 0.97	Average exits:[7.57/15]	Flops:0.80GFlops (55.683%) avg pass time: 0.17s

# python main.py --template ECBSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.96 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x4_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# [DIV2K x4] PSNR: 28.171	SSIM: 0.7995	Threshold: 0.96	Average exits:[6.34/15]	Flops:0.69GFlops (48.387%) avg pass time: 0.13s

# python main.py --template ECBSR_dytest --scale 4 --data_test TEST8K --device $device \
# --n_parallel 1500 --exit_interval 1 --exit_threshold 1 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x4_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [TEST8K x4] PSNR: 33.053	SSIM: 0.8719	Threshold: 1.0	Average exits:[15.00/15]	Flops:1.43GFlops (99.985%) avg pass time: 1.74s

# python main.py --template ECBSR_dytest --scale 4 --data_test TEST8K --device $device \
# --n_parallel 1500 --exit_interval 1 --exit_threshold 0.98 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x4_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [TEST8K x4] PSNR: 32.990	SSIM: 0.8706	Threshold: 0.98	Average exits:[10.47/15]	Flops:1.04GFlops (72.841%) avg pass time: 1.61s

# python main.py --template ECBSR_dytest --scale 4 --data_test TEST8K --device $device \
# --n_parallel 1500 --exit_interval 1 --exit_threshold 0.96 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x4_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [TEST8K x4] PSNR: 32.933	SSIM: 0.8694	Threshold: 0.96	Average exits:[6.80/15]	Flops:0.73GFlops (51.116%) avg pass time: 1.58s

# python main.py --template ECBSR_dytest --scale 4 --data_test TEST8K --device $device \
# --n_parallel 1500 --exit_interval 1 --exit_threshold 0.94 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x4_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [TEST8K x4] PSNR: 32.877	SSIM: 0.8681	Threshold: 0.94	Average exits:[5.02/15]	Flops:0.58GFlops (40.540%) avg pass time: 1.60s

# python main.py --template ECBSR_dytest --scale 4 --data_test TEST8K --device $device \
# --n_parallel 1500 --exit_interval 1 --exit_threshold 0.92 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x4_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [TEST8K x4] PSNR: 32.823	SSIM: 0.8670	Threshold: 0.92	Average exits:[3.53/15]	Flops:0.45GFlops (31.677%) avg pass time: 1.64s