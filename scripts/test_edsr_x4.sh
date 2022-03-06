device=3


# python main.py --template EDSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 1 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x4_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x4] PSNR: 28.783	SSIM: 0.8158	Threshold: 1.0	Average exits:[7.00/7]	Flops:115.79GFlops (99.968%) avg pass time: 0.81s

# python main.py --template EDSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.99 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x4_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x4] PSNR: 28.763	SSIM: 0.8152	Threshold: 0.99	Average exits:[6.00/7]	Flops:104.96GFlops (90.615%) avg pass time: 0.72s

# python main.py --template EDSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.98 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x4_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x4] PSNR: 28.755	SSIM: 0.8149	Threshold: 0.98	Average exits:[5.67/7]	Flops:101.34GFlops (87.488%) avg pass time: 0.70s

# python main.py --template EDSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.97 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x4_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x4] PSNR: 28.747	SSIM: 0.8146	Threshold: 0.97	Average exits:[5.40/7]	Flops:98.42GFlops (84.966%) avg pass time: 0.69s

# python main.py --template EDSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.96 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x4_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x4] PSNR: 28.737	SSIM: 0.8143	Threshold: 0.96	Average exits:[5.11/7]	Flops:95.32GFlops (82.294%) avg pass time: 0.65s

# python main.py --template EDSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.92 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x4_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x4] PSNR: 28.692	SSIM: 0.8128	Threshold: 0.92	Average exits:[4.05/7]	Flops:83.72GFlops (72.280%) avg pass time: 0.60s

# python main.py --template EDSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.91 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x4_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x4] PSNR: 28.680	SSIM: 0.8124	Threshold: 0.91	Average exits:[3.80/7]	Flops:81.00GFlops (69.933%) avg pass time: 0.58s

# python main.py --template EDSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.9 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x4_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x4] PSNR: 28.668	SSIM: 0.8120	Threshold: 0.9	Average exits:[3.58/7]	Flops:78.68GFlops (67.931%) avg pass time: 0.56s

# python main.py --template EDSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.89 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x4_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x4] PSNR: 28.656	SSIM: 0.8116	Threshold: 0.89	Average exits:[3.38/7]	Flops:76.42GFlops (65.978%) avg pass time: 0.60s

# python main.py --template EDSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.88 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x4_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x4] PSNR: 28.644	SSIM: 0.8112	Threshold: 0.88	Average exits:[3.19/7]	Flops:74.41GFlops (64.244%) avg pass time: 0.57s

# python main.py --template EDSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.85 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x4_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x4] PSNR: 28.610	SSIM: 0.8102	Threshold: 0.85	Average exits:[2.71/7]	Flops:69.20GFlops (59.743%) avg pass time: 0.52s

# python main.py --template EDSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.8 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x4_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x4] PSNR: 28.558	SSIM: 0.8087	Threshold: 0.8	Average exits:[2.12/7]	Flops:62.72GFlops (54.146%) avg pass time: 0.49s

# python main.py --template EDSR_dytest --scale 4 --data_test TEST8K --device $device \
# --n_parallel 100 --exit_interval 4 --exit_threshold 1 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x4_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [TEST8K x4] PSNR: 33.389	SSIM: 0.8779	Threshold: 1.0	Average exits:[7.00/7]	Flops:115.81GFlops (99.982%) avg pass time: 11.56s

# python main.py --template EDSR_dytest --scale 4 --data_test TEST8K --device $device \
# --n_parallel 100 --exit_interval 4 --exit_threshold 0.95 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x4_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [TEST8K x4] PSNR: 33.328	SSIM: 0.8764	Threshold: 0.95	Average exits:[4.78/7]	Flops:91.71GFlops (79.175%) avg pass time: 10.36s

python main.py --template EDSR_dytest --scale 4 --data_test TEST8K --device $device \
--n_parallel 100 --exit_interval 4 --exit_threshold 0.94 \
--pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x4_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# [TEST8K x4] PSNR: 33.315	SSIM: 0.8762	Threshold: 0.94	Average exits:[4.47/7]	Flops:88.31GFlops (76.244%) avg pass time: 8.63s

python main.py --template EDSR_dytest --scale 4 --data_test TEST8K --device $device \
--n_parallel 100 --exit_interval 4 --exit_threshold 0.93 \
--pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x4_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# [TEST8K x4] PSNR: 33.301	SSIM: 0.8759	Threshold: 0.93	Average exits:[4.15/7]	Flops:84.89GFlops (73.287%) avg pass time: 8.23s

python main.py --template EDSR_dytest --scale 4 --data_test TEST8K --device $device \
--n_parallel 100 --exit_interval 4 --exit_threshold 0.92 \
--pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x4_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# [TEST8K x4] PSNR: 33.288	SSIM: 0.8756	Threshold: 0.92	Average exits:[3.83/7]	Flops:81.36GFlops (70.237%) avg pass time: 7.99s

python main.py --template EDSR_dytest --scale 4 --data_test TEST8K --device $device \
--n_parallel 100 --exit_interval 4 --exit_threshold 0.91 \
--pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x4_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# [TEST8K x4] PSNR: 33.275	SSIM: 0.8754	Threshold: 0.91	Average exits:[3.55/7]	Flops:78.26GFlops (67.568%) avg pass time: 7.72s

# python main.py --template EDSR_dytest --scale 4 --data_test TEST8K --device $device \
# --n_parallel 100 --exit_interval 4 --exit_threshold 0.9 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x4_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [TEST8K x4] PSNR: 33.264	SSIM: 0.8752	Threshold: 0.9	Average exits:[3.29/7]	Flops:75.53GFlops (65.209%) avg pass time: 8.65s

# python main.py --template EDSR_dytest --scale 4 --data_test TEST8K --device $device \
# --n_parallel 100 --exit_interval 4 --exit_threshold 0.85 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x4_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [TEST8K x4] PSNR: 33.208	SSIM: 0.8741	Threshold: 0.85	Average exits:[2.31/7]	Flops:64.85GFlops (55.989%) avg pass time: 7.73s

# python main.py --template EDSR_dytest --scale 4 --data_test TEST8K --device $device \
# --n_parallel 100 --exit_interval 4 --exit_threshold 0.8 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x4_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [TEST8K x4] PSNR: 33.162	SSIM: 0.8732	Threshold: 0.8	Average exits:[1.69/7]	Flops:58.13GFlops (50.188%) avg pass time: 7.30s
