device=2


# python main.py --template EDSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 1 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x3_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x3] PSNR: 30.665	SSIM: 0.8711	Threshold: 1.0	Average exits:[7.00/7]	Flops:100.77GFlops (100.000%) avg pass time: 1.01s

# python main.py --template EDSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.99 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x3_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x3] PSNR: 30.657	SSIM: 0.8709	Threshold: 0.99	Average exits:[6.55/7]	Flops:95.88GFlops (95.150%) avg pass time: 0.98s

# python main.py --template EDSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.98 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x3_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x3] PSNR: 30.647	SSIM: 0.8706	Threshold: 0.98	Average exits:[6.09/7]	Flops:90.85GFlops (90.156%) avg pass time: 0.94s

# python main.py --template EDSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.97 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x3_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x3] PSNR: 30.636	SSIM: 0.8704	Threshold: 0.97	Average exits:[5.62/7]	Flops:85.76GFlops (85.101%) avg pass time: 0.91s

# python main.py --template EDSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.96 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x3_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x3] PSNR: 30.626	SSIM: 0.8701	Threshold: 0.96	Average exits:[5.20/7]	Flops:81.16GFlops (80.541%) avg pass time: 0.81s

# python main.py --template EDSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.92 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x3_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x3] PSNR: 30.577	SSIM: 0.8689	Threshold: 0.92	Average exits:[3.96/7]	Flops:67.73GFlops (67.209%) avg pass time: 0.72s

# python main.py --template EDSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.91 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x3_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x3] PSNR: 30.565	SSIM: 0.8686	Threshold: 0.91	Average exits:[3.74/7]	Flops:65.26GFlops (64.764%) avg pass time: 0.74s

# python main.py --template EDSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.9 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x3_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x3] PSNR: 30.552	SSIM: 0.8683	Threshold: 0.9	Average exits:[3.53/7]	Flops:63.04GFlops (62.554%) avg pass time: 0.72s

# python main.py --template EDSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.89 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x3_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x3] PSNR: 30.540	SSIM: 0.8681	Threshold: 0.89	Average exits:[3.34/7]	Flops:60.98GFlops (60.511%) avg pass time: 0.74s

# python main.py --template EDSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.88 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x3_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x3] PSNR: 30.528	SSIM: 0.8678	Threshold: 0.88	Average exits:[3.17/7]	Flops:59.08GFlops (58.627%) avg pass time: 0.76s

# python main.py --template EDSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.85 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x3_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x3] PSNR: 30.495	SSIM: 0.8670	Threshold: 0.85	Average exits:[2.72/7]	Flops:54.18GFlops (53.767%) avg pass time: 0.70s

# python main.py --template EDSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 4 --exit_threshold 0.8 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x3_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [DIV2K x3] PSNR: 30.495	SSIM: 0.8670	Threshold: 0.85	Average exits:[2.72/7]	Flops:54.18GFlops (53.767%) avg pass time: 0.70s

# python main.py --template EDSR_dytest --scale 3 --data_test TEST8K --device $device \
# --n_parallel 200 --exit_interval 4 --exit_threshold 1 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x3_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [TEST8K x3] PSNR: 35.683	SSIM: 0.9197	Threshold: 1.0	Average exits:[7.00/7]	Flops:100.77GFlops (99.999%) avg pass time: 17.23s

# python main.py --template EDSR_dytest --scale 3 --data_test TEST8K --device $device \
# --n_parallel 200 --exit_interval 4 --exit_threshold 0.95 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x3_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [TEST8K x3] PSNR: 35.623	SSIM: 0.9188	Threshold: 0.95	Average exits:[4.47/7]	Flops:73.21GFlops (72.647%) avg pass time: 14.40s

# python main.py --template EDSR_dytest --scale 3 --data_test TEST8K --device $device \
# --n_parallel 200 --exit_interval 4 --exit_threshold 0.9 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x3_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [TEST8K x3] PSNR: 35.558	SSIM: 0.9180	Threshold: 0.9	Average exits:[3.05/7]	Flops:57.80GFlops (57.357%) avg pass time: 11.98s

python main.py --template EDSR_dytest --scale 3 --data_test TEST8K --device $device \
--n_parallel 200 --exit_interval 4 --exit_threshold 0.89 \
--pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x3_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# [TEST8K x3] PSNR: 35.545	SSIM: 0.9178	Threshold: 0.89	Average exits:[2.84/7]	Flops:55.53GFlops (55.107%) avg pass time: 9.55s

python main.py --template EDSR_dytest --scale 3 --data_test TEST8K --device $device \
--n_parallel 200 --exit_interval 4 --exit_threshold 0.87 \
--pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x3_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# [TEST8K x3] PSNR: 35.522	SSIM: 0.9175	Threshold: 0.87	Average exits:[2.49/7]	Flops:51.70GFlops (51.305%) avg pass time: 9.02s

# python main.py --template EDSR_dytest --scale 3 --data_test TEST8K --device $device \
# --n_parallel 200 --exit_interval 4 --exit_threshold 0.85 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x3_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [TEST8K x3] PSNR: 35.501	SSIM: 0.9173	Threshold: 0.85	Average exits:[2.21/7]	Flops:48.65GFlops (48.276%) avg pass time: 11.17s

python main.py --template EDSR_dytest --scale 3 --data_test TEST8K --device $device \
--n_parallel 200 --exit_interval 4 --exit_threshold 0.83 \
--pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x3_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# [TEST8K x3] PSNR: 35.482	SSIM: 0.9170	Threshold: 0.83	Average exits:[1.97/7]	Flops:46.05GFlops (45.698%) avg pass time: 8.23s

python main.py --template EDSR_dytest --scale 3 --data_test TEST8K --device $device \
--n_parallel 200 --exit_interval 4 --exit_threshold 0.81 \
--pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x3_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# [TEST8K x3] PSNR: 35.464	SSIM: 0.9168	Threshold: 0.81	Average exits:[1.74/7]	Flops:43.60GFlops (43.270%) avg pass time: 7.89s

# python main.py --template EDSR_dytest --scale 3 --data_test TEST8K --device $device \
# --n_parallel 200 --exit_interval 4 --exit_threshold 0.8 \
# --pre_train /data/shizun/experiment/20220301_EDSR_decision_de3_x3_e300_ps192_lr0.0001_n32_i4/model/model_best.pt
# # [TEST8K x3] PSNR: 35.456	SSIM: 0.9167	Threshold: 0.8	Average exits:[1.65/7]	Flops:42.59GFlops (42.263%) avg pass time: 8.13s
