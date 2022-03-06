device=0


# python main.py --template RCAN_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 1 \
# --pre_train /data/shizun/experiment/20220301_RCAN_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [DIV2K x2] PSNR: 34.358	SSIM: 0.9357	Threshold: 1.0	Average exits:[9.00/9]	Flops:35.36GFlops (99.993%) avg pass time: 1.79s

# python main.py --template RCAN_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.99 \
# --pre_train /data/shizun/experiment/20220301_RCAN_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [DIV2K x2] PSNR: 34.340	SSIM: 0.9355	Threshold: 0.99	Average exits:[6.70/9]	Flops:27.33GFlops (77.288%) avg pass time: 1.50s

# python main.py --template RCAN_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.98 \
# --pre_train /data/shizun/experiment/20220301_RCAN_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [DIV2K x2] PSNR: 34.328	SSIM: 0.9353	Threshold: 0.98	Average exits:[5.59/9]	Flops:23.46GFlops (66.352%) avg pass time: 1.27s

# python main.py --template RCAN_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.97 \
# --pre_train /data/shizun/experiment/20220301_RCAN_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [DIV2K x2] PSNR: 34.316	SSIM: 0.9352	Threshold: 0.97	Average exits:[4.90/9]	Flops:21.04GFlops (59.488%) avg pass time: 1.11s

# python main.py --template RCAN_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.96 \
# --pre_train /data/shizun/experiment/20220301_RCAN_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [DIV2K x2] PSNR: 34.305	SSIM: 0.9350	Threshold: 0.96	Average exits:[4.35/9]	Flops:19.11GFlops (54.050%) avg pass time: 1.02s

# python main.py --template RCAN_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.9 \
# --pre_train /data/shizun/experiment/20220301_RCAN_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [DIV2K x2] PSNR: 34.242	SSIM: 0.9343	Threshold: 0.9	Average exits:[2.82/9]	Flops:13.77GFlops (38.935%) avg pass time: 0.92s

# python main.py --template RCAN_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.85 \
# --pre_train /data/shizun/experiment/20220301_RCAN_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [DIV2K x2] PSNR: 34.197	SSIM: 0.9339	Threshold: 0.85	Average exits:[2.18/9]	Flops:11.55GFlops (32.668%) avg pass time: 0.70s

# python main.py --template RCAN_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.8 \
# --pre_train /data/shizun/experiment/20220301_RCAN_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [DIV2K x2] PSNR: 34.162	SSIM: 0.9336	Threshold: 0.8	Average exits:[1.84/9]	Flops:10.35GFlops (29.275%) avg pass time: 0.62s

# python main.py --template RCAN_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.75 \
# --pre_train /data/shizun/experiment/20220301_RCAN_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [DIV2K x2] PSNR: 34.135	SSIM: 0.9334	Threshold: 0.75	Average exits:[1.62/9]	Flops:9.60GFlops (27.154%) avg pass time: 0.75s

# python main.py --template RCAN_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.7 \
# --pre_train /data/shizun/experiment/20220301_RCAN_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [DIV2K x2] PSNR: 34.111	SSIM: 0.9332	Threshold: 0.7	Average exits:[1.48/9]	Flops:9.10GFlops (25.730%) avg pass time: 0.66s

# python main.py --template RCAN_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.65 \
# --pre_train /data/shizun/experiment/20220301_RCAN_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [DIV2K x2] PSNR: 34.091	SSIM: 0.9331	Threshold: 0.65	Average exits:[1.37/9]	Flops:8.71GFlops (24.632%) avg pass time: 0.56s

# python main.py --template RCAN_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.6 \
# --pre_train /data/shizun/experiment/20220301_RCAN_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [DIV2K x2] PSNR: 34.072	SSIM: 0.9329	Threshold: 0.6	Average exits:[1.28/9]	Flops:8.41GFlops (23.781%) avg pass time: 0.54s

# python main.py --template RCAN_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.55 \
# --pre_train /data/shizun/experiment/20220301_RCAN_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [DIV2K x2] PSNR: 34.055	SSIM: 0.9328	Threshold: 0.55	Average exits:[1.21/9]	Flops:8.17GFlops (23.118%) avg pass time: 0.54s

# python main.py --template RCAN_dytest --scale 2 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.5 \
# --pre_train /data/shizun/experiment/20220301_RCAN_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [DIV2K x2] PSNR: 34.040	SSIM: 0.9327	Threshold: 0.5	Average exits:[1.16/9]	Flops:8.00GFlops (22.626%) avg pass time: 0.53s







# python main.py --template RCAN_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 400 --exit_interval 1 --exit_threshold 1 \
# --pre_train /data/shizun/experiment/20220301_RCAN_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [TEST8K x2] PSNR: 40.225	SSIM: 0.9663	Threshold: 1.0	Average exits:[9.00/9]	Flops:35.36GFlops (99.996%) avg pass time: 28.09s

# python main.py --template RCAN_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.96 \
# --pre_train /data/shizun/experiment/20220301_RCAN_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [TEST8K x2] PSNR: 40.159	SSIM: 0.9659	Threshold: 0.96	Average exits:[3.76/9]	Flops:17.05GFlops (48.231%) avg pass time: 17.05s

# python main.py --template RCAN_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.93 \
# --pre_train /data/shizun/experiment/20220301_RCAN_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [TEST8K x2] PSNR: 40.115	SSIM: 0.9657	Threshold: 0.93	Average exits:[2.69/9]	Flops:13.32GFlops (37.674%) avg pass time: 14.78s

# python main.py --template RCAN_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.9 \
# --pre_train /data/shizun/experiment/20220301_RCAN_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [TEST8K x2] PSNR: 40.078	SSIM: 0.9656	Threshold: 0.9	Average exits:[2.08/9]	Flops:11.20GFlops (31.669%) avg pass time: 12.01s

# python main.py --template RCAN_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.86 \
# --pre_train /data/shizun/experiment/20220301_RCAN_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [TEST8K x2] PSNR: 40.037	SSIM: 0.9654	Threshold: 0.86	Average exits:[1.63/9]	Flops:9.64GFlops (27.271%) avg pass time: 11.18s

# python main.py --template RCAN_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.83 \
# --pre_train /data/shizun/experiment/20220301_RCAN_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [TEST8K x2] PSNR: 40.011	SSIM: 0.9653	Threshold: 0.83	Average exits:[1.44/9]	Flops:8.95GFlops (25.318%) avg pass time: 9.23s

# python main.py --template RCAN_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.8 \
# --pre_train /data/shizun/experiment/20220301_RCAN_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [TEST8K x2] PSNR: 39.990	SSIM: 0.9652	Threshold: 0.8	Average exits:[1.32/9]	Flops:8.55GFlops (24.188%) avg pass time: 10.16s

# python main.py --template RCAN_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.7 \
# --pre_train /data/shizun/experiment/20220301_RCAN_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [TEST8K x2] PSNR: 39.946	SSIM: 0.9651	Threshold: 0.7	Average exits:[1.13/9]	Flops:7.88GFlops (22.293%) avg pass time: 9.86s

# python main.py --template RCAN_dytest --scale 2 --data_test TEST8K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.6 \
# --pre_train /data/shizun/experiment/20220301_RCAN_decision_de3_x2_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [TEST8K x2] PSNR: 39.923	SSIM: 0.9650	Threshold: 0.6	Average exits:[1.06/9]	Flops:7.64GFlops (21.599%) avg pass time: 9.40s