device=2


# python main.py --template ECBSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 1 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x3_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [DIV2K x3] PSNR: 30.156	SSIM: 0.8618	Threshold: 1.0	Average exits:[14.95/15]	Flops:1.40GFlops (99.688%) avg pass time: 0.16s

# python main.py --template ECBSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.99 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x3_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [DIV2K x3] PSNR: 30.134	SSIM: 0.8613	Threshold: 0.99	Average exits:[12.01/15]	Flops:1.15GFlops (81.912%) avg pass time: 0.18s

# python main.py --template ECBSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.98 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x3_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [DIV2K x3] PSNR: 30.119	SSIM: 0.8609	Threshold: 0.98	Average exits:[10.84/15]	Flops:1.05GFlops (74.840%) avg pass time: 0.17s

# python main.py --template ECBSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.97 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x3_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [DIV2K x3] PSNR: 30.100	SSIM: 0.8604	Threshold: 0.97	Average exits:[9.52/15]	Flops:0.94GFlops (66.900%) avg pass time: 0.17s

# python main.py --template ECBSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.96 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x3_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [DIV2K x3] PSNR: 30.077	SSIM: 0.8597	Threshold: 0.96	Average exits:[8.34/15]	Flops:0.84GFlops (59.735%) avg pass time: 0.16s








# python main.py --template ECBSR_dytest --scale 3 --data_test TEST8K --device $device \
# --n_parallel 1500 --exit_interval 1 --exit_threshold 1 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x3_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [TEST8K x3] PSNR: 35.314	SSIM: 0.9159	Threshold: 1.0	Average exits:[14.97/15]	Flops:1.40GFlops (99.821%) avg pass time: 2.39s

# python main.py --template ECBSR_dytest --scale 3 --data_test TEST8K --device $device \
# --n_parallel 1500 --exit_interval 1 --exit_threshold 0.98 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x3_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [TEST8K x3] PSNR: 35.303	SSIM: 0.9154	Threshold: 0.98	Average exits:[11.65/15]	Flops:1.12GFlops (79.766%) avg pass time: 2.34s

# python main.py --template ECBSR_dytest --scale 3 --data_test TEST8K --device $device \
# --n_parallel 1500 --exit_interval 1 --exit_threshold 0.96 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x3_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [TEST8K x3] PSNR: 35.268	SSIM: 0.9147	Threshold: 0.96	Average exits:[8.34/15]	Flops:0.84GFlops (59.757%) avg pass time: 2.23s

# python main.py --template ECBSR_dytest --scale 3 --data_test TEST8K --device $device \
# --n_parallel 1500 --exit_interval 1 --exit_threshold 0.94 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x3_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [TEST8K x3] PSNR: 35.219	SSIM: 0.9138	Threshold: 0.94	Average exits:[5.95/15]	Flops:0.63GFlops (45.318%) avg pass time: 2.36s

# python main.py --template ECBSR_dytest --scale 3 --data_test TEST8K --device $device \
# --n_parallel 1500 --exit_interval 1 --exit_threshold 0.92 \
# --pre_train /data/shizun/experiment/20220228_ECBSR_decision_de3_x3_e300_ps192_lr0.0001_n16_i1/model/model_best.pt
# # [TEST8K x3] PSNR: 35.165	SSIM: 0.9130	Threshold: 0.92	Average exits:[4.50/15]	Flops:0.51GFlops (36.436%) avg pass time: 2.42s