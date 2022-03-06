device=2


# python main.py --template VDSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 2 --exit_threshold 1 \
# --pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x3_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# # [DIV2K x3] PSNR: 29.974	SSIM: 0.8569	Threshold: 1.0	Average exits:[8.00/8]	Flops:13.88GFlops (100.000%) avg pass time: 0.46s
# # [DIV2K x3] PSNR: 29.924	SSIM: 0.8574	Threshold: 1.0	Average exits:[8.00/8]	Flops:13.88GFlops (100.000%) avg pass time: 0.33s

# python main.py --template VDSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 2 --exit_threshold 0.99 \
# --pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x3_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# # [DIV2K x3] PSNR: 29.948	SSIM: 0.8562	Threshold: 0.99	Average exits:[7.44/8]	Flops:13.02GFlops (93.786%) avg pass time: 0.43s
# # [DIV2K x3] PSNR: 29.899	SSIM: 0.8567	Threshold: 0.99	Average exits:[7.42/8]	Flops:12.99GFlops (93.566%) avg pass time: 0.42s

# python main.py --template VDSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 2 --exit_threshold 0.98 \
# --pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x3_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# # [DIV2K x3] PSNR: 29.928	SSIM: 0.8557	Threshold: 0.98	Average exits:[6.94/8]	Flops:12.25GFlops (88.260%) avg pass time: 0.44s
# # [DIV2K x3] PSNR: 29.879	SSIM: 0.8562	Threshold: 0.98	Average exits:[6.88/8]	Flops:12.16GFlops (87.636%) avg pass time: 0.41s

# python main.py --template VDSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 2 --exit_threshold 0.97 \
# --pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x3_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# # [DIV2K x3] PSNR: 29.906	SSIM: 0.8552	Threshold: 0.97	Average exits:[6.35/8]	Flops:11.34GFlops (81.725%) avg pass time: 0.43s
# # [DIV2K x3] PSNR: 29.857	SSIM: 0.8557	Threshold: 0.97	Average exits:[6.18/8]	Flops:11.09GFlops (79.881%) avg pass time: 0.30s

# python main.py --template VDSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 2 --exit_threshold 0.96 \
# --pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x3_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# # [DIV2K x3] PSNR: 29.882	SSIM: 0.8547	Threshold: 0.96	Average exits:[5.59/8]	Flops:10.19GFlops (73.423%) avg pass time: 0.42s
# # [DIV2K x3] PSNR: 29.832	SSIM: 0.8551	Threshold: 0.96	Average exits:[5.40/8]	Flops:9.90GFlops (71.328%) avg pass time: 0.27s



# python main.py --template VDSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 1 \
# --pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x3_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [DIV2K x3] PSNR: 29.935	SSIM: 0.8552	Threshold: 1.0	Average exits:[16.78/17]	Flops:13.71GFlops (98.764%) avg pass time: 0.47s
# # [DIV2K x3] PSNR: 29.886	SSIM: 0.8557	Threshold: 1.0	Average exits:[16.79/17]	Flops:13.72GFlops (98.841%) avg pass time: 0.34s

# python main.py --template VDSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.99 \
# --pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x3_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [DIV2K x3] PSNR: 29.760	SSIM: 0.8521	Threshold: 0.99	Average exits:[10.75/17]	Flops:9.08GFlops (65.453%) avg pass time: 0.46s

# python main.py --template VDSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.98 \
# --pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x3_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [DIV2K x3] PSNR: 29.729	SSIM: 0.8516	Threshold: 0.98	Average exits:[9.41/17]	Flops:8.06GFlops (58.070%) avg pass time: 0.41s

# python main.py --template VDSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.97 \
# --pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x3_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [DIV2K x3] PSNR: 29.702	SSIM: 0.8510	Threshold: 0.97	Average exits:[8.25/17]	Flops:7.17GFlops (51.627%) avg pass time: 0.45s

# python main.py --template VDSR_dytest --scale 3 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.96 \
# --pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x3_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [DIV2K x3] PSNR: 29.667	SSIM: 0.8504	Threshold: 0.96	Average exits:[7.15/17]	Flops:6.33GFlops (45.579%) avg pass time: 0.36s

python main.py --template VDSR_dytest --scale 3 --data_test TEST8K --device $device \
--n_parallel 500 --exit_interval 2 --exit_threshold 1 \
--pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x3_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# [TEST8K x3] PSNR: 35.157	SSIM: 0.9138	Threshold: 1.0	Average exits:[8.00/8]	Flops:13.88GFlops (100.000%) avg pass time: 5.18s

python main.py --template VDSR_dytest --scale 3 --data_test TEST8K --device $device \
--n_parallel 500 --exit_interval 2 --exit_threshold 0.98 \
--pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x3_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# [TEST8K x3] PSNR: 35.120	SSIM: 0.9131	Threshold: 0.98	Average exits:[7.25/8]	Flops:12.74GFlops (91.763%) avg pass time: 6.45s

python main.py --template VDSR_dytest --scale 3 --data_test TEST8K --device $device \
--n_parallel 500 --exit_interval 2 --exit_threshold 0.96 \
--pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x3_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# [TEST8K x3] PSNR: 35.074	SSIM: 0.9124	Threshold: 0.96	Average exits:[5.49/8]	Flops:10.03GFlops (72.273%) avg pass time: 4.35s

python main.py --template VDSR_dytest --scale 3 --data_test TEST8K --device $device \
--n_parallel 500 --exit_interval 2 --exit_threshold 0.94 \
--pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x3_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# [TEST8K x3] PSNR: 35.030	SSIM: 0.9118	Threshold: 0.94	Average exits:[4.16/8]	Flops:7.99GFlops (57.567%) avg pass time: 3.75s

python main.py --template VDSR_dytest --scale 3 --data_test TEST8K --device $device \
--n_parallel 500 --exit_interval 2 --exit_threshold 0.92 \
--pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x3_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# [TEST8K x3] PSNR: 35.000	SSIM: 0.9113	Threshold: 0.92	Average exits:[3.35/8]	Flops:6.75GFlops (48.600%) avg pass time: 3.44s
