device=3


# python main.py --template VDSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 2 --exit_threshold 1 \
# --pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x4_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# # [DIV2K x4] PSNR: 28.097	SSIM: 0.7991	Threshold: 1.0	Average exits:[7.95/8]	Flops:24.53GFlops (99.394%) avg pass time: 0.42s

# python main.py --template VDSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 2 --exit_threshold 0.99 \
# --pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x4_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# # [DIV2K x4] PSNR: 28.077	SSIM: 0.7983	Threshold: 0.99	Average exits:[5.99/8]	Flops:19.19GFlops (77.775%) avg pass time: 0.35s

# python main.py --template VDSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 2 --exit_threshold 0.98 \
# --pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x4_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# # [DIV2K x4] PSNR: 28.072	SSIM: 0.7981	Threshold: 0.98	Average exits:[5.51/8]	Flops:17.88GFlops (72.450%) avg pass time: 0.29s

# python main.py --template VDSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 2 --exit_threshold 0.97 \
# --pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x4_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# # [DIV2K x4] PSNR: 28.066	SSIM: 0.7979	Threshold: 0.97	Average exits:[5.13/8]	Flops:16.84GFlops (68.252%) avg pass time: 0.26s

# python main.py --template VDSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 2 --exit_threshold 0.96 \
# --pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x4_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# # [DIV2K x4] PSNR: 28.060	SSIM: 0.7977	Threshold: 0.96	Average exits:[4.81/8]	Flops:15.98GFlops (64.767%) avg pass time: 0.26s



# python main.py --template VDSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 1 \
# --pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x4_e300_ps192_lr0.0001_n20_i1/model/model_best.pt
# # [DIV2K x4] PSNR: 27.957	SSIM: 0.7947	Threshold: 1.0	Average exits:[16.36/17]	Flops:23.81GFlops (96.466%) avg pass time: 0.34s

# python main.py --template VDSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.99 \
# --pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x4_e300_ps192_lr0.0001_n20_i1/model/model_best.pt

# python main.py --template VDSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.98 \
# --pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x4_e300_ps192_lr0.0001_n20_i1/model/model_best.pt

# python main.py --template VDSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.97 \
# --pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x4_e300_ps192_lr0.0001_n20_i1/model/model_best.pt

# python main.py --template VDSR_dytest --scale 4 --data_test DIV2K --device $device \
# --n_parallel 500 --exit_interval 1 --exit_threshold 0.96 \
# --pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x4_e300_ps192_lr0.0001_n20_i1/model/model_best.pt

python main.py --template VDSR_dytest --scale 4 --data_test TEST8K --device $device \
--n_parallel 500 --exit_interval 2 --exit_threshold 1 \
--pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x4_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# [TEST8K x4] PSNR: 32.826	SSIM: 0.8689	Threshold: 1.0	Average exits:[7.95/8]	Flops:24.53GFlops (99.407%) avg pass time: 5.36s

python main.py --template VDSR_dytest --scale 4 --data_test TEST8K --device $device \
--n_parallel 500 --exit_interval 2 --exit_threshold 0.98 \
--pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x4_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# [TEST8K x4] PSNR: 32.807	SSIM: 0.8683	Threshold: 0.98	Average exits:[5.97/8]	Flops:19.15GFlops (77.593%) avg pass time: 4.43s

python main.py --template VDSR_dytest --scale 4 --data_test TEST8K --device $device \
--n_parallel 500 --exit_interval 2 --exit_threshold 0.96 \
--pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x4_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# [TEST8K x4] PSNR: 32.792	SSIM: 0.8680	Threshold: 0.96	Average exits:[5.05/8]	Flops:16.63GFlops (67.399%) avg pass time: 4.06s

python main.py --template VDSR_dytest --scale 4 --data_test TEST8K --device $device \
--n_parallel 500 --exit_interval 2 --exit_threshold 0.94 \
--pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x4_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# [TEST8K x4] PSNR: 32.774	SSIM: 0.8674	Threshold: 0.94	Average exits:[4.32/8]	Flops:14.65GFlops (59.367%) avg pass time: 3.69s

python main.py --template VDSR_dytest --scale 4 --data_test TEST8K --device $device \
--n_parallel 500 --exit_interval 2 --exit_threshold 0.92 \
--pre_train /data/shizun/experiment/20220228_VDSR_decision_de3_x4_e300_ps192_lr0.0001_n20_i2/model/model_best.pt
# [TEST8K x4] PSNR: 32.751	SSIM: 0.8668	Threshold: 0.92	Average exits:[3.81/8]	Flops:13.25GFlops (53.678%) avg pass time: 3.44s