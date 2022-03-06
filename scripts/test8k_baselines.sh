# device="0,1,2,3"
# n_GPUs=4
device="1"
n_GPUs=1


# python main.py --template EDSR --scale 2 --data_test TEST8K --device $device --n_GPUs $n_GPUs \
# --pre_train "/home/shizun/experiment/20210929_EDSR_paper_x2_e300_ps192_lr0.0001/model/model_best.pt"
# # [TEST8K x2]	PSNR: 39.971	SSIM: 0.9656 (Best: 39.971 @epoch 1) [Average]	 pre:0.0468s	model:10.1119s	post:0.0919s

# python main.py --template EDSR --scale 3 --data_test TEST8K --device $device --n_GPUs $n_GPUs \
# --pre_train "/home/shizun/experiment/20211023_EDSR_paper_x3_e300_ps192_lr0.0001/model/model_best.pt"
# # [TEST8K x3]	PSNR: 35.539	SSIM: 0.9178 (Best: 35.539 @epoch 1) [Average]	 pre:0.0413s	model:4.5355s	post:0.0882s

# python main.py --template EDSR --scale 4 --data_test TEST8K --device $device --n_GPUs $n_GPUs \
# --pre_train "/home/shizun/experiment/20211023_EDSR_paper_x4_e300_ps192_lr0.0001/model/model_best.pt"
# # [TEST8K x4]	PSNR: 33.300	SSIM: 0.8762 (Best: 33.300 @epoch 1) [Average]	 pre:0.0394s	model:3.1198s	post:0.0884s

# python main.py --template RCAN --scale 2 --data_test TEST8K --device $device --n_GPUs $n_GPUs \
# --pre_train "/home/shizun/experiment/20220223_RCAN_my_x2_e300_ps192_lr0.0001/model/model_best.pt"
# # [TEST8K x2]	PSNR: 40.038	SSIM: 0.9657 (Best: 40.038 @epoch 1) [Average]	 pre:0.0470s	model:8.0095s	post:0.0891s

# python main.py --template RCAN --scale 3 --data_test TEST8K --device $device --n_GPUs $n_GPUs \
# --pre_train "/home/shizun/experiment/20220224_RCAN_my_x3_e300_ps192_lr0.0001/model/model_best.pt"
# # [TEST8K x3]	PSNR: 35.440	SSIM: 0.9171 (Best: 35.440 @epoch 1) [Average]	 pre:0.0415s	model:2.8366s	post:0.0883s

# python main.py --template RCAN --scale 4 --data_test TEST8K --device $device --n_GPUs $n_GPUs \
# --pre_train "/home/shizun/experiment/20220222_RCAN_my_x4_e300_ps192_lr0.0001/model/model_best.pt"
# # [TEST8K x4]	PSNR: 33.255	SSIM: 0.8753 (Best: 33.255 @epoch 1) [Average]	 pre:0.0395s	model:2.1818s	post:0.0880s

# python main.py --template ECBSR --scale 2 --data_test TEST8K --device $device --n_GPUs $n_GPUs \
# --pre_train "/home/shizun/experiment/20220228_ECBSR_my_x2_e300_ps192_lr0.0001/model/model_best.pt"
# # [TEST8K x2]	PSNR: 39.827	SSIM: 0.9649 (Best: 39.827 @epoch 1) [Average]	 pre:0.0469s	model:11.7400s	post:0.0900s

# python main.py --template ECBSR --scale 3 --data_test TEST8K --device $device --n_GPUs $n_GPUs \
# --pre_train "/home/shizun/experiment/20220228_ECBSR_my_x3_e300_ps192_lr0.0001/model/model_best.pt"
# # [TEST8K x3]	PSNR: 35.369	SSIM: 0.9158 (Best: 35.369 @epoch 1) [Average]	 pre:0.0535s	model:0.3588s	post:0.1020s

# python main.py --template ECBSR --scale 4 --data_test TEST8K --device $device --n_GPUs $n_GPUs \
# --pre_train "/home/shizun/experiment/20220228_ECBSR_my_x4_e300_ps192_lr0.0001/model/model_best.pt"
# # [TEST8K x4]	PSNR: 33.070	SSIM: 0.8724 (Best: 33.070 @epoch 1) [Average]	 pre:0.0396s	model:3.2533s	post:0.1293s

python main.py --template VDSR --scale 2 --data_test TEST8K --device $device --n_GPUs $n_GPUs \
--pre_train "/home/shizun/experiment/20220223_VDSR_my_x2_e300_ps192_lr0.0001/model/model_best.pt"

python main.py --template VDSR --scale 3 --data_test TEST8K --device $device --n_GPUs $n_GPUs \
--pre_train "/home/shizun/experiment/20220223_VDSR_my_x3_e300_ps192_lr0.0001/model/model_best.pt"

python main.py --template VDSR --scale 4 --data_test TEST8K --device $device --n_GPUs $n_GPUs \
--pre_train "/home/shizun/experiment/20220222_VDSR_my_x4_e300_ps192_lr0.0001/model/model_best.pt"