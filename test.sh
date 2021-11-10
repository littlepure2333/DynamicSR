python main.py --template RCAN_dytest --scale 4 --data_test TEST8K --device 0 --exit_interval 1 --exit_threshold 0.86 \
--pre_train /home/shizun/experiment/20211108_RCAN_decision_x4_e300_ps192_lr0.0001_n10_i1_sum_pretrain_de3/model/model_best.pt

python main.py --template RCAN_dytest --scale 4 --data_test TEST8K --device 0 --exit_interval 1 --exit_threshold 0.88 \
--pre_train /home/shizun/experiment/20211108_RCAN_decision_x4_e300_ps192_lr0.0001_n10_i1_sum_pretrain_de3/model/model_best.pt

python main.py --template RCAN_dytest --scale 4 --data_test TEST8K --device 0 --exit_interval 1 --exit_threshold 0.90 \
--pre_train /home/shizun/experiment/20211108_RCAN_decision_x4_e300_ps192_lr0.0001_n10_i1_sum_pretrain_de3/model/model_best.pt

python main.py --template RCAN_dytest --scale 4 --data_test TEST8K --device 0 --exit_interval 1 --exit_threshold 0.92 \
--pre_train /home/shizun/experiment/20211108_RCAN_decision_x4_e300_ps192_lr0.0001_n10_i1_sum_pretrain_de3/model/model_best.pt

python main.py --template RCAN_dytest --scale 4 --data_test TEST8K --device 0 --exit_interval 1 --exit_threshold 0.94 \
--pre_train /home/shizun/experiment/20211108_RCAN_decision_x4_e300_ps192_lr0.0001_n10_i1_sum_pretrain_de3/model/model_best.pt






# python main.py --template EDSR_dytest --data_test TEST8K --scale 4 --device 2 --exit_interval 4 --exit_threshold 0.8 \
# --pre_train /home/shizun/experiment/20211108_EDSR_decision_x4_e300_ps192_lr0.0001_n32_i4_sTrue_sum_pretrain_de3/model/model_best.pt

# python main.py --template EDSR_dytest --data_test TEST8K --scale 4 --device 2 --exit_interval 2 --exit_threshold 0.9 \
# --pre_train /home/shizun/experiment/20211108_EDSR_decision_x4_e300_ps192_lr0.0001_n32_i2_sTrue_sum_pretrain_de3/model/model_best.pt

# python main.py --template EDSR_dytest --data_test TEST8K --scale 4 --device 2 --exit_interval 1 --exit_threshold 0.99 \
# --pre_train /home/shizun/experiment/20211104_EDSR_decision_x4_e300_ps192_lr0.0001_n32_i1_sTrue_sum_pretrain_de3/model/model_best.pt
