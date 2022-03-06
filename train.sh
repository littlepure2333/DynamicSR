python main.py --template RCAN_dytest --scale 4 --data_test DIV2K --n_parallel 200 --device 0 --exit_interval 1 --exit_threshold 0.88 \
--pre_train /home/shizun/experiment/20211108_RCAN_decision_x4_e300_ps192_lr0.0001_n10_i1_sum_pretrain_de3/model/model_best.pt

python main.py --template EDSR_dytest --scale 4 --data_test DIV2K --n_parallel 200 --device 3 --exit_interval 4 --exit_threshold 0.86 \
--pre_train /home/shizun/experiment/20211108_EDSR_decision_x4_e300_ps192_lr0.0001_n32_i4_sTrue_sum_pretrain_de3/model/model_best.pt



# python main.py --template EDSR_test --data_test DIV2K --scale 2 --device 2 \
# --pre_train /home/shizun/experiment/20210929_EDSR_paper_x2_e300_ps192_lr0.0001/model/model_best.pt



# python main.py --template RCAN_test --data_test TEST8K --scale 3 --device 2 \
# --pre_train /home/shizun/experiment/20211105_RCAN_paper_x3_e300_ps192_lr0.0001/model/model_best.pt

# python main.py --template RCAN_test --data_test TEST8K --scale 2 --device 2 \
# --pre_train /home/shizun/experiment/20211105_RCAN_paper_x2_e300_ps192_lr0.0001/model/model_best.pt

# python main.py --template EDSR_test --data_test TEST8K --scale 3 --device 2 \
# --pre_train /home/shizun/experiment/20211023_EDSR_paper_x3_e300_ps192_lr0.0001/model/model_best.pt

# python main.py --template EDSR_test --data_test TEST8K --scale 2 --device 2 \
# --pre_train /home/shizun/experiment/20210929_EDSR_paper_x2_e300_ps192_lr0.0001/model/model_best.pt

# python main.py --template RCAN_test --data_test TEST8K --scale 4 --device 0 \
# --pre_train /home/shizun/experiment/20211105_RCAN_paper_x4_e300_ps192_lr0.0001/model/model_best.pt

# python main.py --template RCAN_test --data_test DIV2K --scale 2 --device 0 \
# --pre_train /home/shizun/experiment/20211105_RCAN_paper_x2_e300_ps192_lr0.0001/model/model_best.pt

# python main.py --template RCAN_test --data_test DIV2K --scale 3 --device 0 \
# --pre_train /home/shizun/experiment/20211105_RCAN_paper_x3_e300_ps192_lr0.0001/model/model_best.pt

python main.py --template RCAN_test --data_test DIV2K --n_parallel 200 --scale 4 --device 0 \
--pre_train /home/shizun/experiment/20211105_RCAN_paper_x4_e300_ps192_lr0.0001/model/model_best.pt

# python main.py --template EDSR_test --data_test TEST8K --scale 4 --device 3 \
# --pre_train /home/shizun/experiment/20211023_EDSR_paper_x4_e300_ps192_lr0.0001/model/model_best.pt

# python main.py --template EDSR_test --data_test DIV2K --scale 2 --device 3 \
# --pre_train /home/shizun/experiment/20210929_EDSR_paper_x2_e300_ps192_lr0.0001/model/model_best.pt

# python main.py --template EDSR_test --data_test DIV2K --scale 3 --device 3 \
# --pre_train /home/shizun/experiment/20211023_EDSR_paper_x3_e300_ps192_lr0.0001/model/model_best.pt

python main.py --template EDSR_test --data_test DIV2K --n_parallel 200 --scale 4 --device 3 \
--pre_train /home/shizun/experiment/20211023_EDSR_paper_x4_e300_ps192_lr0.0001/model/model_best.pt


# python main.py --template EDSR_dytest --scale 3 --data_test DIV2K --device 0 --exit_interval 1 --exit_threshold 0.995 \
# --pre_train /home/shizun/experiment/20211104_EDSR_decision_x3_e300_ps192_lr0.0001_n32_i1_sTrue_sum_pretrain_de3/model/model_best.pt

# python main.py --template EDSR_dytest --scale 3 --data_test DIV2K --device 0 --exit_interval 1 --exit_threshold 0.999 \
# --pre_train /home/shizun/experiment/20211104_EDSR_decision_x3_e300_ps192_lr0.0001_n32_i1_sTrue_sum_pretrain_de3/model/model_best.pt

# python main.py --template EDSR_dytest --scale 3 --data_test DIV2K --device 0 --exit_interval 1 --exit_threshold 0.9999 \
# --pre_train /home/shizun/experiment/20211104_EDSR_decision_x3_e300_ps192_lr0.0001_n32_i1_sTrue_sum_pretrain_de3/model/model_best.pt

# python main.py --template EDSR_dytest --scale 3 --data_test DIV2K --device 0 --exit_interval 1 --exit_threshold 1 \
# --pre_train /home/shizun/experiment/20211104_EDSR_decision_x3_e300_ps192_lr0.0001_n32_i1_sTrue_sum_pretrain_de3/model/model_best.pt
