python main.py --template RCAN_test --data_test TEST8K --scale 4 --device 0 \
--pre_train /home/shizun/experiment/20211105_RCAN_paper_x4_e300_ps192_lr0.0001/model/model_best.pt

python main.py --template RCAN_test --data_test DIV2K --scale 2 --device 0 \
--pre_train /home/shizun/experiment/20211105_RCAN_paper_x2_e300_ps192_lr0.0001/model/model_best.pt

python main.py --template RCAN_test --data_test DIV2K --scale 3 --device 0 \
--pre_train /home/shizun/experiment/20211105_RCAN_paper_x3_e300_ps192_lr0.0001/model/model_best.pt

python main.py --template RCAN_test --data_test DIV2K --scale 4 --device 0 \
--pre_train /home/shizun/experiment/20211105_RCAN_paper_x4_e300_ps192_lr0.0001/model/model_best.pt

# python main.py --template EDSR_test --data_test TEST8K --scale 4 --device 3 \
# --pre_train /home/shizun/experiment/20211023_EDSR_paper_x4_e300_ps192_lr0.0001/model/model_best.pt

# python main.py --template EDSR_test --data_test DIV2K --scale 2 --device 3 \
# --pre_train /home/shizun/experiment/20210929_EDSR_paper_x2_e300_ps192_lr0.0001/model/model_best.pt

# python main.py --template EDSR_test --data_test DIV2K --scale 3 --device 3 \
# --pre_train /home/shizun/experiment/20211023_EDSR_paper_x3_e300_ps192_lr0.0001/model/model_best.pt

# python main.py --template EDSR_test --data_test DIV2K --scale 4 --device 3 \
# --pre_train /home/shizun/experiment/20211023_EDSR_paper_x4_e300_ps192_lr0.0001/model/model_best.pt

