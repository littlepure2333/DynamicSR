CUDA_VISIBLE_DEVICES=0 python main.py --model adaedsr --patch_size 96 --save 20200613_adaedsr_baseline_x2 --ext sep --n_GPUs 1 --n_colors 3 --test_only --resume -2 --loss 1*L1