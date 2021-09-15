import os
import torch
import torch.nn as nn


def main():
    baseline_path = 'result/espcn_baseline_x2_rk/model/model_best.pt'
    cond_path = 'result/espcn_cond_x2_rk/model/model_best.pt'
    output_path = 'result/espcn_cond_x2_rk_init.pt'
    baseline_ckpt = torch.load(baseline_path)
    cond_ckpt = torch.load(cond_path)
    for k, v in baseline_ckpt.items():
        v1, v2 = v, cond_ckpt[k]
        v2 = v1.flatten().repeat(v2.shape[-1], 1).permute(1, 0)
        print(v1.shape, v2.shape)

    print(output_path)
    torch.save(cond_ckpt, output_path)


if __name__ == '__main__':
    main()
