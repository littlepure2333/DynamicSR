# AugSampling

This repository is an official PyTorch implementation of the paper **"SamplingAug: On the Importance of Patch Sampling Augmentation for Single Image Super-Resolution"** .

## Dependencies
* Python 3.6
* PyTorch >= 1.0.0
* numpy
* skimage
* **imageio**
* matplotlib
* tqdm
* cv2 >= 3.xx (Only if you want to use video input/output)

## Datasets

We used [DIV2K](http://www.vision.ee.ethz.ch/%7Etimofter/publications/Agustsson-CVPRW-2017.pdf) dataset to train our model. Please download it from [here](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (7.1GB).


You can evaluate your models with widely-used benchmark datasets: 
[Set5](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html),
[Set14](https://sites.google.com/site/romanzeyde/research-interests),
[B100](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/),
[Urban100](https://sites.google.com/site/jbhuang0604/publications/struct_sr).
You can download [benchmark datasets](https://cv.snu.ac.kr/research/EDSR/benchmark.tar) (250MB). Set ``--dir_data <where_benchmark_folder_located>`` to evaluate the models with the benchmarks.


## Data preprocessing

### Unpack location
Unpack the tar file to any place you want. Then, change the ```dir_data``` argument in ```src/option.py``` to the place where DIV2K images are located.

### Preprocessing binaries
We recommend you to pre-process the images before training. This step will decode all **png** files and save them as binaries. Use ``--ext sep_reset`` argument on your first run. You can skip the decoding part and use saved binaries with ``--ext sep`` argument.

### Preprocessing psnr list
Once bin files of DIV2K dataset has been generated, `dataset/create_bin_psnr_fastest.py` can create an index list sorted by psnr (ascending). index is the (u,v)th patch (u - y axis, v - x axis). low psnr indicates difficult sample, which is more worth training

### Preprocessing std list
Once bin files of DIV2K dataset has been generated
`dataset/create_bin_std.py` can create an index list sorted by std (descending). index is the (i,j)th patch.
high std may indicates difficult sample, which is more worth training


### Preprocessing psnr map
Once bin files of DIV2K dataset has been generated, `dataset/create_bin_psnr_map.py` can create psnr maps using integral map accelerating

### Preprocessing psnr TDS list
`dataset/create_bin_psnr_darts.py` can create a sparse index_psnr list (ascending) by by throwing darts.
index is the (u,v)th patch (u - y axis, v - x axis).
low psnr indicates difficult sample, which is more worth training.
This script should be used after generating psnr_map by `dataset/create_bin_psnr_map.py` script.

### Preprocessing psnr nms list
`dataset/create_bin_psnr_nms.py`  can create a sparse index_psnr list (ascending) by by throwing darts.
index is the (u,v)th patch (u - y axis, v - x axis).
low psnr indicates difficult sample, which is more worth training.
This script should be used after generating psnr_map by `dataset/create_bin_psnr_map.py` script.


## Training

There are lots of template in `template.py`, run them by command:
```python
python main.py --template xxx
```
And the args explaination is in the `options.py`.
Here we give some instructions of args:

* For normal training,
`args.data_train` and `args.data_test` are set to `DIV2K` by default.

* For AugSampling training,
`args.data_train` should be set to `DIV2K_PSNR`, and `args.data_test` remains `DIV2K`. Meanwhile `args.data_partion` should be specified a portion (e.g. 0.1). And `args.file_suffix` also should be set for the retrieval of index list files, different files represents different metrics or different sampling strategies (e.g. `_psnr_up_new.pt`, `_std0_rgb_p192_new.pt`, `_psnr_nms_1000.pt`, `_psnr_darts.pt`).

* For OHEM training,
`args.ohem` should be active, and `args.data_partion` indicates the portion of allowing back-propagation loss in a batch.

* For CutBlur training,
`args.cutblur` should be a float between 0 and 1.

* For test,
`args.test_only` can be set true, and if GPU memory is not enough, `args.chop` can be activated for memory-efficient forwarding.