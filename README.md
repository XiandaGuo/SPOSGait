# Gait Recognition in the Wild: A Large-scale Benchmark and NAS-based Baseline

### [Paper](https://arxiv.org/pdf/2205.02692)

> [Gait Recognition in the Wild: A Large-scale Benchmark and NAS-based Baseline](https://arxiv.org/pdf/2205.02692)

> [Xianda Guo](https://scholar.google.com/citations?user=jPvOqgYAAAAJ), [Zheng Zhu](https://scholar.google.com.hk/citations?user=NmwjI0AAAAAJ&hl=zh-CN), Tian Yang, [BeiBei Lin](https://scholar.google.com.hk/citations?user=KyvHam4AAAAJ&hl=zh-CN), Junjie Huang, [Jiankang Deng](https://scholar.google.com.hk/citations?user=Z_UoQFsAAAAJ&hl=zh-CN), Guan Huang, [Jie Zhou](https://scholar.google.com.hk/citations?user=6a79aPwAAAAJ&hl=zh-CN), [Jiwen Lu](https://scholar.google.com.hk/citations?user=TN8uDQoAAAAJ&hl=zh-CN).

## News 
- **[2025/2]** This paper has been accepted to T-PAMI.
- **[2024/6/24]** Training and evaluation code release.
- **[2024/1]** Paper released on [arXiv](https://arxiv.org/pdf/2205.02692).

## Getting Started

### 0. Prepare datasets

We provide the following tutorials for your reference:
- [Download GREW dataset](docs/1.download_GREW.md)
- [Prepare dataset](docs/2.prepare_dataset.md)


### 1. SupernetTraining
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=8 opengait/main.py --cfgs configs/sposgait/sposgait_large_GREW_supertraining_triplet.yaml --phase train
```
- `python -m torch.distributed.launch` [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) launch instruction.
- `--nproc_per_node` The number of gpus to use, and it must equal the length of `CUDA_VISIBLE_DEVICES`.
- `--cfgs` The path to config file.
- `--phase` Specified as `train`.
<!-- - `--iter` You can specify the number of iterations or use `restore_hint` in the config file and resume training from there. -->
- `--log_to_file` If specified, the terminal log will be written on disk simultaneously. 

You can run commands in [train.sh](train.sh) to train different models.


### 2. Search
```
多卡搜索
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=8  opengait/search.py --cfgs ./configs/sposgait/sposgait_large_GREW_supertraining_triplet.yaml --max-epochs 20
```


### 3. ReTrain
Train a model by
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=8 opengait/main.py --cfgs ./configs/sposgait/retrain/sposgait_large_GREW-train20000id_retrain.yaml --phase train
```
- `python -m torch.distributed.launch` [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) launch instruction.
- `--nproc_per_node` The number of gpus to use, and it must equal the length of `CUDA_VISIBLE_DEVICES`.
- `--cfgs` The path to config file.
- `--phase` Specified as `train`.
<!-- - `--iter` You can specify a number of iterations or use `restore_hint` in the config file and resume training from there. -->
- `--log_to_file` If specified, the terminal log will be written on disk simultaneously. 

You can run commands in [train.sh](train.sh) to train different models.

### 4. Test
Evaluate the trained model by
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 opengait/main.py --cfgs ./configs/sposgait/retrain/sposgait_large_GREW-train20000id_retrain.yaml --phase test
```
- `--phase` Specified as `test`.
- `--iter` Specify an iteration checkpoint.

You can run commands in [test.sh](test.sh) to train different models.

Participants must package the submission.csv for submission using zip xxx.zip $CSV_PATH and then upload it to [codalab](https://codalab.lisn.upsaclay.fr/competitions/3409).

### Calculate_flops_and_params
```
CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch --nproc_per_node=1 opengait/calculate_flops_and_params.py --cfgs configs/sposgait/retrain/sposgait_large_GREW-train20000id_retrain.yaml
```


## Acknowledgement
- [DyGait](https://github.com/M-Candy77/DyGait)
- [OpenGait](https://github.com/ShiqiYu/OpenGait)


## Citation
If this work is helpful for your research, please consider citing the following BibTeX entries.
```
@inproceedings{zhu2021gait,
  title={Gait recognition in the wild: A benchmark},
  author={Zhu, Zheng and Guo, Xianda and Yang, Tian and Huang, Junjie and Deng, Jiankang and Huang, Guan and Du, Dalong and Lu, Jiwen and Zhou, Jie},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={14789--14799},
  year={2021}
}
@ARTICLE{10906429,
  author={Guo, Xianda and Zhu, Zheng and Yang, Tian and Lin, Beibei and Huang, Junjie and Deng, Jiankang and Huang, Guan and Zhou, Jie and Lu, Jiwen},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Gait Recognition in the Wild: A Large-scale Benchmark and NAS-based Baseline}, 
  year={2025},
  volume={},
  number={},
  pages={1-18},
  keywords={Gait recognition;Benchmark testing;Training;Three-dimensional displays;Legged locomotion;Cameras;Videos;Streams;Face recognition;Neural architecture search;Large-scale Gait Recognition;Biometric Authentication;Neural Architecture Search},
  doi={10.1109/TPAMI.2025.3546482}
}

```
**Note**: This code is only used for academic purposes, people cannot use this code for anything that might be considered commercial use.
