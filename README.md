# Gait Recognition in the Wild: A Large-scale Benchmark and NAS-based Baseline

### [Paper](https://arxiv.org/pdf/2205.02692)

> Gait Recognition in the Wild: A Large-scale Benchmark and NAS-based Baseline

> [Xianda Guo](https://scholar.google.com/citations?user=jPvOqgYAAAAJ), [Zheng Zhu](https://scholar.google.com.hk/citations?user=NmwjI0AAAAAJ&hl=zh-CN), Tian Yang, [BeiBei Lin](https://scholar.google.com.hk/citations?user=KyvHam4AAAAJ&hl=zh-CN), Junjie Huang, [Jiankang Deng](https://scholar.google.com.hk/citations?user=Z_UoQFsAAAAJ&hl=zh-CN), Guan Huang, [Jie Zhou](https://scholar.google.com.hk/citations?user=6a79aPwAAAAJ&hl=zh-CN), [Jiwen Lu](https://scholar.google.com.hk/citations?user=TN8uDQoAAAAJ&hl=zh-CN).

## News 

- **[2024/6/24]** Training and evaluation code release.
- **[2024/1]** Paper released on [arXiv](https://arxiv.org/pdf/2205.02692).

## SupernetTraining
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=8 opengait/main.py --cfgs configs/sposgait/sposgait_large_grew_supertraining_triplet.yaml --phase train
```
- `python -m torch.distributed.launch` [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) launch instruction.
- `--nproc_per_node` The number of gpus to use, and it must equal the length of `CUDA_VISIBLE_DEVICES`.
- `--cfgs` The path to config file.
- `--phase` Specified as `train`.
<!-- - `--iter` You can specify a number of iterations or use `restore_hint` in the config file and resume training from there. -->
- `--log_to_file` If specified, the terminal log will be written on disk simultaneously. 

You can run commands in [train.sh](train.sh) for training different models.


## Search
```
多卡搜索
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=8  opengait/search.py --cfgs ./configs/sposgait/sposgait_medium_grew_supertraining_triplet.yaml --max-epochs 10
```

## Calculate_flops_and_params
```
CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch --nproc_per_node=1 opengait/calculate_flops_and_params.py --cfgs configs/sposgait/retrain/sposgait_large_GREW-train20000id_retrain.yaml
```


## ReTrain
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

You can run commands in [train.sh](train.sh) for training different models.

## Test
Evaluate the trained model by
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 opengait/main.py --cfgs ./configs/sposgait/retrain/sposgait_large_GREW-train20000id_retrain.yaml --phase test
```
- `--phase` Specified as `test`.
- `--iter` Specify a iteration checkpoint.

## Citation
```
@article{guo2022gait,
  title={Gait Recognition in the Wild: A Large-scale Benchmark and NAS-based Baseline},
  author={Guo, Xianda and Zhu, Zheng and Yang, Tian and Lin, Beibei and Huang, Junjie and Deng, Jiankang and Huang, Guan and Zhou, Jie and Lu, Jiwen},
  journal={arXiv e-prints},
  pages={arXiv--2205},
  year={2022}
}
```
**Note**: This code is only used for academic purposes, people cannot use this code for anything that might be considered commercial use.