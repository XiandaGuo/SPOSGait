# Gait Recognition in the Wild: A Large-scale Benchmark and NAS-based Baseline

### [Paper](https://arxiv.org/pdf/2205.02692)

> Gait Recognition in the Wild: A Large-scale Benchmark and NAS-based Baseline

> [Xianda Guo](https://scholar.google.com/citations?user=jPvOqgYAAAAJ), [Zheng Zhu](https://scholar.google.com.hk/citations?user=NmwjI0AAAAAJ&hl=zh-CN), Tian Yang, [BeiBei Lin](https://scholar.google.com.hk/citations?user=KyvHam4AAAAJ&hl=zh-CN), Junjie Huang, [Jiankang Deng](https://scholar.google.com.hk/citations?user=Z_UoQFsAAAAJ&hl=zh-CN), Guan Huang, [Jie Zhou](https://scholar.google.com.hk/citations?user=6a79aPwAAAAJ&hl=zh-CN), [Jiwen Lu](https://scholar.google.com.hk/citations?user=TN8uDQoAAAAJ&hl=zh-CN).

## News 

- **[2024/6/24]** Training and evaluation code release.
- **[2024/1]** Paper released on [arXiv](https://arxiv.org/pdf/2205.02692).

## Getting Started

### 0. Prepare datasets

We provide the following tutorials for your reference:
- [Download GREW dataset](docs/1.download_GREW.md)
- [Prepare dataset](docs/2.prepare_dataset.md)


### 1. SupernetTraining
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


### 2. Search
```
多卡搜索
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=8  opengait/search.py --cfgs ./configs/sposgait/sposgait_medium_grew_supertraining_triplet.yaml --max-epochs 10
```

### Calculate_flops_and_params
```
CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch --nproc_per_node=1 opengait/calculate_flops_and_params.py --cfgs configs/sposgait/retrain/sposgait_large_GREW-train20000id_retrain.yaml
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

You can run commands in [train.sh](train.sh) for training different models.

### 4. Test
Evaluate the trained model by
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 opengait/main.py --cfgs ./configs/sposgait/retrain/sposgait_large_GREW-train20000id_retrain.yaml --phase test
```
- `--phase` Specified as `test`.
- `--iter` Specify a iteration checkpoint.

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
@article{guo2022gait,
  title={Gait Recognition in the Wild: A Large-scale Benchmark and NAS-based Baseline},
  author={Guo, Xianda and Zhu, Zheng and Yang, Tian and Lin, Beibei and Huang, Junjie and Deng, Jiankang and Huang, Guan and Zhou, Jie and Lu, Jiwen},
  journal={arXiv e-prints},
  pages={arXiv--2205},
  year={2022}
}
@inproceedings{wang2022gaitstrip,
  title={Gaitstrip: Gait recognition via effective strip-based feature representations and multi-level framework},
  author={Wang, Ming and Lin, Beibei and Guo, Xianda and Li, Lincheng and Zhu, Zheng and Sun, Jiande and Zhang, Shunli and Liu, Yu and Yu, Xin},
  booktitle={Proceedings of the Asian conference on computer vision},
  pages={536--551},
  year={2022}
}
@inproceedings{wang2023dygait,
  title={DyGait: Exploiting dynamic representations for high-performance gait recognition},
  author={Wang, Ming and Guo, Xianda and Lin, Beibei and Yang, Tian and Zhu, Zheng and Li, Lincheng and Zhang, Shunli and Yu, Xin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={13424--13433},
  year={2023}
}

```
**Note**: This code is only used for academic purposes, people cannot use this code for anything that might be considered commercial use.