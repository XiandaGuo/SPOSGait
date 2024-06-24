# # **************** For GREW ****************
# SPOSGait-Small
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -u -m torch.distributed.launch --nproc_per_node=8  \\
# opengait/main.py --cfgs configs/sposgait/retrain/sposgait_small_GREW-train500id_retrain.yaml --phase test
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -u -m torch.distributed.launch --nproc_per_node=8  \\
# opengait/main.py --cfgs configs/sposgait/retrain/sposgait_small_GREW-train5000id_retrain.yaml --phase test
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -u -m torch.distributed.launch --nproc_per_node=8  \\
# opengait/main.py --cfgs configs/sposgait/retrain/sposgait_small_GREW-train20000id_retrain.yaml --phase test

# SPOSGait-Medium
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -u -m torch.distributed.launch --nproc_per_node=8  \\
# opengait/main.py --cfgs ./configs/sposgait/retrain/sposgait_medium_GREW-500id_retrain.yaml --phase test
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -u -m torch.distributed.launch --nproc_per_node=8  \\
# opengait/main.py --cfgs ./configs/sposgait/retrain/sposgait_medium_GREW-5000id_retrain.yaml --phase test
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=8 \\
# opengait/main.py --cfgs ./configs/sposgait/retrain/sposgait_medium_GREW-20000id_retrain.yaml --phase test

# SPOSGait-Large
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -u -m torch.distributed.launch --nproc_per_node=8  \\
# opengait/main.py --cfgs configs/sposgait/retrain/sposgait_large_GREW-train500id_retrain.yaml --phase test
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -u -m torch.distributed.launch --nproc_per_node=8  \\
# opengait/main.py --cfgs configs/sposgait/retrain/sposgait_large_GREW-train5000id_retrain.yaml --phase test
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -u -m torch.distributed.launch --nproc_per_node=8  \\
 opengait/main.py --cfgs configs/sposgait/retrain/sposgait_large_GREW-train20000id_retrain.yaml --phase test