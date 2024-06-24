# # # **************** 1.Supernet Training ********************************

# # ********For GREW    ****************
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=8  \\
#opengait/main.py --cfgs configs/sposgait/retrain/sposgait_large_grew_supertraining_triplet.yaml

# # ********For CASIA-B ****************
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=8  \\
#opengait/main.py --cfgs configs/sposgait/sposgait_small_casiab_supertraining_2loss.yaml

# # # ****************2. Searching   **************************************
# # *******For CASIA-B ****************
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=8  \\
#opengait/search.py --cfgs configs/sposgait/sposgait_small_casiab_supertraining_2loss.yaml --max-epochs 2

# # *******For GREW ****************
# SPOSGait-Small
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=8  \\
#opengait/search.py --cfgs configs/sposgait/sposgait_small_grew_supertraining_triplet.yaml --max-epochs 2

# SPOSGait-Medium
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=8  \\
#opengait/search.py --cfgs configs/sposgait/sposgait_medium_grew_supertraining_triplet.yaml --max-epochs 10

# SPOSGait-Large
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=8  \\
#opengait/search.py --cfgs configs/sposgait/sposgait_large_grew_supertraining_triplet.yaml --max-epochs 20

# # # ****************3.Retraining *****************************************
# # *******For OUMVLP ****************
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -u -m torch.distributed.launch --nproc_per_node=8 opengait/main.py --cfgs ./configs/sposgait/retrain/sposgait_large_OUMVLP_retrain.yaml

# # *******For Gait3D ****************
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -u -m torch.distributed.launch --nproc_per_node=8 \\
# opengait/main.py --cfgs ./configs/sposgait/retrain/sposgait_large_Gait3D_retrain.yaml

# # *******For GREW ****************
# SPOSGait-Small
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -u -m torch.distributed.launch --nproc_per_node=8  \\
# opengait/main.py --cfgs configs/sposgait/retrain/sposgait_small_GREW-train500id_retrain.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -u -m torch.distributed.launch --nproc_per_node=8  \\
# opengait/main.py --cfgs configs/sposgait/retrain/sposgait_small_GREW-train5000id_retrain.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -u -m torch.distributed.launch --nproc_per_node=8  \\
# opengait/main.py --cfgs configs/sposgait/retrain/sposgait_small_GREW-train20000id_retrain.yaml

# SPOSGait-Medium
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -u -m torch.distributed.launch --nproc_per_node=8  \\
# opengait/main.py --cfgs ./configs/sposgait/retrain/sposgait_medium_GREW-500id_retrain.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -u -m torch.distributed.launch --nproc_per_node=8  \\
# opengait/main.py --cfgs ./configs/sposgait/retrain/sposgait_medium_GREW-5000id_retrain.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=8 \\
# opengait/main.py --cfgs ./configs/sposgait/retrain/sposgait_medium_GREW-20000id_retrain.yaml

# SPOSGait-Large
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -u -m torch.distributed.launch --nproc_per_node=8  \\
# opengait/main.py --cfgs configs/sposgait/retrain/sposgait_large_GREW-train500id_retrain.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -u -m torch.distributed.launch --nproc_per_node=8  \\
# opengait/main.py --cfgs configs/sposgait/retrain/sposgait_large_GREW-train5000id_retrain.yaml
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -u -m torch.distributed.launch --nproc_per_node=8  \\
 opengait/main.py --cfgs configs/sposgait/retrain/sposgait_large_GREW-train20000id_retrain.yaml

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -u -m torch.distributed.launch --nproc_per_node=8  --master_port 29501 opengait/search.py --cfgs configs/sposgait/sposgait_large_grew_supertraining_2loss.yaml --max-epochs 10 >> output/GREW/logs/sposgait_large_GREW-train20000id_2loss_search.log &