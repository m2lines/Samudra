#!/bin/bash
export BASE_OE_DIR=$PWD

###########################################################################################
# 3D

# All history=0 CM4 no fast inp/out
# ./.python-empireai submitit_hydra.py compute/greene=4x2 compute/greene/node=a100_30hrs wandb.mode=online exp=train_unet_global_3D_all_CM4 name="$(date +%F)-convnextunet_CM4_hist0" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[157,200,250,300,400] hist=0 epochs=70 exp_num_in=3D_noFast_all exp_num_out=3D_noFast_all --qos=regular

# All history=1 CM4 no fast inp/out - approx 29 hrs for 8 GPUs
./.python-empireai submitit_hydra.py compute/greene=4x2 compute/greene/node=a100_30hrs wandb.mode=online exp=train_unet_global_3D_all_CM4 name="$(date +%F)-convnextunet_CM4_hist0_with_SAT_tos" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[157,200,250,300,400] hist=0 exp_num_in=3D_noFast_all exp_num_extra=3D_all_SAT_tos exp_num_out=3D_noFast_all --qos=regular

# All history=1 CM4 All vars - approx 22 hrs for 16 GPUs
# ./.python-empireai submitit_hydra.py compute/greene=8x2 compute/greene/node=a100_30hrs wandb.mode=online exp=train_unet_global_3D_all_CM4 name="$(date +%F)-convnextunet_CM4_hist1_allvars_135epochs" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[157,200,250,300,400] hist=1 epochs=135 --qos=regular

### Swin
# All history=1 CM4 no fast inp/out
# ./.python-empireai submitit_hydra.py compute/greene=4x2 compute/greene/node=a100_30hrs wandb.mode=online exp=train_swin_global_3D_all_CM4 name="$(date +%F)-swinv1_CM4_hist1_nofast" region=global_3D batch_size=4 scheduler=True rand_seed=15 hist=1 epochs=70 exp_num_in=3D_noFast_all exp_num_out=3D_noFast_all --qos=regular
