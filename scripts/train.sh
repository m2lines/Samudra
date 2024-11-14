#!/bin/bash
export BASE_OE_DIR=$PWD

###########################################################################################
# 3D

# All history=1 hfds anom
./.python-greene submitit_hydra.py compute/greene=1x4 compute/greene/node=a100_30hrs wandb.mode=online exp=train_unet_global_3D_all_hfds_anom_1975 name="$(date +%F)-convnextunet_v021_hist1_hfds_anom_1975_seed1" region=global_3D batch_size=4 scheduler=True rand_seed=1 unet.ch_width=[157,200,250,300,400] hist=1 epochs=70

# All history=1 hfds anom + no fast inp/out 
# ./.python-greene submitit_hydra.py compute/greene=1x4 compute/greene/node=a100_1day wandb.mode=online exp=train_unet_global_3D_all_hfds_anom_1975 name="$(date +%F)-convnextunet_v021_hist1_hfds_anom_1975_nofast_seed1" region=global_3D batch_size=4 scheduler=True rand_seed=1 unet.ch_width=[157,200,250,300,400] hist=1 epochs=70 exp_num_in=3D_noFast_all exp_num_out=3D_noFast_all
