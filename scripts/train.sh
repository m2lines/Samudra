#!/bin/bash
export BASE_OE_DIR=$PWD

###########################################################################################
# 3D

# All history=0 CM4 no fast inp/out 
./.python-perlmutter submitit_hydra.py compute/greene=2x2 compute/greene/node=a100_30hrs wandb.mode=online exp=train_unet_global_3D_all_CM4 name="$(date +%F)-convnextunet_CM4_hist0" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[157,200,250,300,400] hist=0 epochs=70 exp_num_in=3D_noFast_all exp_num_out=3D_noFast_all --qos=regular

# All history=1 CM4 no fast inp/out 
# ./.python-perlmutter submitit_hydra.py compute/greene=2x2 compute/greene/node=a100_30hrs wandb.mode=online exp=train_unet_global_3D_all_CM4 name="$(date +%F)-convnextunet_CM4_hist1" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[157,200,250,300,400] hist=1 epochs=70 exp_num_in=3D_noFast_all exp_num_out=3D_noFast_all --qos=regular