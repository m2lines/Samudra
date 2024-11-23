#!/bin/bash

# Obvious / static arguments
comp="compute=local"
export BASE_OE_DIR=$PWD

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# local train

###########################################################################################
# 3D
### ConvNext
# Surface only
# ./.python-perlmutter submitit_hydra.py $comp testing=true exp=train_unet_global_2Dv0.0 name="$(date +%F)-local_train_convnextunet_global_2D_v0.0" region=global_3D batch_size=16 scheduler=True rand_seed=10

# 5
# ./.python-perlmutter submitit_hydra.py $comp testing=true exp=train_unet_global_3D_5 name="$(date +%F)-local_train_convnextunet_global_3D_5" region=global_3D batch_size=4 scheduler=True rand_seed=10 unet.ch_width=[80,100,150,300,400] hist=1

# 5 No fast output
# ./.python-perlmutter submitit_hydra.py $comp testing=true exp=train_unet_global_3D_5 wandb.mode=online name="$(date +%F)-local_train_convnextunet_global_3D_5" region=global_3D batch_size=4 scheduler=True rand_seed=10 unet.ch_width=[80,100,150,300,400] hist=1 exp_num_out=3D_noFast_5

# Hist 0
# ./.python-perlmutter submitit_hydra.py $comp exp=train_unet_global_3D_all testing=true name="$(date +%F)-local_train_convnextunet_global_3Dv021" region=global_3D batch_size=4 scheduler=True rand_seed=10 unet.ch_width=[80,100,150,300,400] hist=0

# Hist 1 CM4 all
# ./.python-perlmutter submitit_hydra.py compute=local testing=true exp=train_unet_global_3D_all_CM4 name="$(date +%F)-convnextunet_v021_hist1_CM4" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[157,200,250,300,400] hist=1 epochs=100

# Hist 1 CM4 no fast
# ./.python-perlmutter submitit_hydra.py compute=local testing=true exp=train_unet_global_3D_all_CM4 name="$(date +%F)-convnextunet_v021_hist1_CM4_nofast" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[157,200,250,300,400] hist=1 epochs=100 exp_num_in=3D_noFast_all exp_num_out=3D_noFast_all 

### Swin
# Hist 1 CM4 all
# ./.python-perlmutter submitit_hydra.py $comp testing=true exp=train_swin_global_3D_all_CM4 name="$(date +%F)-local_train_swin_global_3D_all_CM4" region=global_3D batch_size=4 scheduler=True rand_seed=10 hist=1

# Hist 1 CM4 no fast
./.python-perlmutter submitit_hydra.py $comp testing=true exp=train_swin_global_3D_all_CM4 name="$(date +%F)-local_train_swin_global_3D_all_CM4_nofast" region=global_3D batch_size=4 scheduler=True rand_seed=10 hist=1 exp_num_in=3D_noFast_all exp_num_out=3D_noFast_all 
