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
./.python-perlmutter submitit_hydra.py $comp testing=true exp=train_unet_global_3D_5 wandb.mode=online name="$(date +%F)-local_train_convnextunet_global_3D_5" region=global_3D batch_size=4 scheduler=True rand_seed=10 unet.ch_width=[80,100,150,300,400] hist=1 exp_num_out=3D_5_noFast

# Hist 0
# ./.python-perlmutter submitit_hydra.py $comp exp=train_unet_global_3D_all testing=true name="$(date +%F)-local_train_convnextunet_global_3Dv021" region=global_3D batch_size=4 scheduler=True rand_seed=10 unet.ch_width=[80,100,150,300,400] hist=0

# Hist 1
# ./.python-perlmutter submitit_hydra.py compute=local exp=train_unet_global_3D_all testing=true name="$(date +%F)-local_train_convnextunet_global_3D_seedtest2" region=global_3D batch_size=4 scheduler=True rand_seed=10 unet.ch_width=[157,200,250,300,400] hist=1


### Swin
# ./.python-perlmutter submitit_hydra.py $comp testing=true exp=train_swin_global_3D_all name="$(date +%F)-local_train_swin_global_3D_all" region=global_3D batch_size=4 scheduler=True rand_seed=10 exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample swin.embed_dim=180 hist=1
