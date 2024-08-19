#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# local train

###########################################################################################
# 3D
### ConvNext
# Surface only
# ./.python-perlmutter submitit_hydra.py $comp testing=true exp=train_unet_global_3D name="$(date +%F)-local_train_convnextunet_global_3D_surface" region=global_3D batch_size=16 scheduler=True rand_seed=10

# 5
# ./.python-perlmutter submitit_hydra.py $comp exp=train_unet_global_3D_5 name="$(date +%F)-local_train_convnextunet_global_3D_5" region=global_3D batch_size=4 scheduler=True rand_seed=10 unet.ch_width=[80,100,150,300,400] hist=1

# Hist 0
# ./.python-perlmutter submitit_hydra.py $comp exp=train_unet_global_3D_all testing=true name="$(date +%F)-local_train_convnextunet_global_3Dv021" region=global_3D batch_size=4 scheduler=True rand_seed=10 unet.ch_width=[80,100,150,300,400] hist=0

# Hist 1
./.python-perlmutter submitit_hydra.py $comp exp=train_unet_global_3D_all name="$(date +%F)-local_train_convnextunet_global_3D" region=global_3D batch_size=4 scheduler=True rand_seed=10 unet.ch_width=[157,200,250,300,400] hist=1 resume_ckpt_path=/pscratch/sd/s/suryad/Ocean_Emulator/train_3D/2024-08-16-convnextunet_v021float64_hist1_out2_35epochs_randseed10/hist1/saved_nets/convnextunet_epoch_35_steps_4_global_3D_all_N_train_4000_Lateral_Data_025_no_smooth.pt


### Swin
# ./.python-perlmutter submitit_hydra.py $comp testing=true exp=train_swin_global_3D_all name="$(date +%F)-local_train_swin_global_3D_all" region=global_3D batch_size=4 scheduler=True rand_seed=10 exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample swin.embed_dim=180 hist=1
