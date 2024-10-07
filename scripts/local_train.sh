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

# Hist 1
# ./.python-perlmutter submitit_hydra.py compute=local testing=true exp=train_unet_global_3D_all name="$(date +%F)-local_train_convnextunet_global_3D_hist1" region=global_3D batch_size=4 scheduler=True rand_seed=10 unet.ch_width=[157,200,250,300,400] hist=1 loss=mse_cos_weighted

# Hist 1 anom
# ./.python-perlmutter submitit_hydra.py compute=local exp=train_unet_global_3D_all_hfds_anom name="$(date +%F)-local_train_convnextunet_global_3D_hist1_hfds_anom" region=global_3D batch_size=4 scheduler=True rand_seed=10 unet.ch_width=[157,200,250,300,400] hist=1 

# anom 1975 + TS
# ./.python-perlmutter submitit_hydra.py compute=local testing=true exp=train_unet_global_3D_all_hfds_anom_1975 name="$(date +%F)-convnextunet_v021_hist1_hfds_anom_1975_TS_hist1" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[157,200,250,300,400] hist=2 epochs=70 exp_num_in=3D_TS_all exp_num_out=3D_TS_all

# anom 1975 + nofast - T,S,SSH
./.python-perlmutter submitit_hydra.py compute=local testing=true exp=train_unet_global_3D_all_hfds_anom_1975 name="$(date +%F)-convnextunet_v021_hist1_hfds_anom_1975_nofast" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[157,200,250,300,400] hist=1 epochs=70 exp_num_in=3D_noFast_all exp_num_out=3D_noFast_all 

# Hist 1 anom 1975 + Only temp
# ./.python-perlmutter submitit_hydra.py compute=local testing=true exp=train_unet_global_3D_all_hfds_anom_1975_finetune name="$(date +%F)-convnextunet_v021_hist1_hfds_anom_1975_onlytemp_deltaTmix_Full_Finetune" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[157,200,250,300,400] hist=1 epochs=70 exp_num_in=3D_onlyTemp_all exp_num_out=3D_onlyTemp_all resume_ckpt_path=/pscratch/sd/s/suryad/Ocean_Emulator/train_3D/2024-09-29-convnextunet_v021_hist1_hfds_anom_1975_onlyTemp_deltaTmix_Full/convnextunet_epoch_20_beststeps_4_global_3D_all_N_train_2850_Lateral_Data_025_no_smooth.pt

# Hist 1 anom 1982_90 Reordered
# ./.python-perlmutter submitit_hydra.py compute=local exp=train_unet_global_3D_all_hfds_anom_1980_90reordered name="$(date +%F)-local_train_convnextunet_v021_hist1_hfds_anom_80_92" region=global_3D batch_size=4 scheduler=True rand_seed=10 unet.ch_width=[157,200,250,300,400] hist=1 

# Hist 1 anom 1982_90 Reordered + no fast
# ./.python-perlmutter submitit_hydra.py compute=local testing=true exp=train_unet_global_3D_all_hfds_anom_1980_90reordered name="$(date +%F)-local_train_convnextunet_v021_hist1_hfds_anom_80_92_nofast" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[157,200,250,300,400] hist=1 epochs=70 exp_num_in=3D_noFast_all exp_num_out=3D_noFast_all

# Hist 1 Fast Smoothed 30
# ./.python-perlmutter submitit_hydra.py compute=local exp=train_unet_global_3D_all_smoothed30 name="$(date +%F)-local_train_convnextunet_global_3D_smoothed" region=global_3D batch_size=4 scheduler=True rand_seed=10 unet.ch_width=[157,200,250,300,400] hist=1

# Hist 1 No fast input/output
# ./.python-perlmutter submitit_hydra.py compute=local exp=train_unet_global_3D_all testing=true name="$(date +%F)-local_train_convnextunet_global_3D_seedtest2" region=global_3D batch_size=4 scheduler=True rand_seed=10 unet.ch_width=[157,200,250,300,400] hist=1 exp_num_in=3D_noFast_all exp_num_extra=3D_noFast_all exp_num_out=3D_noFast_all

# Hist 1 Only fast output
# ./.python-perlmutter submitit_hydra.py compute=local exp=train_unet_global_3D_all testing=true name="$(date +%F)-local_train_convnextunet_global_3D_seedtest2" region=global_3D batch_size=4 scheduler=True rand_seed=10 unet.ch_width=[157,200,250,300,400] hist=1 exp_num_out=3D_onlyFast_all

### Swin
# ./.python-perlmutter submitit_hydra.py $comp testing=true exp=train_swin_global_3D_all name="$(date +%F)-local_train_swin_global_3D_all" region=global_3D batch_size=4 scheduler=True rand_seed=10 exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample swin.embed_dim=180 hist=1
