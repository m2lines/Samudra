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

# All
# ./.python-perlmutter submitit_hydra.py $comp exp=train_unet_global_3D_5 name="$(date +%F)-local_train_convnextunet_global_3D_5" region=global_3D batch_size=4 scheduler=True rand_seed=10 unet.ch_width=[80,100,150,300,400] hist=1

./.python-perlmutter submitit_hydra.py $comp testing=true exp=train_unet_global_3D_all name="$(date +%F)-local_train_convnextunet_global_3D_all_upscale2" region=global_3D batch_size=4 scheduler=True rand_seed=10 unet.ch_width=[157,200,250,300,400] hist=1 

### Swin
# ./.python-perlmutter submitit_hydra.py $comp testing=true exp=train_swin_global_3D_all name="$(date +%F)-local_train_swin_global_3D_all" region=global_3D batch_size=4 scheduler=True rand_seed=10 exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample swin.embed_dim=180 hist=1

###########################################################################################
# Global_1 Training

# 1. AdamUNet Global
# ./.python-perlmutter submitit_hydra.py $comp testing=true exp=train_adamunet_global name="$(date +%F)-local_train_adamunet_global_1" region=global_1 batch_size=16 scheduler=True rand_seed=10

# 2. ConvNext UNet Global
# ./.python-perlmutter submitit_hydra.py $comp testing=true exp=train_unet_global name="$(date +%F)-local_train_convnextunet_global_1" region=global_1 batch_size=8 scheduler=True rand_seed=10

# ./.python-perlmutter submitit_hydra.py $comp testing=true exp=train_unet_global name="$(date +%F)-local_train_convnextunet_global_1_7k" region=global_1 batch_size=8 scheduler=True rand_seed=10 N_samples=7000

# 3. Swin Global
# ./.python-perlmutter submitit_hydra.py $comp testing=true exp=train_swin_global name="$(date +%F)-local_train_swin_global_1" region=global_1 batch_size=16 scheduler=True rand_seed=10 exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample swin.embed_dim=60


###########################################################################################
# Global_2x Training

# 1. AdamUNet Global
# ./.python-perlmutter submitit_hydra.py $comp testing=true exp=train_adamunet_global name="$(date +%F)-local_train_adamunet_global_2x" region=global_2x batch_size=16 scheduler=True rand_seed=10

# 2. ConvNext UNet Global
# ./.python-perlmutter submitit_hydra.py $comp testing=true exp=train_unet_global name="$(date +%F)-local_train_convnextunet_global_2x" region=global_2x batch_size=8 scheduler=True rand_seed=10

# 3. Swin Global
# ./.python-perlmutter submitit_hydra.py $comp testing=true exp=train_swin_global name="$(date +%F)-local_train_swin_global_2x" region=global_2x batch_size=16 scheduler=True rand_seed=10


###########################################################################################
# Global_1 loaded Global_2x Training
# 1. AdamUNet Global
# ./.python-perlmutter submitit_hydra.py $comp testing=true exp=train_adamunet_global name="$(date +%F)-local_train_adamunet_load1_global_2x_50p" region=global_2x batch_size=16 scheduler=True rand_seed=10 data_percent=0.5 preload='/scratch/sg7761/m2lines/Ocean_Emulator/train/2024-05-13-foundation_train_adamunet_global_1/adamunetseed/saved_nets/adamunet_best_steps_4_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt'

# 2. ConvNext UNet Global
# ./.python-perlmutter submitit_hydra.py $comp testing=true exp=train_unet_global name="$(date +%F)-local_train_convnextunet_load1_global_2x_05p" region=global_2x batch_size=8 scheduler=True rand_seed=10 data_percent=0.05 preload='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-05-11-foundation_train_convnextunet_global_1/next/saved_nets/convnextunet_best_steps_4_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt'

# 3. Swin Global
# ./.python-perlmutter submitit_hydra.py $comp testing=true exp=train_swin_global name="$(date +%F)-local_train_swin_load1_global_2x_50p" region=global_2x batch_size=16 scheduler=True rand_seed=10 data_percent=0.5 exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample swin.embed_dim=60 preload='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-05-11-foundation_train_swintrans60_global_1/swintrans60/saved_nets/swin_best_steps_4_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt'

###########################################################################################
# Global_1_2x Training

# 1. AdamUNet Global
# ./.python-perlmutter submitit_hydra.py $comp testing=true exp=train_adamunet_global name="$(date +%F)-local_train_adamunet_global_1_2x" region=global_1_2x batch_size=16 scheduler=True rand_seed=10

# 2. ConvNext UNet Global
# ./.python-perlmutter submitit_hydra.py $comp testing=true exp=train_unet_global name="$(date +%F)-local_train_convnextunet_global_1_2x" region=global_1_2x batch_size=8 scheduler=True rand_seed=10

# 3. Swin Global
# ./.python-perlmutter submitit_hydra.py $comp testing=true exp=train_swin_global name="$(date +%F)-local_train_swin_global_1_2x" region=global_1_2x batch_size=16 scheduler=True rand_seed=10 exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample swin.embed_dim=60
