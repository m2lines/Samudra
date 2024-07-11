#!/bin/bash

###########################################################################################
# 3D

### ConvNext
# Surface only
# ./.python-perlmutter submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet_global_3D wandb.mode=online name="$(date +%F)-train_convnextunet_global_3D_surface_fromdisk" region=global_3D batch_size=16 scheduler=True rand_seed=9

# All
./.python-perlmutter submitit_hydra.py compute/greene=1x1 compute/greene/node=a100 wandb.mode=online exp=train_unet_global_3D_all name="$(date +%F)-train_convnextunet_global_3D_all_debug_multigpu1x1" region=global_3D batch_size=4 scheduler=True rand_seed=5 unet.ch_width=[80,100,150,300,400]

### Swin

# All
# ./.python-perlmutter submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000 exp=train_swin_global_3D_all wandb.mode=online name="$(date +%F)-train_swin_global_3D_all" region=global_3D batch_size=16 scheduler=True rand_seed=10 exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample swin.embed_dim=60

###########################################################################################
# Global_1 Training

# 1. AdamUNet Global
# ./.python-perlmutter submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_adamunet_global wandb.mode=online name="$(date +%F)-foundation_train_adamunet_global_1" region=global_1 batch_size=16 scheduler=True rand_seed=10

# 2. ConvNext UNet Global
# ./.python-perlmutter submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_6hrs exp=train_unet_global wandb.mode=online name="$(date +%F)-foundation_train_convnextunet_global_1" region=global_1 batch_size=8 scheduler=True rand_seed=10

# ./.python-perlmutter submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_6hrs exp=train_unet_global wandb.mode=online name="$(date +%F)-foundation_train_convnextunet_global_1_7k_seed10" region=global_1 batch_size=8 scheduler=True rand_seed=10 N_samples=7000

# 3. Swin Global
# ./.python-perlmutter submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_swin_global wandb.mode=online name="$(date +%F)-foundation_train_swintrans_global_1" region=global_1 batch_size=16 scheduler=True rand_seed=12 exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample swin.embed_dim=60

###########################################################################################
# Global_2x Training

# 1. AdamUNet Global
# ./.python-perlmutter submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_adamunet_global wandb.mode=online name="$(date +%F)-foundation_train_adamunet_global_2x" region=global_2x batch_size=16 scheduler=True rand_seed=10

# 2. ConvNext UNet Global
# ./.python-perlmutter submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_6hrs exp=train_unet_global wandb.mode=online name="$(date +%F)-foundation_train_convnextunet_global_2x" region=global_2x batch_size=8 scheduler=True rand_seed=10

# 3. Swin Global
# ./.python-perlmutter submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_swin_global wandb.mode=online name="$(date +%F)-foundation_train_swin_global_2x" region=global_2x batch_size=16 scheduler=True rand_seed=12 exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample swin.embed_dim=60


###########################################################################################
# Global_1 preloaded Global_2x Training

# 1. AdamUNet Global
# ./.python-perlmutter submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_adamunet_global wandb.mode=online name="$(date +%F)-foundation_train_adamunet_loadedglobal1_global_2x_50p" region=global_2x batch_size=16 scheduler=True data_percent=0.5 rand_seed=10 preload='/scratch/sg7761/m2lines/Ocean_Emulator/train/2024-05-13-foundation_train_adamunet_global_1/adamunetseed/saved_nets/adamunet_best_steps_4_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt'

# 2. ConvNext UNet Global
# ./.python-perlmutter submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_6hrs exp=train_unet_global wandb.mode=online name="$(date +%F)-foundation_train_convnextunet_loadedglobal1_global_2x_50p" region=global_2x batch_size=8 scheduler=True data_percent=0.5 rand_seed=10 preload='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-05-11-foundation_train_convnextunet_global_1/next/saved_nets/convnextunet_best_steps_4_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt'

# 3. Swin Global
# ./.python-perlmutter submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_swin_global wandb.mode=online name="$(date +%F)-foundation_train_swin_loadedglobal1_global_2x_50p" region=global_2x batch_size=16 scheduler=True data_percent=0.5 rand_seed=12 exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample swin.embed_dim=60 preload='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-05-11-foundation_train_swintrans60_global_1/swintrans60/saved_nets/swin_best_steps_4_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt'


###########################################################################################
# Global_1_2x Training

# 1. AdamUNet Global
# ./.python-perlmutter submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_adamunet_global wandb.mode=online name="$(date +%F)-foundation_train_adamunet_global_1_2x" region=global_1_2x batch_size=16 scheduler=True rand_seed=10

# 2. ConvNext UNet Global
# ./.python-perlmutter submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_6hrs exp=train_unet_global wandb.mode=online name="$(date +%F)-foundation_train_convnextunet_global_1_2x" region=global_1_2x batch_size=8 scheduler=True rand_seed=10

# 3. Swin Global
# ./.python-perlmutter submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_swin_global wandb.mode=online name="$(date +%F)-foundation_train_swin_global_1_2x" region=global_1_2x batch_size=16 scheduler=True rand_seed=12 exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample swin.embed_dim=60
