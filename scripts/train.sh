#!/bin/bash
export BASE_OE_DIR=$PWD

###########################################################################################
# 3D

### ConvNext
# Surface only
# v0.0
# ./.python-perlmutter submitit_hydra.py compute/greene=2x2 compute/greene/node=a100_4hrs wandb.mode=online exp=train_unet_global_2Dv0.0 name="$(date +%F)-train_convnextunet_global_2Dv0.0_100epochs" epochs=100 region=global_3D batch_size=16 scheduler=True rand_seed=10 --qos=regular

# v0.2.1
# ./.python-perlmutter submitit_hydra.py compute/greene=2x2 compute/greene/node=a100_3hrs wandb.mode=online exp=train_unet_global_2D name="$(date +%F)-train_convnextunet_global_2D" region=global_3D batch_size=16 scheduler=True rand_seed=10

# 5 levels
# ./.python-perlmutter submitit_hydra.py compute/greene=2x2 compute/greene/node=a100_1day wandb.mode=online exp=train_unet_global_3D_5 name="$(date +%F)-train_convnextunet_global_3D_5levels" region=global_3D batch_size=4 scheduler=True rand_seed=10 unet.ch_width=[80,100,150,300,400] hist=1 --qos=regular

# 5 No fast output
# ./.python-perlmutter submitit_hydra.py compute/greene=2x2 compute/greene/node=a100_1day wandb.mode=online exp=train_unet_global_3D_5 name="$(date +%F)-train_convnextunet_global_3D_5levels_NoFastOuts" region=global_3D batch_size=4 scheduler=True rand_seed=10 unet.ch_width=[80,100,150,300,400] hist=1 exp_num_out=3D_noFast_5 --qos=regular

# ./.python-perlmutter submitit_hydra.py compute/greene=2x2 compute/greene/node=a100_debug testing=true exp=train_unet_global_3D_all name="$(date +%F)-convnextunet_v021_testingseed2" region=global_3D batch_size=4 scheduler=True rand_seed=10 unet.ch_width=[157,200,250,300,400] hist=1 N_samples=128 --qos=debug

# All history=1
# ./.python-perlmutter submitit_hydra.py compute/greene=4x2 compute/greene/node=a100_2days wandb.mode=online exp=train_unet_global_3D_all name="$(date +%F)-convnextunet_v021_hist1_8steps" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[157,200,250,300,400] hist=1 epochs=70 steps=8 amp_mode=fp16 enable_fused=True --qos=regular

# All history=1 Fast Smoothed 30
./.python-perlmutter submitit_hydra.py compute/greene=2x2 compute/greene/node=a100_1day wandb.mode=online exp=train_unet_global_3D_all_smoothed30 name="$(date +%F)-convnextunet_v021_hist1_fastsmoothed10_110G" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[157,200,250,300,400] hist=1 epochs=35 --qos=regular

# All history=1 Fast Smoothed 30 TEST
# ./.python-perlmutter submitit_hydra.py compute/greene=2x2 compute/greene/node=a100_debug wandb.mode=online testing=true exp=train_unet_global_3D_all_smoothed30 name="$(date +%F)-convnextunet_v021_hist1_fastsmoothed30_explicit_110g" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[157,200,250,300,400] hist=1 epochs=70 --qos=debug

# All history=1 No fast input/output
# ./.python-perlmutter submitit_hydra.py compute/greene=2x2 compute/greene/node=a100_1day wandb.mode=online exp=train_unet_global_3D_all name="$(date +%F)-convnextunet_v021_hist1_nofastinout_epochs70" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[157,200,250,300,400] hist=1 exp_num_in=3D_noFast_all exp_num_out=3D_noFast_all epochs=70 --qos=regular

# All history=1 Only fast output
# ./.python-perlmutter submitit_hydra.py compute/greene=2x2 compute/greene/node=a100_1day wandb.mode=online exp=train_unet_global_3D_all name="$(date +%F)-convnextunet_v021_hist1_onlyfastout" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[157,200,250,300,400] hist=1 exp_num_out=3D_onlyFast_all --qos=regular


# All history=0
# ./.python-perlmutter submitit_hydra.py compute/greene=2x2 compute/greene/node=a100_1day wandb.mode=online exp=train_unet_global_3D_all name="$(date +%F)-convnextunet_hist0_35epochs" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[80,100,150,300,400] hist=0 --qos=regular


### Swin

# All
# ./.python-perlmutter submitit_hydra.py compute/greene=2x2 compute/greene/node=a100_1day wandb.mode=online exp=train_swin_global_3D_all name="$(date +%F)-train_swin_global_3D_all_out2_35epochs" region=global_3D batch_size=4 scheduler=True rand_seed=10 exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample swin.embed_dim=180 hist=1

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
