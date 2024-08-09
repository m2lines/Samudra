#!/bin/bash

###########################################################################################
# 3D

### ConvNext
# 5
# ENABLE_NSYS_PROFILING=1 PROFILE_OUTPUT=out ./.python-perlmutter submitit_hydra.py compute/greene=1x1 compute/greene/node=a100_debug exp=train_unet_global_3D_5 epochs=4 name="$(date +%F)-profile_convnextunet_global_3D_hist1_5levels_1GPUReal" region=global_3D batch_size=4 scheduler=True rand_seed=10 unet.ch_width=[80,100,150,300,400] hist=1 N_samples=160 --qos=debug


# ENABLE_NSYS_PROFILING=1 PROFILE_OUTPUT=out-gpu-metrics-4x1 ./.python-perlmutter submitit_hydra.py compute/greene=4x1 compute/greene/node=a100_debug exp=train_unet_global_3D_5 epochs=4 name="$(date +%F)-profile_convnextunet_global_3D_hist1_5levels_4GPUReal" region=global_3D batch_size=4 scheduler=True rand_seed=10 unet.ch_width=[80,100,150,300,400] hist=1 N_samples=160 --qos=debug


# ENABLE_NSYS_PROFILING=1 PROFILE_OUTPUT=out-gpu-metrics-1x4 ./.python-perlmutter submitit_hydra.py compute/greene=1x4 compute/greene/node=a100_debug exp=train_unet_global_3D_5 epochs=4 name="$(date +%F)-profile_convnextunet_global_3D_hist1_5levels_4GPUReal" region=global_3D batch_size=4 scheduler=True rand_seed=10 unet.ch_width=[80,100,150,300,400] hist=1 N_samples=160 --qos=debug

ENABLE_NSYS_PROFILING=1 PROFILE_OUTPUT=out-gpu-metrics-1x4-non_blocking_dataloader ./.python-perlmutter submitit_hydra.py compute/greene=1x4 compute/greene/node=a100_debug exp=train_unet_global_3D_5 epochs=4 name="$(date +%F)-profile_convnextunet_global_3D_hist1_5levels_nb_dataloader" region=global_3D batch_size=4 scheduler=True rand_seed=10 unet.ch_width=[80,100,150,300,400] hist=1 N_samples=160 --qos=debug

# All
# ENABLE_NSYS_PROFILING=1 PROFILE_OUTPUT=out ./.python-perlmutter submitit_hydra.py compute/greene=2x2 compute/greene/node=a100_debug exp=train_unet_global_3D_all epochs=4 name="$(date +%F)-profile_convnextunet_global_3D_hist1_130M" region=global_3D batch_size=4 scheduler=True rand_seed=10 unet.ch_width=[157,200,250,300,400] hist=1 N_samples=160 --qos=debug
