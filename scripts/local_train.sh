#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# local train

###########################################################################################
# Global_1 Training

# 1. AdamUNet Global
# ./.python-greene submitit_hydra.py $comp testing=true exp=train_adamunet_global name="$(date +%F)-local_train_adamunet_global_1" region=global_1 batch_size=16 scheduler=True rand_seed=10

# 2. ConvNext UNet Global
# ./.python-greene submitit_hydra.py $comp testing=true exp=train_unet_global name="$(date +%F)-local_train_convnextunet_global_1" region=global_1 batch_size=8 scheduler=True rand_seed=10 exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample

# 3. Swin Global
# ./.python-greene submitit_hydra.py $comp testing=true exp=train_swin_global name="$(date +%F)-local_train_swin_global_1" region=global_1 batch_size=16 scheduler=True rand_seed=10 exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample swin.embed_dim=60


###########################################################################################
# Global_2x Training

# 1. AdamUNet Global
# ./.python-greene submitit_hydra.py $comp testing=true exp=train_adamunet_global name="$(date +%F)-local_train_adamunet_global_2x" region=global_2x batch_size=16 scheduler=True rand_seed=10

# 2. ConvNext UNet Global
# ./.python-greene submitit_hydra.py $comp testing=true exp=train_unet_global name="$(date +%F)-local_train_convnextunet_global_2x" region=global_2x batch_size=8 scheduler=True rand_seed=10

# 3. Swin Global
# ./.python-greene submitit_hydra.py $comp testing=true exp=train_swin_global name="$(date +%F)-local_train_swin_global_2x" region=global_2x batch_size=16 scheduler=True rand_seed=10


###########################################################################################
# Global_1_2x Training

# 1. AdamUNet Global
# ./.python-greene submitit_hydra.py $comp testing=true exp=train_adamunet_global name="$(date +%F)-local_train_adamunet_global_1_2x" region=global_1_2x batch_size=16 scheduler=True rand_seed=10

# 2. ConvNext UNet Global
# ./.python-greene submitit_hydra.py $comp testing=true exp=train_unet_global name="$(date +%F)-local_train_convnextunet_global_1_2x" region=global_1_2x batch_size=8 scheduler=True rand_seed=10

# 3. Swin Global
./.python-greene submitit_hydra.py $comp testing=true exp=train_swin_global name="$(date +%F)-local_train_swin_global_1_2x" region=global_1_2x batch_size=16 scheduler=True rand_seed=10 exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample swin.embed_dim=60