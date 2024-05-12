#!/bin/bash


###########################################################################################
# Global_1 Training

# 1. AdamUNet Global
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_adamunet_global wandb.mode=online name="$(date +%F)-foundation_train_adamunet_global_1" region=global_1 batch_size=16 scheduler=True rand_seed=10

# 2. ConvNext UNet Global
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_6hrs exp=train_unet_global wandb.mode=online name="$(date +%F)-foundation_train_convnextunet_global_1" region=global_1 batch_size=8 scheduler=True rand_seed=10

# 3. Swin Global
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_swin_global wandb.mode=online name="$(date +%F)-foundation_train_swin_global_1" region=global_1 batch_size=16 scheduler=True rand_seed=10

./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_swin_global wandb.mode=online name="$(date +%F)-foundation_train_swintrans_global_1" region=global_1 batch_size=16 scheduler=True rand_seed=12 exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample swin.embed_dim=60

###########################################################################################
# Global_2x Training

# 1. AdamUNet Global
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_adamunet_global wandb.mode=online name="$(date +%F)-foundation_train_adamunet_global_2x" region=global_2x batch_size=16 scheduler=True rand_seed=10

# 2. ConvNext UNet Global
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_6hrs exp=train_unet_global wandb.mode=online name="$(date +%F)-foundation_train_convnextunet_global_2x" region=global_2x batch_size=8 scheduler=True rand_seed=10

# 3. Swin Global
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_swin_global wandb.mode=online name="$(date +%F)-foundation_train_swin_global_2x" region=global_2x batch_size=16 scheduler=True rand_seed=10