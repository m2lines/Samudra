#!/bin/bash

# 1. Basic UNet - GulfStream, Global
# 2. ConvNeXT UNet, best model - GulfStream, Global
# 3. Swin Transformer - GulfStream, Global

###########################################################################################
# SWIN

# Swin Transformer (40M)
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_swin wandb.mode=online name="$(date +%F)-train_swin" batch_size=16 swin.embed_dim=48 scheduler=True


# Swin transformer Global
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_swin_global wandb.mode=online name="$(date +%F)-train_swin_global" batch_size=8 swin.embed_dim=48 scheduler=True

###########################################################################################
# UNET

# Basic UNet
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet wandb.mode=online name="$(date +%F)-train_basicunet"  batch_size=8 scheduler=True unet.encoder.n_channels=[180,360,720] unet.decoder.n_channels=[720,360,180]

# Basic UNet - with dil
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet wandb.mode=online name="$(date +%F)-train_basicunet_dil" batch_size=8 scheduler=True unet.encoder.n_channels=[180,360,720] unet.decoder.n_channels=[720,360,180] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1]

# ConvNext UNet
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet wandb.mode=online name="$(date +%F)-train_convnextunet" batch_size=8 scheduler=True unet.encoder.n_channels=[45,90,180] unet.decoder.n_channels=[180,90,45] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block

# Original ConvNext UNet
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet wandb.mode=online name="$(date +%F)-train_orgconvnextunet" batch_size=8 scheduler=True unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block_orig exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block_orig

# ConvNext Inverted UNet
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet wandb.mode=online name="$(date +%F)-train_convnextinvunet" batch_size=8 scheduler=True unet.encoder.n_channels=[90,45,23] unet.decoder.n_channels=[23,45,90] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block

# Original ConvNext Inverted UNet
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet wandb.mode=online name="$(date +%F)-train_orgconvnextinvunet" batch_size=8 scheduler=True unet.encoder.n_channels=[180,90,45] unet.decoder.n_channels=[45,90,180] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block_orig exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block_orig

# Basic UNet Global
./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet_global wandb.mode=online name="$(date +%F)-train_basicunet_global"  batch_size=8 scheduler=True unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90]

# ConvNext UNet Global
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_6hrs exp=train_unet_global wandb.mode=online name="$(date +%F)-train_convnextunet_global_bs4_6hrs" batch_size=4 scheduler=True unet.encoder.n_channels=[45,90,180] unet.decoder.n_channels=[180,90,45] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block