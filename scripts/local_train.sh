#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# local train

###########################################################################################
# SWIN

# # Swin Transformer
# ./.python-greene submitit_hydra.py $comp exp=train_swin testing=true name="$(date +%F)-test_train_swin" batch_size=16 swin.embed_dim=48 scheduler=True

# Swin Transformer Global
# ./.python-greene submitit_hydra.py $comp exp=train_swin_global testing=true name="$(date +%F)-local_train_swin_global_sched" batch_size=8 swin.embed_dim=48 scheduler=True

###########################################################################################
# UNET

# Basic UNet
# ./.python-greene submitit_hydra.py $comp exp=train_unet testing=true name="$(date +%F)-test_train_unet_withsched" batch_size=8 scheduler=True unet.encoder.n_channels=[180,360,720] unet.decoder.n_channels=[720,360,180]

# Basic UNet - with dil
# ./.python-greene submitit_hydra.py $comp exp=train_unet testing=true name="$(date +%F)-test_train_unet_dil" batch_size=8 scheduler=True unet.encoder.n_channels=[180,360,720] unet.decoder.n_channels=[720,360,180] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1]

# ConvNext UNet
# ./.python-greene submitit_hydra.py $comp exp=train_unet testing=true name="$(date +%F)-test_train_convnextunet" batch_size=8 scheduler=True unet.encoder.n_channels=[45,90,180] unet.decoder.n_channels=[180,90,45] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block

# ConvNext original
# ./.python-greene submitit_hydra.py $comp exp=train_unet testing=true name="$(date +%F)-test_train_orgconvnextunet" batch_size=8 scheduler=True unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block_orig exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block_orig

# ConvNext Inverted UNet
# ./.python-greene submitit_hydra.py $comp exp=train_unet testing=true name="$(date +%F)-test_train_convnextinvunet" batch_size=8 scheduler=True unet.encoder.n_channels=[90,45,23] unet.decoder.n_channels=[23,45,90] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block

# Original ConvNext Inverted UNet
# ./.python-greene submitit_hydra.py $comp exp=train_unet testing=true name="$(date +%F)-test_train_orgconvnextinvunet" batch_size=8 scheduler=True unet.encoder.n_channels=[180,90,45] unet.decoder.n_channels=[45,90,180] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block_orig exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block_orig


# Basic UNet Global
# ./.python-greene submitit_hydra.py $comp exp=train_unet_global testing=true name="$(date +%F)-test_train_basicunet_global" batch_size=8 scheduler=True unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90]

# ConvNext UNet Global
# ./.python-greene submitit_hydra.py $comp exp=train_unet_global testing=true name="$(date +%F)-local_train_convnextinvunet_global" batch_size=4 scheduler=True unet.encoder.n_channels=[45,90,180] unet.decoder.n_channels=[180,90,45] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block

# ConvNext original Global
./.python-greene submitit_hydra.py $comp exp=train_unet_global testing=true name="$(date +%F)-test_train_orgconvnextunet_global" batch_size=4 scheduler=True unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block_orig exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block_orig
