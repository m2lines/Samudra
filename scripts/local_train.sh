#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# local train

###########################################################################################
# SWIN

# Swin Transformer Global
# ./.python-greene submitit_hydra.py $comp exp=train_swin_global name="$(date +%F)-local_train_swin_global" batch_size=8 swin.embed_dim=48 wandb.mode=online # testing=True

./.python-greene submitit_hydra.py $comp exp=train_swin_global name="$(date +%F)-local_train_swin_global_sched" batch_size=8 swin.embed_dim=48 scheduler=True wandb.mode=online # testing=True

# # Swin Transformer
# ./.python-greene submitit_hydra.py $comp exp=train_swin name="$(date +%F)-test_train_swin" batch_size=32 swin.embed_dim=24 testing=True

###########################################################################################
# UNET

# ConvNext Inverted UNet Global
# ./.python-greene submitit_hydra.py $comp exp=train_unet_global name="$(date +%F)-test_train_convnextinvunet_global" batch_size=8 scheduler=True unet.encoder.n_channels=[180,90,45] unet.decoder.n_channels=[45,90,180] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block testing=True

# Simple UNet - No scheduler
# ./.python-greene submitit_hydra.py $comp exp=train_unet name="$(date +%F)-test_train_unet_nosched" batch_size=16 unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] +exp/unet/modules/activations@model.encoder.conv_block.activation=capped_leaky_relu +exp/unet/modules/activations@model.decoder.conv_block.activation=capped_leaky_relu testing=True

# Simple UNet - with scheduler
# ./.python-greene submitit_hydra.py $comp exp=train_unet name="$(date +%F)-test_train_unet_withsched" batch_size=16 scheduler=True unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] +exp/unet/modules/activations@model.encoder.conv_block.activation=capped_leaky_relu +exp/unet/modules/activations@model.decoder.conv_block.activation=capped_leaky_relu testing=True

# Simple UNet - with dil
# ./.python-greene submitit_hydra.py $comp exp=train_unet name="$(date +%F)-test_train_unet_dil" batch_size=16 scheduler=True unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/activations@model.encoder.conv_block.activation=capped_leaky_relu +exp/unet/modules/activations@model.decoder.conv_block.activation=capped_leaky_relu testing=True

# ConvNext UNet
# ./.python-greene submitit_hydra.py $comp exp=train_unet name="$(date +%F)-test_train_convnextunet" batch_size=16 scheduler=True unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block testing=True

# ConvNext Inverted UNet
# ./.python-greene submitit_hydra.py $comp exp=train_unet name="$(date +%F)-test_train_convnextinvunet" batch_size=8 scheduler=True unet.encoder.n_channels=[360,180,90] unet.decoder.n_channels=[90,180,360] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block testing=True

# Original ConvNext Inverted UNet
# ./.python-greene submitit_hydra.py $comp exp=train_unet name="$(date +%F)-test_train_orgconvnextinvunet" batch_size=8 scheduler=True unet.encoder.n_channels=[180,90,45] unet.decoder.n_channels=[45,90,180] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block_orig +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block_orig testing=True

# ConvNext Inverted UNet - Absolute Pred
# ./.python-greene submitit_hydra.py $comp exp=train_unet name="$(date +%F)-test_train_convnextinvunet" batch_size=16 scheduler=True unet.pred_residuals=False unet.encoder.n_channels=[180,90,45] unet.decoder.n_channels=[45,90,180] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block testing=True

# ConvNext Inverted UNet - KE loss
# ./.python-greene submitit_hydra.py $comp exp=train_unet name="$(date +%F)-test_train_convnextinvunet_ke" loss=mse_ke batch_size=16 scheduler=True unet.encoder.n_channels=[180,90,45] unet.decoder.n_channels=[45,90,180] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block testing=True
