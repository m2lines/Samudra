#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# local train
# Simple UNet - No scheduler
# ./.python-greene submitit_hydra.py $comp exp=train_unet name="$(date +%F)-test_train_unet_nosched" batch_size=16 unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] +exp/unet/modules/activations@model.encoder.conv_block.activation=capped_leaky_relu +exp/unet/modules/activations@model.decoder.conv_block.activation=capped_leaky_relu

# Simple UNet - with scheduler
# ./.python-greene submitit_hydra.py $comp exp=train_unet name="$(date +%F)-test_train_unet_withsched" batch_size=16 scheduler=True unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] +exp/unet/modules/activations@model.encoder.conv_block.activation=capped_leaky_relu +exp/unet/modules/activations@model.decoder.conv_block.activation=capped_leaky_relu

# Simple UNet - with dil
# ./.python-greene submitit_hydra.py $comp exp=train_unet name="$(date +%F)-test_train_unet_dil" batch_size=16 scheduler=True unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/activations@model.encoder.conv_block.activation=capped_leaky_relu +exp/unet/modules/activations@model.decoder.conv_block.activation=capped_leaky_relu

# ConvNext UNet
# ./.python-greene submitit_hydra.py $comp exp=train_unet name="$(date +%F)-test_train_convnextunet" batch_size=16 scheduler=True unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block

# ConvNext Inverted UNet
# ./.python-greene submitit_hydra.py $comp exp=train_unet name="$(date +%F)-test_train_convnextinvunet" batch_size=16 scheduler=True unet.encoder.n_channels=[180,90,45] unet.decoder.n_channels=[45,90,180] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block

# ConvNext Inverted UNet - Absolute Pred
# ./.python-greene submitit_hydra.py $comp exp=train_unet name="$(date +%F)-test_train_convnextinvunet" batch_size=16 scheduler=True unet.pred_residuals=False unet.encoder.n_channels=[180,90,45] unet.decoder.n_channels=[45,90,180] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block

# ConvNext Inverted UNet - KE loss
# ./.python-greene submitit_hydra.py $comp exp=train_unet name="$(date +%F)-test_train_convnextinvunet_ke" loss=mse_ke batch_size=16 scheduler=True unet.encoder.n_channels=[180,90,45] unet.decoder.n_channels=[45,90,180] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block

# Doesnt work yet
# ConvNext Inverted RecUNet
# ./.python-greene submitit_hydra.py $comp exp=train_recunet name="$(date +%F)-test_train_convnextinvrecunet" batch_size=16 scheduler=True unet.encoder.n_channels=[136,68,34] unet.decoder.n_channels=[34,68,136] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block

# ConvNext Inverted RecUNet No dil
# ./.python-greene submitit_hydra.py $comp exp=train_recunet name="$(date +%F)-test_train_convnextinvrecunet_nodil" batch_size=16 scheduler=True unet.encoder.n_channels=[136,68,34] unet.decoder.n_channels=[34,68,136] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block

# ConvNext Inverted RecUNet No dil - Large
# ./.python-greene submitit_hydra.py $comp exp=train_recunet name="$(date +%F)-test_train_convnextinvrecunet_nodil_large" batch_size=8 scheduler=True unet.encoder.n_channels=[360,180,90] unet.decoder.n_channels=[90,180,360] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block


# ViT
# ./.python-greene submitit_hydra.py $comp exp=train_vit_gulfext3 name="$(date +%F)-train_vit_gulfext3"
