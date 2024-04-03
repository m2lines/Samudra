#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# local eval
# Simple UNet - No scheduler
# ./.python-greene submitit_hydra.py $comp exp=eval_unet name="$(date +%F)-eval_unet_nosched" unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] +exp/unet/modules/activations@model.encoder.conv_block.activation=capped_leaky_relu +exp/unet/modules/activations@model.decoder.conv_block.activation=capped_leaky_relu

# Simple UNet - with dil
# ./.python-greene submitit_hydra.py $comp exp=eval_unet name="$(date +%F)-eval_unet_dil" unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/activations@model.encoder.conv_block.activation=capped_leaky_relu +exp/unet/modules/activations@model.decoder.conv_block.activation=capped_leaky_relu

# ConvNext Inverted UNet
./.python-greene submitit_hydra.py $comp exp=eval_unet name="$(date +%F)-eval_unet_convnextinvunet" unet.encoder.n_channels=[180,90,45] unet.decoder.n_channels=[45,90,180] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block

# Doesnt work yet
# 
# ./.python-greene submitit_hydra.py $comp exp=eval_recunet name="$(date +%F)-eval_convnextinvrecunet_nodil_large" unet.encoder.n_channels=[360,180,90] unet.decoder.n_channels=[90,180,360] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block



# ./.python-greene submitit_hydra.py $comp exp=eval name="$(date +%F)-eval_test"

# ./.python-greene submitit_hydra.py $comp exp=eval_recunet name="$(date +%F)-eval_recunet_119"

# ./.python-greene submitit_hydra.py $comp exp=eval_norecunet name="$(date +%F)-eval_norecunet_on_vit_ext"
