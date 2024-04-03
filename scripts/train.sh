#!/bin/bash

# Simple UNet - No scheduler
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet name="$(date +%F)-train_unet_nosched" batch_size=24 unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] +exp/unet/modules/activations@model.encoder.conv_block.activation=capped_leaky_relu +exp/unet/modules/activations@model.decoder.conv_block.activation=capped_leaky_relu wandb.mode=online

# Simple UNet - with scheduler
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet name="$(date +%F)-train_unet_withsched" batch_size=24 scheduler=True unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] +exp/unet/modules/activations@model.encoder.conv_block.activation=capped_leaky_relu +exp/unet/modules/activations@model.decoder.conv_block.activation=capped_leaky_relu wandb.mode=online

# Simple UNet - with dil
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet name="$(date +%F)-train_unet_dil" batch_size=24 scheduler=True unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/activations@model.encoder.conv_block.activation=capped_leaky_relu +exp/unet/modules/activations@model.decoder.conv_block.activation=capped_leaky_relu wandb.mode=online

# ConvNext UNet
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet name="$(date +%F)-train_convnextunet" batch_size=16 scheduler=True unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block wandb.mode=online

# ConvNext Inverted UNet
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet name="$(date +%F)-train_convnextinvunet" batch_size=16 scheduler=True unet.encoder.n_channels=[180,90,45] unet.decoder.n_channels=[45,90,180] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block wandb.mode=online

# ConvNext Inverted UNet Absolute Pred
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet name="$(date +%F)-train_convnextinvunet_abs_pred" batch_size=16 scheduler=True unet.pred_residuals=False unet.encoder.n_channels=[180,90,45] unet.decoder.n_channels=[45,90,180] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block wandb.mode=online

# ConvNext Inverted UNet MSE_KE loss
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet name="$(date +%F)-train_convnextinvunet_keloss" loss=mse_ke batch_size=16 scheduler=True unet.encoder.n_channels=[180,90,45] unet.decoder.n_channels=[45,90,180] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block wandb.mode=online

# ConvNext Inverted UNet MSE_KE loss Absolute Pred
./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet name="$(date +%F)-train_convnextinvunet_abs_pred_keloss" loss=mse_ke unet.pred_residuals=False batch_size=16 scheduler=True unet.encoder.n_channels=[180,90,45] unet.decoder.n_channels=[45,90,180] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block wandb.mode=online

# ViT
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train name="$(date +%F)-1GPU_train_CosineAnnealWarmLR_1step"
