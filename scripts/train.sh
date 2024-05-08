#!/bin/bash


###########################################################################################
# Global_1 Training


# 1. Basic UNet Global
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet_global wandb.mode=online name="$(date +%F)-train_basicunet_global_1" region=global_1 batch_size=16 scheduler=True unet.encoder.n_channels=[64,128,256,512] unet.encoder.n_layers=[2,2,2,2] unet.encoder.dilations=[1,1,1,1] unet.decoder.n_channels=[512,256,128,64] unet.decoder.n_layers=[2,2,2,2] unet.decoder.dilations=[1,1,1,1] exp/unet/modules/blocks@unet.decoder.up_sampling_block=bilinear_upsample

# 2. ConvNext UNet Global
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet_global wandb.mode=online name="$(date +%F)-train_convnextunet_global_1" region=global_1 batch_size=8 scheduler=True unet.encoder.n_channels=[24,45,90,180] unet.decoder.n_channels=[180,90,45,24] unet.encoder.dilations=[1,2,4,8] unet.decoder.dilations=[8,4,2,1] unet.encoder.n_layers=[1,1,1,1] unet.decoder.n_layers=[1,1,1,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block

# 3. AdamUNet Global
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_adamunet_global wandb.mode=online name="$(date +%F)-train_adamunet_global_1" region=global_1 batch_size=16 scheduler=True

# 4. Swin transformer Global
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_swin_global wandb.mode=online name="$(date +%F)-train_swin_global_1" region=global_1 batch_size=16 swin.embed_dim=24 scheduler=True

# Misc. Original ConvNext UNet Global
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_12hrs exp=train_unet_global wandb.mode=online name="$(date +%F)-train_orgconvnextunet_global_1" region=global_1 batch_size=2 scheduler=True unet.encoder.n_channels=[111,222,444] unet.decoder.n_channels=[444,222,111] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block_orig2 exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block_orig2


###########################################################################################
# Global_2x Training


# 1. Basic UNet Global
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet_global wandb.mode=online name="$(date +%F)-train_basicunet_global_2x" region=global_2x batch_size=16 scheduler=True unet.encoder.n_channels=[64,128,256,512] unet.encoder.n_layers=[2,2,2,2] unet.encoder.dilations=[1,1,1,1] unet.decoder.n_channels=[512,256,128,64] unet.decoder.n_layers=[2,2,2,2] unet.decoder.dilations=[1,1,1,1] exp/unet/modules/blocks@unet.decoder.up_sampling_block=bilinear_upsample

# 3. AdamUNet Global
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_adamunet_global wandb.mode=online name="$(date +%F)-train_adamunet_global_2x" region=global_2x batch_size=16 scheduler=True


###########################################################################################
# Regional Training

# Swin Transformer (40M)
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_swin wandb.mode=online name="$(date +%F)-train_swin" batch_size=16 swin.embed_dim=48 scheduler=True

# Basic UNet
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet wandb.mode=online name="$(date +%F)-train_basicunet"  batch_size=8 scheduler=True unet.encoder.n_channels=[180,360,720] unet.decoder.n_channels=[720,360,180]

# Basic UNet - with dil
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet wandb.mode=online name="$(date +%F)-train_basicunet_dil" batch_size=8 scheduler=True unet.encoder.n_channels=[180,360,720] unet.decoder.n_channels=[720,360,180] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1]

# ConvNext UNet
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet wandb.mode=online name="$(date +%F)-train_convnextunet" batch_size=8 scheduler=True unet.encoder.n_channels=[45,90,180] unet.decoder.n_channels=[180,90,45] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block

# Original ConvNext UNet
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet wandb.mode=online name="$(date +%F)-train_orgconvnextunet_rescon" batch_size=8 scheduler=True unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block_orig exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block_orig

# Original ConvNext UNet + Dil
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet wandb.mode=online name="$(date +%F)-train_orgconvnextunet_rescon_dil" batch_size=8 scheduler=True unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block_orig exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block_orig

# Original ConvNext UNet2 + Dil
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet wandb.mode=online name="$(date +%F)-train_orgconvnextunet2_rescon_dil" batch_size=8 scheduler=True unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block_orig2 exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block_orig2

# Original ConvNext UNet2 + Dil 15M
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet wandb.mode=online name="$(date +%F)-train_orgconvnextunet2_rescon_dil_15M" batch_size=8 scheduler=True unet.encoder.n_channels=[111,222,444] unet.decoder.n_channels=[444,222,111] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block_orig2 exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block_orig2

# ConvNext Inverted UNet
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet wandb.mode=online name="$(date +%F)-train_convnextinvunet" batch_size=8 scheduler=True unet.encoder.n_channels=[90,45,23] unet.decoder.n_channels=[23,45,90] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block

# Original ConvNext Inverted UNet
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train_unet wandb.mode=online name="$(date +%F)-train_orgconvnextinvunet" batch_size=8 scheduler=True unet.encoder.n_channels=[180,90,45] unet.decoder.n_channels=[45,90,180] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block_orig exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block_orig
