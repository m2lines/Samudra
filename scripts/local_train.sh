#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# local train

###########################################################################################
# Global_1 Training

# 1. Basic UNet Global
# ./.python-greene submitit_hydra.py $comp exp=train_unet_global name="$(date +%F)-test_train_basicunet_global_1" region=global_1 batch_size=16 scheduler=True unet.encoder.n_channels=[64,128,256,512] unet.encoder.n_layers=[2,2,2,2] unet.encoder.dilations=[1,1,1,1] unet.decoder.n_channels=[512,256,128,64] unet.decoder.n_layers=[2,2,2,2] unet.decoder.dilations=[1,1,1,1]

# 2. ConvNext UNet Global
# ./.python-greene submitit_hydra.py $comp exp=train_unet_global name="$(date +%F)-local_train_convnextinvunet_global_1" region=global_1 batch_size=16 scheduler=True unet.encoder.n_channels=[24,45,90,180] unet.decoder.n_channels=[180,90,45,24] unet.encoder.dilations=[1,2,4,8] unet.decoder.dilations=[8,4,2,1] unet.encoder.n_layers=[1,1,1,1] unet.decoder.n_layers=[1,1,1,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block

# 3. AdamUNet Global
# ./.python-greene submitit_hydra.py $comp exp=train_adamunet_global name="$(date +%F)-local_train_adamunet_global_1" region=global_1 batch_size=16 scheduler=True

# 4. Swin Transformer Global
./.python-greene submitit_hydra.py $comp exp=train_swin_global name="$(date +%F)-local_train_swin_global_1" region=global_1 batch_size=16 swin.embed_dim=48 scheduler=True

# Misc. ConvNext original Global
# ./.python-greene submitit_hydra.py $comp exp=train_unet_global testing=true name="$(date +%F)-test_train_orgconvnextunet_global_1" region=global_1 batch_size=3 scheduler=True unet.encoder.n_channels=[111,222,444] unet.decoder.n_channels=[444,222,111] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block_orig2 exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block_orig2

###########################################################################################
# Global_2x Training

# 1. Basic UNet Global
# ./.python-greene submitit_hydra.py $comp exp=train_unet_global testing=true name="$(date +%F)-test_train_basicunet_global_2x" region=global_2x batch_size=8 scheduler=True unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90]

# 2. ConvNext UNet Global
# ./.python-greene submitit_hydra.py $comp exp=train_unet_global testing=true name="$(date +%F)-local_train_convnextinvunet_global_2x" region=global_2x batch_size=4 scheduler=True unet.encoder.n_channels=[45,90,180] unet.decoder.n_channels=[180,90,45] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block

# 3. AdamUNet Global
# ./.python-greene submitit_hydra.py $comp exp=train_adamunet_global testing=true name="$(date +%F)-local_train_adamunet_global_2x" region=global_2x batch_size=8 scheduler=True

# 4. Swin Transformer Global
# ./.python-greene submitit_hydra.py $comp exp=train_swin_global testing=true name="$(date +%F)-local_train_swin_global_2x" region=global_2x batch_size=8 swin.embed_dim=48 scheduler=True

# Misc. ConvNext original Global
# ./.python-greene submitit_hydra.py $comp exp=train_unet_global testing=true name="$(date +%F)-test_train_orgconvnextunet_global_2x" region=global_2x batch_size=3 scheduler=True unet.encoder.n_channels=[111,222,444] unet.decoder.n_channels=[444,222,111] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block_orig2 exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block_orig2


###########################################################################################
# Regional Training

# # Swin Transformer
# ./.python-greene submitit_hydra.py $comp exp=train_swin testing=true name="$(date +%F)-test_train_swin" batch_size=16 swin.embed_dim=48 scheduler=True

# Basic UNet
# ./.python-greene submitit_hydra.py $comp exp=train_unet testing=true name="$(date +%F)-test_train_unet_withsched" batch_size=8 scheduler=True unet.encoder.n_channels=[180,360,720] unet.decoder.n_channels=[720,360,180]

# Basic UNet - with dil
# ./.python-greene submitit_hydra.py $comp exp=train_unet testing=true name="$(date +%F)-test_train_unet_dil" batch_size=8 scheduler=True unet.encoder.n_channels=[180,360,720] unet.decoder.n_channels=[720,360,180] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1]

# ConvNext UNet
# ./.python-greene submitit_hydra.py $comp exp=train_unet testing=true name="$(date +%F)-test_train_convnextunet" batch_size=8 scheduler=True unet.encoder.n_channels=[45,90,180] unet.decoder.n_channels=[180,90,45] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block

# ConvNext original
# ./.python-greene submitit_hydra.py $comp exp=train_unet testing=true name="$(date +%F)-test_train_orgconvnextunet" batch_size=8 scheduler=True unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block_orig exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block_orig

# ConvNext original + dil
# ./.python-greene submitit_hydra.py $comp exp=train_unet testing=true name="$(date +%F)-test_train_orgconvnextunet_dil_moreconvs2" batch_size=8 scheduler=True unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block_orig exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block_orig

# ConvNext original 2 + dil
# ./.python-greene submitit_hydra.py $comp exp=train_unet testing=true name="$(date +%F)-test_train_orgconvnextunet2_dil_100" batch_size=8 scheduler=True unet.encoder.n_channels=[111,222,444] unet.decoder.n_channels=[444,222,111] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block_orig2 exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block_orig2

# ConvNext Inverted UNet
# ./.python-greene submitit_hydra.py $comp exp=train_unet testing=true name="$(date +%F)-test_train_convnextinvunet" batch_size=8 scheduler=True unet.encoder.n_channels=[90,45,23] unet.decoder.n_channels=[23,45,90] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block

# Original ConvNext Inverted UNet
# ./.python-greene submitit_hydra.py $comp exp=train_unet testing=true name="$(date +%F)-test_train_orgconvnextinvunet" batch_size=8 scheduler=True unet.encoder.n_channels=[180,90,45] unet.decoder.n_channels=[45,90,180] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block_orig exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block_orig

