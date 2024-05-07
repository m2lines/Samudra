#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

###########################################################################################
# Global_1 Train - Global_1 Eval

# 1. Basic UNet Global
# ./.python-greene submitit_hydra.py $comp exp=eval_unet_global network="Basic UNet" name="$(date +%F)-eval_basicunetwetin_g1_g1" train_region=global_1 region=global_1 ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-05-06-train_basicunet_global_1/basic/saved_nets/unet_best_steps_4_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt' unet.encoder.n_channels=[64,128,256,512] unet.encoder.n_layers=[2,2,2,2] unet.encoder.dilations=[1,1,1,1] unet.decoder.n_channels=[512,256,128,64] unet.decoder.n_layers=[2,2,2,2] unet.decoder.dilations=[1,1,1,1] exp/unet/modules/blocks@unet.decoder.up_sampling_block=bilinear_upsample

# 2. ConvNext UNet Global
# ./.python-greene submitit_hydra.py $comp exp=eval_unet_global network="ConvNext UNet" name="$(date +%F)-eval_convnextunet_g1_g1" train_region=global_1 region=global_1 ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-05-04-train_convnextunet_global_1/convnext/saved_nets/unet_best_steps_4_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt' unet.encoder.n_channels=[45,90,180] unet.decoder.n_channels=[180,90,45] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block

# 3. AdamUNet Global
# ./.python-greene submitit_hydra.py $comp exp=eval_adamunet_global network="Adam UNet" name="$(date +%F)-eval_adamunet_g1_g1" train_region=global_1 region=global_1 ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-05-04-train_adamunet_global_1/adam/saved_nets/adamunet_best_steps_4_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt'

# 4. Swin Transformer Global
./.python-greene submitit_hydra.py $comp exp=eval_swin_global network="Swin" name="$(date +%F)-eval_swin_g1_g1" train_region=global_1 region=global_1 swin.embed_dim=24 ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-05-06-train_swin24_global_1/swin24/saved_nets/swin_epoch_60_steps_4_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt'

# ./.python-greene submitit_hydra.py $comp exp=eval_swin_global network="Swin2" name="$(date +%F)-eval_swinbi_g1_g1" train_region=global_1 region=global_1 swin.embed_dim=24 exp/unet/modules/blocks@swin.up_sampling_block=bilinear_upsample ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-05-06-train_swin24_bilinear_global_1/swin_bi/saved_nets/swin_best_steps_4_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt'


###########################################################################################
# Global_1 Train - Global_2x Eval

# Basic UNet Global
# ./.python-greene submitit_hydra.py $comp exp=eval_unet_global network="Basic UNet" name="$(date +%F)-eval_basicunet_g1_g2x" train_region=global_1 region=global_2x ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-05-04-train_basicunet_global_1/basic/saved_nets/unet_best_steps_4_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt' unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90]

###########################################################################################
# Global_2x Train - Global_1 Eval

###########################################################################################
# Global_2x Train - Global_2x Eval





###########################################################################################
# Regional Eval

# local eval sched 48
# ./.python-greene submitit_hydra.py $comp exp=eval_swin swin.embed_dim=48 network="Swin" name="$(date +%F)-eval_swin_test" ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train_swin/2024-04-08-train_swin_bs16_emb48/emb48/saved_nets/swin_best_steps_8_Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt'

# Basic UNet
# ./.python-greene submitit_hydra.py $comp exp=eval_unet network="Basic UNet" name="$(date +%F)-eval_gulfstream" ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-04-18-train_basicunet/basicunet/saved_nets/unet_best_steps_8_Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt' unet.encoder.n_channels=[180,360,720] unet.decoder.n_channels=[720,360,180]

# Basic UNet - with dil
# ./.python-greene submitit_hydra.py $comp exp=eval_unet ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-04-18-train_basicunet_dil/dil/saved_nets/unet_best_steps_8_Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt' network="Basic UNet + dil" name="$(date +%F)-eval_basicunet_dil" unet.encoder.n_channels=[180,360,720] unet.decoder.n_channels=[720,360,180] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1]

# ConvNext UNet
# ./.python-greene submitit_hydra.py $comp exp=eval_unet network="ConvNext UNet + dil" name="$(date +%F)-eval_convnextunet" ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-04-18-train_convnextunet/convnext/saved_nets/unet_best_steps_8_Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt' unet.encoder.n_channels=[45,90,180] unet.decoder.n_channels=[180,90,45] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block

# Original ConvNext UNet
# ./.python-greene submitit_hydra.py $comp exp=eval_unet network="OrgConvNext UNet" name="$(date +%F)-eval_unet_orgconvnextunet" ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-04-26-train_orgconvnextunet_rescon/org/saved_nets/unet_best_steps_8_Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt' unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block_orig exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block_orig

# ConvNext original + dil
# ./.python-greene submitit_hydra.py $comp exp=eval_unet network="OrgConvNext UNet + dil" name="$(date +%F)-eval_unet_orgconvnextunet_dil_moreconvs2" ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-04-26-train_orgconvnextunet_rescon_dil/org/saved_nets/unet_best_steps_8_Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt' unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block_orig exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block_orig

# ConvNext original 2 + dil
# ./.python-greene submitit_hydra.py $comp exp=eval_unet network="OrgConvNext2 UNet + dil" name="$(date +%F)-eval_unet_orgconvnextunet2_dil" ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-04-26-train_orgconvnextunet2_rescon_dil/org/saved_nets/unet_best_steps_8_Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt' unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block_orig2 exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block_orig2

# ConvNext original 2 + dil + 15M
# ./.python-greene submitit_hydra.py $comp exp=eval_unet network="OrgConvNext2 UNet + dil15M" name="$(date +%F)-eval_unet_orgconvnextunet2_dil15M" ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-04-26-train_orgconvnextunet2_rescon_dil_15M/org/saved_nets/unet_best_steps_8_Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt' unet.encoder.n_channels=[111,222,444] unet.decoder.n_channels=[444,222,111] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block_orig2 exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block_orig2

# ConvNext Inverted UNet
# ./.python-greene submitit_hydra.py $comp exp=eval_unet network="ConvNext Inv UNet + dil" name="$(date +%F)-eval_unet_convnextinvunet" ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-04-18-train_convnextinvunet/convnextinv/saved_nets/unet_best_steps_8_Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt' unet.encoder.n_channels=[90,45,23] unet.decoder.n_channels=[23,45,90] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block