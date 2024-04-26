#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

###########################################################################################
# SWIN

# local eval sched 48
# ./.python-greene submitit_hydra.py $comp exp=eval_swin swin.embed_dim=48 network="Swin" name="$(date +%F)-eval_swin_test" ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train_swin/2024-04-08-train_swin_bs16_emb48/emb48/saved_nets/swin_best_steps_8_Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt'

# local eval global
# ./.python-greene submitit_hydra.py $comp exp=eval_swin_global swin.embed_dim=48 network="Swin" name="$(date +%F)-eval_swin_global" ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-04-18-train_swin_global/swin/saved_nets/swin_best_steps_4_global_21_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt'

###########################################################################################
# UNET

# Basic UNet
# ./.python-greene submitit_hydra.py $comp exp=eval_unet network="Basic UNet" name="$(date +%F)-eval_gulfstream" ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-04-18-train_basicunet/basicunet/saved_nets/unet_best_steps_8_Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt' unet.encoder.n_channels=[180,360,720] unet.decoder.n_channels=[720,360,180]

# Basic UNet - with dil
# ./.python-greene submitit_hydra.py $comp exp=eval_unet ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-04-18-train_basicunet_dil/dil/saved_nets/unet_best_steps_8_Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt' network="Basic UNet + dil" name="$(date +%F)-eval_basicunet_dil" unet.encoder.n_channels=[180,360,720] unet.decoder.n_channels=[720,360,180] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1]

# ConvNext UNet
# ./.python-greene submitit_hydra.py $comp exp=eval_unet network="ConvNext UNet + dil" name="$(date +%F)-eval_convnextunet" ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-04-18-train_convnextunet/convnext/saved_nets/unet_best_steps_8_Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt' unet.encoder.n_channels=[45,90,180] unet.decoder.n_channels=[180,90,45] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block

# Original ConvNext UNet
# ./.python-greene submitit_hydra.py $comp exp=eval_unet network="OrgConvNext UNet" name="$(date +%F)-eval_unet_orgconvnextunet" ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-04-18-train_orgconvnextunet/org/saved_nets/unet_best_steps_8_Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt' unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block_orig exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block_orig

# ConvNext Inverted UNet
# ./.python-greene submitit_hydra.py $comp exp=eval_unet network="ConvNext Inv UNet + dil" name="$(date +%F)-eval_unet_convnextinvunet" ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-04-18-train_convnextinvunet/convnextinv/saved_nets/unet_best_steps_8_Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt' unet.encoder.n_channels=[90,45,23] unet.decoder.n_channels=[23,45,90] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block

# Basic UNet Global
./.python-greene submitit_hydra.py $comp exp=eval_unet_global network="Basic UNet" name="$(date +%F)-eval_global" ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-04-19-train_basicunet_global/basic/saved_nets/unet_best_steps_4_global_21_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt' unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90]

# ConvNext UNet Global
# ./.python-greene submitit_hydra.py $comp exp=eval_unet_global network="ConvNext UNet + dil" name="$(date +%F)-eval_convnextunet_global" ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-04-19-train_convnextunet_global_bs4_6hrs/convnext/saved_nets/unet_best_steps_4_global_21_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt' unet.encoder.n_channels=[45,90,180] unet.decoder.n_channels=[180,90,45] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] exp/unet/modules/blocks@unet.encoder.conv_block=conv_next_block exp/unet/modules/blocks@unet.decoder.conv_block=conv_next_block
