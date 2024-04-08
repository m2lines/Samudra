#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# local eval no sched
./.python-greene submitit_hydra.py $comp exp=eval_swin swin.embed_dim=24 ckpt_path='' network="Swin No sched" name="$(date +%F)-eval_swin_nosched"

# local eval sched
# ./.python-greene submitit_hydra.py $comp exp=eval_swin swin.embed_dim=24 ckpt_path='' network="Swin sched" name="$(date +%F)-eval_swin"

# local eval sched 48
# ./.python-greene submitit_hydra.py $comp exp=eval_swin swin.embed_dim=48 ckpt_path='' network="Swin sched 48" name="$(date +%F)-eval_swin_48"

# Simple UNet - No scheduler
# ./.python-greene submitit_hydra.py $comp exp=eval_unet ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train_unet/2024-04-02-train_unet_nosched/nosched/saved_nets/unet_best_steps_8_Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt' network="UNet" name="$(date +%F)-eval_unet_nosched" unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] +exp/unet/modules/activations@model.encoder.conv_block.activation=capped_leaky_relu +exp/unet/modules/activations@model.decoder.conv_block.activation=capped_leaky_relu

# Simple UNet - with scheduler
# ./.python-greene submitit_hydra.py $comp exp=eval_unet ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train_unet/2024-04-02-train_unet_withsched/withsched/saved_nets/unet_best_steps_8_Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt' network="UNet + sched" name="$(date +%F)-eval_unet_withsched" unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] +exp/unet/modules/activations@model.encoder.conv_block.activation=capped_leaky_relu +exp/unet/modules/activations@model.decoder.conv_block.activation=capped_leaky_relu

# Simple UNet - with dil
# ./.python-greene submitit_hydra.py $comp exp=eval_unet ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train_unet/2024-04-02-train_unet_dil/dil/saved_nets/unet_best_steps_8_Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt' network="UNet + sched + dil" name="$(date +%F)-eval_unet_dil" unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/activations@model.encoder.conv_block.activation=capped_leaky_relu +exp/unet/modules/activations@model.decoder.conv_block.activation=capped_leaky_relu

# ConvNext UNet
# ./.python-greene submitit_hydra.py $comp exp=eval_unet ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train_unet/2024-04-02-train_convnextunet/convnext/saved_nets/unet_best_steps_8_Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt' network="ConvNext UNet + sched + dil" name="$(date +%F)-eval_convnextunet" unet.encoder.n_channels=[90,180,360] unet.decoder.n_channels=[360,180,90] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block

# ConvNext Inverted UNet
# ./.python-greene submitit_hydra.py $comp exp=eval_unet ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train_unet/2024-04-02-train_convnextinvunet/convnextinv/saved_nets/unet_best_steps_8_Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt' network="ConvNext Inverted UNet + sched + dil" name="$(date +%F)-eval_unet_convnextinvunet" unet.encoder.n_channels=[180,90,45] unet.decoder.n_channels=[45,90,180] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block

# ConvNext Inverted UNet
# ./.python-greene submitit_hydra.py $comp exp=eval_unet ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train_unet/2024-04-06-train_convnextinvunet_large/large/saved_nets/unet_epoch_20_steps_8_Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt' network="cnextLarge_20" name="$(date +%F)-eval_unet_convnextinvunet_Large20" unet.encoder.n_channels=[360,180,90] unet.decoder.n_channels=[90,180,360] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block

# ConvNext Inverted UNet Absolute Pred
# ./.python-greene submitit_hydra.py $comp exp=eval_unet ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train_unet/2024-04-04-train_convnextinvunet_abs_pred/abs/saved_nets/unet_best_steps_8_Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt' network="CNextInv_sched_dil_pred_abs" name="$(date +%F)-eval_unet_convnextinvunet_predabs" unet.pred_residuals=False unet.encoder.n_channels=[180,90,45] unet.decoder.n_channels=[45,90,180] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block

# ConvNext Inverted UNet MSE_KE loss
# ./.python-greene submitit_hydra.py $comp exp=eval_unet ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train_unet/2024-04-04-train_convnextinvunet_keloss/ke/saved_nets/unet_best_steps_8_Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt' network="CNextInv_sched_dil_ke" name="$(date +%F)-eval_unet_convnextinvunet_ke" loss=mse_ke unet.encoder.n_channels=[180,90,45] unet.decoder.n_channels=[45,90,180] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block

# ConvNext Inverted UNet MSE_KE loss Absolute Pred
# ./.python-greene submitit_hydra.py $comp exp=eval_unet ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train_unet/2024-04-04-train_convnextinvunet_abs_pred_keloss/abs_ke/saved_nets/unet_best_steps_8_Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth.pt' network="CNextInv_sched_dil_pred_abs_ke" name="$(date +%F)-eval_unet_convnextinvunet_ke_predabs" unet.pred_residuals=False loss=mse_ke unet.encoder.n_channels=[180,90,45] unet.decoder.n_channels=[45,90,180] unet.encoder.dilations=[1,2,4] unet.decoder.dilations=[4,2,1] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block


# Doesnt work yet
#
# ./.python-greene submitit_hydra.py $comp exp=eval_recunet name="$(date +%F)-eval_convnextinvrecunet_nodil_large" unet.encoder.n_channels=[360,180,90] unet.decoder.n_channels=[90,180,360] +exp/unet/modules/blocks@model.encoder.conv_block=conv_next_block +exp/unet/modules/blocks@model.decoder.conv_block=conv_next_block



# ./.python-greene submitit_hydra.py $comp exp=eval name="$(date +%F)-eval_test"

# ./.python-greene submitit_hydra.py $comp exp=eval_recunet name="$(date +%F)-eval_recunet_119"

# ./.python-greene submitit_hydra.py $comp exp=eval_norecunet name="$(date +%F)-eval_norecunet_on_vit_ext"
