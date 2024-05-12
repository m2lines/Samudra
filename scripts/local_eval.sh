#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

###########################################################################################
# Global_1 Train - Global_1 Eval

# 1. AdamUNet Global
# ./.python-greene submitit_hydra.py $comp exp=eval_adamunet_global network="Foundation Adam UNet Train1Eval1" name="$(date +%F)-foundation-eval_adamunet_g1_g1" train_region=global_1 region=global_1 ckpt_path=''

# 2. ConvNext UNet Global
# ./.python-greene submitit_hydra.py $comp exp=eval_unet_global model_name_replace="Foundation ConvNext UNet" network="ConvNext UNet Train1Eval1" name="$(date +%F)-foundation-eval_convnextunet_g1_g1" train_region=global_1 region=global_1 ckpt_path=''

# 3. Swin Global
./.python-greene submitit_hydra.py $comp exp=eval_swin_global network="Foundation Swin Train1Eval1 Epoch100" name="$(date +%F)-foundation-eval-swin100_g1_g1" train_region=global_1 region=global_1 ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-05-11-foundation_train_swin_global_1/swin/saved_nets/swin_epoch_100_steps_4_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt'


###########################################################################################
# Global_1 Train - Global_2x Eval

# 1. AdamUNet Global
# ./.python-greene submitit_hydra.py $comp exp=eval_adamunet_global network="Foundation Adam UNet Train1Eval2x" name="$(date +%F)-foundation-eval_adamunet_g1_g2x" train_region=global_1 region=global_2x ckpt_path=''

# 2. ConvNext UNet Global
# ./.python-greene submitit_hydra.py $comp exp=eval_unet_global model_name_replace="ConvNext UNet" network="Foundation ConvNext UNet Train1Eval2x" name="$(date +%F)-foundation-eval_convnextunet_g1_g2x" train_region=global_1 region=global_2x ckpt_path=''

# 3. Swin Global
# ./.python-greene submitit_hydra.py $comp exp=eval_swin_global network="Foundation Swin Train1Eval2x" name="$(date +%F)-foundation-eval_swin_g1_g2x" train_region=global_1 region=global_2x ckpt_path=''

###########################################################################################
# Global_2x Train - Global_2x Eval

# 1. AdamUNet Global
# ./.python-greene submitit_hydra.py $comp exp=eval_adamunet_global network="Foundation Adam UNet Train2xEval2x" name="$(date +%F)-foundation-eval_adamunet_g2x_g2x" train_region=global_2x region=global_2x ckpt_path=''

# 2. ConvNext UNet Global
# ./.python-greene submitit_hydra.py $comp exp=eval_unet_global model_name_replace="ConvNext UNet" network="Foundation ConvNext UNet Train2xEval2x" name="$(date +%F)-foundation-eval_convnextunet_g2x_g2x" train_region=global_2x region=global_2x ckpt_path=''

# 3. Swin Global
# ./.python-greene submitit_hydra.py $comp exp=eval_swin_global network="Foundation Swin Train2xEval2x" name="$(date +%F)-foundation-eval_swin_g2x_g2x" train_region=global_2x region=global_2x ckpt_path=''

###########################################################################################
# Global_2x Train - Global_1 Eval

# 1. AdamUNet Global
# ./.python-greene submitit_hydra.py $comp exp=eval_adamunet_global network="Foundation Adam UNet Train2xEval1" name="$(date +%F)-foundation-eval_adamunet_g2x_g1" train_region=global_2x region=global_1 ckpt_path=''

# 2. ConvNext UNet Global
# ./.python-greene submitit_hydra.py $comp exp=eval_unet_global model_name_replace="ConvNext UNet" network="Foundation ConvNext UNet Train2xEval1" name="$(date +%F)-foundation-eval_convnextunet_g2x_g1" train_region=global_2x region=global_1 ckpt_path=''

# 3. Swin Global
# ./.python-greene submitit_hydra.py $comp exp=eval_swin_global network="Foundation Swin Train2xEval1" name="$(date +%F)-foundation-eval_swin_g2x_g1" train_region=global_2x region=global_1 ckpt_path=''