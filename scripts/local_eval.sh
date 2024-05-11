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
# ./.python-greene submitit_hydra.py $comp exp=eval_swin_global network="Foundation Swin Train1Eval1" name="$(date +%F)-foundation-eval_swin_g1_g1" train_region=global_1 region=global_1 ckpt_path=''


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