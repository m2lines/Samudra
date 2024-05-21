#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

###########################################################################################
# Global_1 Train - Global_1 Eval

# 1. AdamUNet Global
# ./.python-greene submitit_hydra.py $comp exp=eval_adamunet_global model_name_replace="Foundation Adam UNet" network="Foundation Adam UNet Train1Eval1" name="$(date +%F)-foundation-eval_adamunet_g1_g1" train_region=global_1 region=global_1 ckpt_path='/scratch/sg7761/m2lines/Ocean_Emulator/train/2024-05-13-foundation_train_adamunet_global_1/adamunetseed/saved_nets/adamunet_best_steps_4_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt'

# 2. ConvNext UNet Global
# ./.python-greene submitit_hydra.py $comp exp=eval_unet_global model_name_replace="Foundation ConvNext UNet" network="Foundation ConvNext UNet Train1Eval1" name="$(date +%F)-foundation-eval_convnextunet_g1_g1" train_region=global_1 region=global_1 ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-05-11-foundation_train_convnextunet_global_1/next/saved_nets/convnextunet_best_steps_4_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt'

# 3. Swin Global
# ./.python-greene submitit_hydra.py $comp exp=eval_swin_global network="Foundation Swin Train1Eval1" name="$(date +%F)-foundation-eval-swin_g1_g1" train_region=global_1 region=global_1 swin.embed_dim=60 exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-05-11-foundation_train_swintrans60_global_1/swintrans60/saved_nets/swin_best_steps_4_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt'


###########################################################################################
# Global_1 Train - Global_2x Eval

# 1. AdamUNet Global
# ./.python-greene submitit_hydra.py $comp exp=eval_adamunet_global network="Foundation Adam UNet Train1Eval2x" name="$(date +%F)-foundation-eval_adamunet_g1_g2x" train_region=global_1 region=global_2x ckpt_path='/scratch/sg7761/m2lines/Ocean_Emulator/train/2024-05-13-foundation_train_adamunet_global_1/adamunetseed/saved_nets/adamunet_best_steps_4_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt'

# 2. ConvNext UNet Global
# ./.python-greene submitit_hydra.py $comp exp=eval_unet_global model_name_replace="Foundation ConvNext UNet" network="Foundation ConvNext UNet Train1Eval2x" name="$(date +%F)-foundation-eval_convnextunet_g1_g2x" train_region=global_1 region=global_2x ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-05-11-foundation_train_convnextunet_global_1/next/saved_nets/convnextunet_best_steps_4_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt'

# 3. Swin Global
# ./.python-greene submitit_hydra.py $comp exp=eval_swin_global network="Foundation Swin Train1Eval2x" name="$(date +%F)-foundation-eval_swin_g1_g2x" train_region=global_1 region=global_2x swin.embed_dim=60 exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-05-11-foundation_train_swintrans60_global_1/swintrans60/saved_nets/swin_best_steps_4_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt'


###########################################################################################
# Global_2x Train - Global_2x Eval

# 1. AdamUNet Global
# ./.python-greene submitit_hydra.py $comp exp=eval_adamunet_global network="Foundation Adam UNet Train2xEval2x" name="$(date +%F)-foundation-eval_adamunet_g2x_g2x" train_region=global_2x region=global_2x ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-05-13-foundation_train_adamunet_global_2x/adam/saved_nets/adamunet_best_steps_4_global_2x_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt'

# 2. ConvNext UNet Global
# ./.python-greene submitit_hydra.py $comp exp=eval_unet_global model_name_replace="Foundation ConvNext UNet" network="Foundation ConvNext UNet Train2xEval2x" name="$(date +%F)-foundation-eval_convnextunet_g2x_g2x" train_region=global_2x region=global_2x ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-05-13-foundation_train_convnextunet_global_2x/conv/saved_nets/convnextunet_best_steps_4_global_2x_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt'

# 3. Swin Global
# ./.python-greene submitit_hydra.py $comp exp=eval_swin_global network="Foundation Swin Train2xEval2x" name="$(date +%F)-foundation-eval_swin_g2x_g2x" train_region=global_2x region=global_2x swin.embed_dim=60 exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-05-20-foundation_train_swin60_global_2x/swin2x/saved_nets/swin_best_steps_4_global_2x_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt'

###########################################################################################
# Global_2x Train - Global_1 Eval

# 1. AdamUNet Global
# ./.python-greene submitit_hydra.py $comp exp=eval_adamunet_global network="Foundation Adam UNet Train2xEval1" name="$(date +%F)-foundation-eval_adamunet_g2x_g1" train_region=global_2x region=global_1 ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-05-13-foundation_train_adamunet_global_2x/adam/saved_nets/adamunet_best_steps_4_global_2x_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt'

# 2. ConvNext UNet Global
# ./.python-greene submitit_hydra.py $comp exp=eval_unet_global model_name_replace="Foundation ConvNext UNet" network="Foundation ConvNext UNet Train2xEval1" name="$(date +%F)-foundation-eval_convnextunet_g2x_g1" train_region=global_2x region=global_1 ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-05-13-foundation_train_convnextunet_global_2x/conv/saved_nets/convnextunet_best_steps_4_global_2x_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt'

# 3. Swin Global
./.python-greene submitit_hydra.py $comp exp=eval_swin_global network="Foundation Swin Train2xEval1" name="$(date +%F)-foundation-eval_swin_g2x_g1" train_region=global_2x region=global_1 swin.embed_dim=60 exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample ckpt_path='/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-05-20-foundation_train_swin60_global_2x/swin2x/saved_nets/swin_best_steps_4_global_2x_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt'



###########################################################################################
# Global_1_2x Train - Global_4x Eval

# 1. AdamUNet Global
# ./.python-greene submitit_hydra.py $comp exp=eval_adamunet_global model_name_replace="Foundation Adam UNet" network="Foundation Adam UNet Train12xEval4x" name="$(date +%F)-foundation-eval_adamunet_g1_2x_g4x" train_region=combined_global_1 region=global_4x N_samples=0 N_val=0 N_test=2000 ckpt_path='/scratch/sg7761/m2lines/Ocean_Emulator/train/2024-05-13-foundation_train_adamunet_global_1_2x/adam/saved_nets/adamunet_best_steps_4_global_1_2x_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt'

# 2. ConvNext UNet Global
# ./.python-greene submitit_hydra.py $comp exp=eval_unet_global model_name_replace="Foundation ConvNext UNet" network="Foundation ConvNext UNet Train12xEval4x" name="$(date +%F)-foundation-eval_convnextunet_g1_2x_g4x" train_region=combined_global_1 region=global_4x N_samples=0 N_val=0 N_test=2000 ckpt_path='/scratch/sg7761/m2lines/Ocean_Emulator/train/2024-05-13-foundation_train_convnextunet_global_1_2x/convnext/saved_nets/convnextunet_best_steps_4_global_1_2x_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt' 

# 3. Swin Global
# ./.python-greene submitit_hydra.py $comp exp=eval_swin_global model_name_replace="Foundation Swin" network="Foundation Swin Train12xEval4x" name="$(date +%F)-foundation-eval-swin_g1_2x_g4x" train_region=combined_global_1 region=global_4x swin.embed_dim=60 exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample N_samples=0 N_val=0 N_test=2000 ckpt_path='/scratch/sg7761/m2lines/Ocean_Emulator/train/2024-05-13-foundation_train_swin_global_1_2x/foundationswin/saved_nets/swin_best_steps_4_global_1_2x_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt' 

# Swin old
# ./.python-greene submitit_hydra.py $comp exp=eval_swin_old_global model_name_replace="Swin" network="Swin48 Train12xEval4x" name="$(date +%F)-foundation-eval-swin48_g1_2x_g4x" train_region=combined_global_1 region=global_4x swin.embed_dim=48 N_samples=0 N_val=0 N_test=2000 ckpt_path='/scratch/sg7761/m2lines/Ocean_Emulator/train/2024-05-13-train_swin48_global_1_2x/swin/saved_nets/swin_best_steps_4_global_1_2x_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt' 