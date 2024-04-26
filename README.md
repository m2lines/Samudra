# Ocean Emulator

Ocean modeling serves as an indispensable tool in marine science, enabling the simulation of the ocean's various processes. Although research using both numerical and machine learning methods addresses sub-seasonal and short-term predictions, given the many difficulties of ocean emulation, there is a need for reliable and accurate models over long-term periods in a non-idealized setting.


We explore UNet and Transformer architectures to predict Ocean states in the Gulf Stream Region and Global Region respectively. The ocean state (u, v, T) represents zonal velocity, meridional velocity, and temperature, respectively, in the surface layer.

## Gulf Stream Region Predictions
### Zonal Velocity
![Gulf Stream Predictions](assets/eval_gulfstream/Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth_Gulf_Stream_Ext_u.gif)

### Meridional Velocity
![Gulf Stream Predictions](assets/eval_gulfstream/Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth_Gulf_Stream_Ext_v.gif)


### Temperature
![Gulf Stream Predictions](assets/eval_gulfstream/Gulf_Stream_Ext_Test_in_um_vm_Tm_ext_tau_u_tau_v_t_ref__outum_vm_Tm_N_train_4000_Lateral_Data_025_no_smooth_Gulf_Stream_Ext_T.gif)




## Global Region Predictions
### Zonal Velocity
![Gulf Stream Predictions](assets/eval_global/global_21_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth_global_21_u.gif)

### Meridional Velocity
![Gulf Stream Predictions](assets/eval_global/global_21_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth_global_21_v.gif)

### Temperature
![Gulf Stream Predictions](assets/eval_global/global_21_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth_global_21_T.gif)

