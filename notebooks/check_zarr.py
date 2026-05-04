#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

# coding: utf-8

# In[1]:


import xarray as xr


# In[2]:


pred = xr.open_zarr(
    "/pscratch/sd/s/suryad/Ocean_Emulator/Preds/2024-09-25_ConvNextUNetTrain3Dv021Eval3DhfdsanomsNetZeroHf1975Epochs70Epoch55Years100_10repeat_2002_Train_global_3D_Test_global_3D_all_N_train_0_Lateral_Data_025_no_smooth/Pred_lateral_Fast_Data_025_global_3D_all_N_samples_0_rand_seed_1.zarr"
)
pred


# In[7]:


pred = xr.open_zarr(
    "/pscratch/sd/s/suryad/Ocean_Emulator/Preds/2024-09-25_ConvNextUNetTrain3Dv021Eval3DhfdsanomsNetZeroHfTempOnly1975Epochs70Epoch55Years100_10repeat_2002_Train_global_3D_Test_global_3D_all_N_train_0_Lateral_Data_025_no_smooth/Pred_lateral_Fast_Data_025_global_3D_all_N_samples_0_rand_seed_1.zarr"
)
pred


# In[4]:


pred = xr.open_zarr(
    "/pscratch/sd/s/suryad/Ocean_Emulator/Preds/2024-09-25_ConvNextUNetTrain3Dv021Eval3DhfdsanomsNetZeroHf1975Epochs70Epoch55Years100_10repeat_1998_Train_global_3D_Test_global_3D_all_N_train_0_Lateral_Data_025_no_smooth/Pred_lateral_Fast_Data_025_global_3D_all_N_samples_0_rand_seed_1.zarr"
)
pred


# In[ ]:


# In[ ]:
