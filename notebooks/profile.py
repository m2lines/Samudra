import xarray as xr
import numpy as np
import cftime
import cmocean as cm
import matplotlib.pyplot as plt
import regionmask
from xmip.regionmask import merged_mask
import cartopy.crs as ccrs
import os
import pandas as pd
from pandas import Timestamp
from xarrayutils.plotting import linear_piecewise_scale
from xarrayutils.plotting import box_plot
# %matplotlib inline


# 2002 No warming
# Pred_path_temp = '/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/Preds/2024-09-25_ConvNextUNetTrain3Dv021Eval3DhfdsanomsNetZeroHfTempOnly1975Epochs70Epoch55Years100_10repeat_2002_Train_global_3D_Test_global_3D_all_N_train_0_Lateral_Data_025_no_smooth/Pred_lateral_Fast_Data_025_global_3D_all_N_samples_0_rand_seed_1.zarr' # temp no warming
# label_temp = '2002-2012 - Slow'
# Pred_path_all = '/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/Preds/2024-09-25_ConvNextUNetTrain3Dv021Eval3DhfdsanomsNetZeroHf1975Epochs70Epoch55Years100_10repeat_2002_Train_global_3D_Test_global_3D_all_N_train_0_Lateral_Data_025_no_smooth/Pred_lateral_Fast_Data_025_global_3D_all_N_samples_0_rand_seed_1.zarr' # all no warming
# label_all = '2002-2012 - Slow+Fast'

# No warming
Pred_path_temp = '/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/Preds/2024-09-12_ConvNextUNetTrain3Dv021Eval3DhfdsanomsNetZeroHfTempOnly1975Epochs70Epoch55Years100_10repeat_36_6k_Train_global_3D_Test_global_3D_all_N_train_0_Lateral_Data_025_no_smooth/Pred_lateral_Fast_Data_025_global_3D_all_N_samples_0_rand_seed_1.zarr' # temp no warming
label_temp = 'Thermo'
Pred_path_all = '/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/Preds/2024-09-12_ConvNextUNetTrain3Dv021Eval3DhfdsanomsNetZeroHf1975Epochs70Epoch55Years100_10repeat_36_6k_Train_global_3D_Test_global_3D_all_N_train_0_Lateral_Data_025_no_smooth/Pred_lateral_Fast_Data_025_global_3D_all_N_samples_0_rand_seed_1.zarr' # all no warming
label_all = 'Thermo+Dynamic'

warming = False
if warming:
    suffix = "_warming"
else:
    suffix = ""
    
def post_processor(ds: xr.Dataset, ds_truth: xr.Dataset, ls) -> xr.Dataset:
    """Converts the prediction output to an xarray dataset with the same dimensions/variables as input"""
    # Always run the ds_input_validate in non-deep mode here

    # correct swapped dimensions and warn
    if len(ds.x) == 180 and len(ds.y) == 360:
        ds = ds.rename({"x": "x_i", "y": "y_i"}).rename({"x_i": "y", "y_i": "x"})


    da = ds["__xarray_dataarray_variable__"]
    n_lev = 19
    if set(ls) - {"zos"} == set(["uo", "vo", "thetao", "so"]):
        variables = ["uo", "vo", "thetao", "so"]
    elif set(ls) - {"zos"} == set(["thetao", "so"]):
        variables = ["thetao", "so"]
    elif set(ls) - {"zos"} == set(["uo", "vo"]):
        variables = ["uo", "vo"]
    slices = [slice(i, i + n_lev) for i in range(0, len(variables) * n_lev, n_lev)]
    var_slices = {k: sl for k, sl in zip(variables, slices)}
    variables = {
        k: da.isel(var=sl).rename({"var": "lev"}) for k, sl in var_slices.items()
    }
    variables["zos"] = da.isel(var=-1).squeeze()

    ds_out = xr.Dataset(variables)
    for var in ds_out.data_vars:
        if "lev" in ds_out[var].dims:
            ds_out[var] = ds_out[var].where(ds_truth.wetmask)
        else:
            ds_out[var] = ds_out[var].where(ds_truth.wetmask.isel(lev=0))

    ## attach all coordinates from input
    ds_out = ds_out.assign_coords({co: ds_truth[co] for co in ds_truth.coords})

    return ds_out

levels = 19
emulation_stability=True
smooth = False

# OM4 v0.2.1
ds_input = xr.open_zarr(
    os.path.join("/vast/sd5313/data/m2lines/3D_ocean_data", "OM4_5daily_v0.2.1.zarr")
)

# Smooth the data 
if smooth:
    window = 10
    with ProgressBar():
        ds_input['uo'] = ds_input.uo.rolling(time=window, min_periods=1, center=False).mean().compute()
        ds_input['vo'] = ds_input.vo.rolling(time=window, min_periods=1, center=False).mean().compute()


# our groundtruth is always just a time slice of the training (training is a bad name

if emulation_stability:
    repeats = 10
    ds_groundtruth = ds_input.isel(lev=slice(None, levels))
    ds_groundtruth = ds_groundtruth.sel(time=slice("1990-01-01", "1999-12-31"))
    new_time = pd.date_range(start=str(ds_groundtruth.time[0].values), periods=repeats * len(ds_groundtruth.time), freq="5D")
    ds_groundtruth = xr.concat([ds_groundtruth] * repeats, dim="time")
    ds_groundtruth['time'] = new_time
    ds_groundtruth = ds_groundtruth.isel(time=slice(3, 7303))

else:
    ds_groundtruth = ds_input.isel(time=slice(2903, 3503)).isel(lev=slice(None, levels))

ls_all = ['uo', 'vo', 'thetao', 'so', 'zos'] #['uo', 'vo', 'thetao', 'so', 'zos'], ['thetao', 'so', 'zos']
ls_temp = ['thetao', 'so', 'zos']
output_folder_all = Pred_path_all.split("/")[-2].split("_Train")[0]
output_path_all = os.path.join("./temp", output_folder_all)
output_folder_temp = Pred_path_temp.split("/")[-2].split("_Train")[0]
output_path_temp = os.path.join("./temp", output_folder_temp)

output_path = "../outputs/" + label_temp+"_"+label_all
print("Using Output Folder : ", output_path)
if not os.path.isdir(os.path.join(output_path)):
    os.makedirs(os.path.join(output_path))


ds_prediction_raw_all = xr.open_zarr(Pred_path_all).isel(time=slice(None,7200))
ds_prediction_raw_temp = xr.open_zarr(Pred_path_temp).isel(time=slice(None,7200))
ds_groundtruth = ds_groundtruth.isel(time=slice(0, ds_prediction_raw_temp.time.size))

# ds_prediction_all = post_processor(
#     ds_prediction_raw_all, ds_groundtruth, ls_all
# )
ds_prediction_all = post_processor(
    ds_prediction_raw_all, ds_groundtruth.isel(time = slice(0,ds_prediction_raw_temp.time.size)), ls_all
)

ds_prediction_temp = post_processor(
    ds_prediction_raw_temp, ds_groundtruth.isel(time = slice(0,ds_prediction_raw_temp.time.size)), ls_temp
)


# Run the test to make sure the output is formatted correctly
ds_prediction_all = ds_prediction_all.transpose('time','lev',...)
ds_prediction_temp = ds_prediction_temp.transpose('time','lev',...)


ds_prediction_temp = ds_prediction_temp.transpose('time','lev',...)
ds_prediction_all = ds_prediction_all.transpose('time','lev',...)

ds_prediction_all['y']  = ds_prediction_all.y.assign_attrs(long_name='latitude', units = r"${^o}$")
ds_prediction_all['x']  = ds_prediction_all.x.assign_attrs(long_name='longitude', units = r"${^o}$")
ds_prediction_all['thetao'] = ds_prediction_all['thetao'].assign_attrs(long_name = r'$\theta_O$', units = r"${^oC}$")

ds_prediction_temp['y']  = ds_prediction_temp.y.assign_attrs(long_name='latitude', units = r"${^o}$")
ds_prediction_temp['x']  = ds_prediction_temp.x.assign_attrs(long_name='longitude')
ds_prediction_temp['thetao'] = ds_prediction_temp['thetao'].assign_attrs(long_name = r'$\theta_O$', units = r"${^oC}$")


# Time slice 
ds_groundtruth = ds_groundtruth.sel(time=slice("1990-01-01", "1999-12-31"))
ds_prediction_all = ds_prediction_all.sel(time=slice("1990-01-01", "1999-12-31"))
ds_prediction_temp = ds_prediction_temp.sel(time=slice("1990-01-01", "1999-12-31"))

color_1 = '#DE3A41'
color_2 = '#277DC7'

clist = ["#D7191C","#DE7400","#00BD8E","#3300EA"]
color_2 = '#ff807a'
color_1 = '#1e8685'

# Saving depth profiles
# OM4
da_temp = ds_groundtruth['thetao']
section_mask = np.isnan(da_temp).all('x').isel(time=0)
da_temp_int_x = da_temp.weighted(ds_prediction_temp['areacello']).mean('x').mean('time')
temp_pred = da_temp_int_x.where(~section_mask)
temp_pred = temp_pred.rename(r'$\theta_O$').assign_attrs(units=r'\degree C')
temp_pred['y'] = temp_pred.y.assign_attrs(long_name='latitude', units=r'$\degree$')
temp_pred['lev'] = temp_pred.lev.assign_attrs(long_name='depth', units='m')
om4_temp_profile = temp_pred.compute()
# om4_temp_profile.to_zarr("../Figures/om4_temp_profile", mode="w", safe_chunks=False)
# del om4_temp_profile

# Thermo
da_temp = ds_prediction_temp['thetao']
section_mask = np.isnan(da_temp).all('x').isel(time=0)
da_temp_int_x = da_temp.weighted(ds_prediction_temp['areacello']).mean('x').mean('time')
temp_pred = da_temp_int_x.where(~section_mask)
temp_pred = temp_pred.rename(r'$\theta_O$').assign_attrs(units=r'\degree C')
temp_pred['y'] = temp_pred.y.assign_attrs(long_name='latitude', units=r'$\degree$')
temp_pred['lev'] = temp_pred.lev.assign_attrs(long_name='depth', units='m')
thermo_temp_profile = temp_pred.compute()#.chunk({'y': 10, 'lev': 10}) 
# thermo_temp_profile.to_zarr("../Figures/thermo_temp_profile", mode="w", safe_chunks=False)
# del thermo_temp_profile


# # Thermo+Dynamic 
da_temp = ds_prediction_all['thetao']
section_mask = np.isnan(da_temp).all('x').isel(time=0)
da_temp_int_x = da_temp.weighted(ds_prediction_temp['areacello']).mean('x').mean('time')
temp_pred = da_temp_int_x.where(~section_mask)
temp_pred = temp_pred.rename(r'$\theta_O$').assign_attrs(units=r'\degree C')
temp_pred['y'] = temp_pred.y.assign_attrs(long_name='latitude', units=r'$\degree$')
temp_pred['lev'] = temp_pred.lev.assign_attrs(long_name='depth', units='m')
thermodynamic_temp_profile = temp_pred.compute()#.chunk({'y': 10, 'lev': 10}) 
# thermodynamic_temp_profile.to_zarr("../Figures/thermodynamic_temp_profile", mode="w", safe_chunks=False)
# del thermodynamic_temp_profile

import cmocean as cm
import matplotlib.pyplot as plt
new_cmap = cm.cm.balance 
new_cmap.set_bad('grey', .6)
plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(1, 3, figsize=(16, 3), gridspec_kw={'width_ratios': [1, 1, 1], 
                                    'height_ratios': [1],
                                    'wspace': 0.02, 'hspace': 0.5})
vmin, vmax = -0.5, 0.5

i = 0

im = (thermo_temp_profile-om4_temp_profile).plot(ax=ax[i], cmap=new_cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
ax[i].invert_yaxis()
ax[i].set_title(f"Thermo - OM4", fontsize=14)
linear_piecewise_scale(1000, 5, ax=ax[i])
ax[i].axhline(1000, color='0.5', ls='--')
ax[i].set_yticks([0, 250, 500, 750, 1000, 3000, 5000])

i=i+1
im = (thermodynamic_temp_profile-om4_temp_profile).plot(ax=ax[i], cmap=new_cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
ax[i].invert_yaxis()
ax[i].set_title(f"Thermo+Dynamic - OM4", fontsize=14)
linear_piecewise_scale(1000, 5, ax=ax[i])
ax[i].axhline(1000, color='0.5', ls='--')
# ax[i].set_yticks([0, 250, 500, 750, 1000, 3000, 5000])
ax[i].set_yticks([])
ax[i].set_ylabel("")

i=i+1
im = (thermo_temp_profile-thermodynamic_temp_profile).plot(ax=ax[i], cmap=new_cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
ax[i].invert_yaxis()
ax[i].set_title(f"Thermo - Thermo+Dynamic ", fontsize=14)
linear_piecewise_scale(1000, 5, ax=ax[i])
ax[i].axhline(1000, color='0.5', ls='--')
# ax[i].set_yticks([0, 250, 500, 750, 1000, 3000, 5000])
ax[i].set_yticks([])
ax[i].set_ylabel("")
cbar = fig.colorbar(im, ax=ax[:], orientation='vertical', fraction=0.02, pad=0.02)
cbar.set_label(r"$\theta_O$ [$\degree C$]")

plt.savefig(os.path.join(output_path, "Temperature_Diff_Global_Profile_Long_10yr"), bbox_inches='tight', dpi=600)
# plt.show()
plt.close()



# var = 'thetao'
# data = ds_groundtruth
# import numpy as np
# mae = np.abs((ds_prediction_temp[var] - data[var]).weighted(data['areacello']*data['dz']).mean(['x', 'y', 'lev', 'time']))
# cor = ((ds_prediction_temp[var]*data[var]).weighted(data['areacello']*data['dz']).mean(['x', 'y', 'lev']) / np.sqrt((ds_prediction_temp[var]**2).weighted(data['areacello']*data['dz']).mean(['x', 'y', 'lev']) * (data[var]**2).weighted(data['areacello']*data['dz']).mean(['x', 'y', 'lev']))).mean()
# print(f"Thermo - \nMAE : {mae.compute()}\nCOR : {cor.compute()}")

# mae = np.abs((ds_prediction_all[var] - data[var]).weighted(data['areacello']*data['dz']).mean(['x', 'y', 'lev', 'time']))
# cor = ((ds_prediction_all[var]*data[var]).weighted(data['areacello']*data['dz']).mean(['x', 'y', 'lev']) / np.sqrt((ds_prediction_all[var]**2).weighted(data['areacello']*data['dz']).mean(['x', 'y', 'lev']) * (data[var]**2).weighted(data['areacello']*data['dz']).mean(['x', 'y', 'lev']))).mean()
# print(f"Thermo+Dynamic - \nMAE : {mae.compute()}\nCOR : {cor.compute()}")