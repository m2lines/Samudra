# this script calculates the mean and std of the LLC4320 dataset, storing as .zarr files
import numpy as np
import xarray as xr
import zarr
import dask
from pathlib import Path
from dask.distributed import Client, LocalCluster

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True
)
log = logging.getLogger(__name__)

def calc_time_indices(num_time_samples):
    indices = np.linspace(0, 10311, num_time_samples+2, dtype=int)[1:-1]
    dif = indices[1]-indices[0]
    for i, val in enumerate(indices):
        offset = np.random.randint(int(-dif/2.5), int(dif/2.5)) # divide by 2.5 keeps indices more centered
        indices[i] = val + offset
    return indices

def main():

    log.info('initializing')

    
    cluster = LocalCluster(
        n_workers=11,       # should be <= --cpus-per-task in slurm job 
        threads_per_worker=1,
        memory_limit='60GB', # memory_limit * n_workers should be < --mem in slurm job -------
        dashboard_address=None#,
       # local_directory="/scratch/codycruz/dask_tmp"
    )
    client = Client(cluster)
    log.info(client)

    # set params -----------------------------------------------------------------------------------------------------------
    experiments = [96] #[2,4,12,24,48,96]# must be less than length, prescribes number of timepoints selected randomly per experiment
    downsample = 8 # downsample the spatial (i,j) resolution by a factor of n: if 4, the 4320x4320 res becomes 1080x1080

    # define features (state variables and surface fluxes/forcings) to operate on
    surface_features = ['Eta', 'oceQnet', 'oceTAUX', 'oceTAUY']
    ubiquitous_features = ['U', 'V', 'Theta', 'Salt']#'W']

    # open dataset
    log.info("opening dataset")
    LLC_path = '/orcd/data/abodner/003/LLC4320/LLC4320' # path to LLC folder
    LLC = xr.open_zarr(LLC_path,consolidated=False)[surface_features + ubiquitous_features]

    # downsample and chunk
    LLC = LLC.isel(i=slice(None, None, downsample), j=slice(None, None, downsample)) # downsample i,j spatial resolution
    LLC = LLC.chunk({'i': 4320/(downsample * 2), 'j': 4320/(downsample * 2), 'k': 1, 'time': 1, 'face': 13})

    # some other things
    length = len(LLC[ubiquitous_features[0]].time)-1
    total_exps = len(experiments)

    # define depths
    depth_extent = len(LLC[ubiquitous_features[0]].k)

    for exp, num_time_samples in enumerate(experiments):

        log.info(f'beginning experiment {exp + 1}/{total_exps} with {num_time_samples} time points selected')

        # set paths
        output_root = Path('/orcd/data/abodner/002/cody/LLC_means_stds')
        mean_path = output_root / 'var_96_LLC_means.zarr' #f'{num_time_samples}_LLC_means.zarr'
        std_path  = output_root /'var_96_LLC_stds.zarr'  #f'{num_time_samples}_LLC_stds.zarr'

        # select random time indices
        assert length >= num_time_samples

        time_samples = calc_time_indices(num_time_samples)

        # sample LLC temporal subset
        LLC_sampled = xr.concat([LLC.isel(k = slice(0, depth_extent), time = t) for t in time_samples], dim = "time_sampled")

        # initialize mean/std dictionaries
        mean_dict = {}
        std_dict = {}

        # mean and std: UBIQUITOUS FEATURES
        for var in ubiquitous_features:
            da = LLC_sampled[var]
            if var == 'U':
                mean_da = da.mean(skipna=True, dim=['j', 'i_g', 'face']).persist()
        #        std_da = ((da - mean_da)**2).mean(skipna=True, dim=['j', 'i_g', 'face'])**0.5  # calcs std spatially, averages across time
                var_da = ((da - mean_da)**2).mean(skipna=True, dim=['j', 'i_g', 'face']) # calcs std spatiotemporally
            elif var == 'V': 
                mean_da = da.mean(skipna=True, dim=['j_g', 'i', 'face']).persist()
        #        std_da = ((da - mean_da)**2).mean(skipna=True, dim=['j_g', 'i', 'face'])**0.5 
                var_da = ((da - mean_da)**2).mean(skipna=True, dim=['j_g', 'i', 'face'])
               
            else: 
                mean_da = da.mean(skipna=True, dim=['j', 'i', 'face']).persist()
        #        std_da = ((da - mean_da)**2).mean(skipna=True, dim=['j', 'i', 'face'])**0.5 
                var_da = ((da - mean_da)**2).mean(skipna=True, dim=['j', 'i', 'face'])

            # average over times and convert variance to standard deviation
            mean_da = mean_da.mean(dim='time_sampled')
            std_da  = (var_da.mean(dim='time_sampled'))**(0.5) # average variance across time, take sqrt --> std

            #if var == 'W':
            #    for klev in range(depth_extent):
            #        mean_dict[f"{var}_lev_{klev}"] = (
            #            mean_da.isel(k_p1=klev).reset_coords(drop=True))

            #        std_dict[f"{var}_lev_{klev}"] = (
            #            std_da.isel(k_p1=klev).reset_coords(drop=True))


            #else:
            for klev in range(depth_extent):
                mean_dict[f"{var}_lev_{klev}"] = (
                    mean_da.isel(k=klev).reset_coords(drop=True))

                std_dict[f"{var}_lev_{klev}"] = (
                    std_da.isel(k=klev).reset_coords(drop=True))


        # mean and std: SURFACE FEATURES
        for var in surface_features:
            da = LLC_sampled[var]
            if var == 'oceTAUX':
                mean_da = da.mean(skipna=True, dim=['j', 'i_g', 'face']).reset_coords(drop=True).persist()
        #        std_da = ((da - mean_da)**2).mean(skipna=True, dim=['j', 'i_g', 'face'])**0.5
                var_da = ((da - mean_da)**2).mean(skipna=True, dim=['j', 'i_g', 'face'])
            
            elif var == 'oceTAUY':
                mean_da = da.mean(skipna=True, dim=['j_g', 'i', 'face']).reset_coords(drop=True).persist()
        #        std_da = ((da - mean_da)**2).mean(skipna=True, dim=['j_g', 'i', 'face'])**0.5
                var_da = ((da - mean_da)**2).mean(skipna=True, dim=['j_g', 'i', 'face'])

            else:
                mean_da = da.mean(skipna=True, dim=['j', 'i', 'face']).reset_coords(drop=True).persist()
        #        std_da = ((da - mean_da)**2).mean(skipna=True, dim=['j', 'i', 'face'])**0.5
                var_da = ((da - mean_da)**2).mean(skipna=True, dim=['j', 'i', 'face'])


            # average over times and convert variance to standard deviation
            mean_dict[var] = mean_da.mean(dim='time_sampled')
            std_dict[var]  = (var_da.mean(dim='time_sampled'))**(0.5)

        # combine
        mean_ds = xr.Dataset(mean_dict)
        std_ds  = xr.Dataset(std_dict)

        # write to zarr
        mean_ds.to_zarr(mean_path, mode="w")
        std_ds.to_zarr(std_path, mode="w")

        log.info(f"finished experiment {exp + 1}/{total_exps} with {num_time_samples} time points selected")

        client.restart(wait_for_workers=True)

if __name__ == "__main__":
    main()
