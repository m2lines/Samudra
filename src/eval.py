from constants import INPT_VARS, EXTRA_VARS, OUT_VARS, REGIONS
import hydra
from hydra.utils import instantiate
from pathlib import Path
import os

from utils.data_utils import get_wet_mask, get_train_test_ranges, \
                             gen_data_025_lateral, gen_data_in_test,\
                             gen_data_out_test, data_CNN_Lateral
from utils.eval_utils import recur_pred_lateral, compute_mean, compute_var,compute_corrs_area,\
                                compute_rmse, compute_corrs, compute_KE, compute_time_spec,\
                                compute_ACC, gen_KE_spectrum, gen_enstrophy_spectrum, gen_enstrophy,\
                                compute_corrs_single, compute_ACC_single, compute_RMSE_single
from utils.subgrid_utils import coarse_grid, get_area_tensor
from utils.climate_utils import compute_laplacian_wet
from utils.plot_utils import plot_short_time_stats, plot_long_time_stats, plot_all_metrics

import numpy as np
import torch
import xarray as xr
import copy


class Eval:
    def __init__(self, args):
         # Getting input, extra input and output
        self.inputs = INPT_VARS[args.exp_num_in]
        self.extra_in = EXTRA_VARS[args.exp_num_extra]
        self.outputs = OUT_VARS[args.exp_num_out]

        self.str_in = "".join([i + "_" for i in self.inputs])
        self.str_ext = "".join([i + "_" for i in self.extra_in])
        self.str_out = "".join([i + "_" for i in self.outputs])

        print("inputs: " + self.str_in)
        print("extra inputs: " + self.str_ext)
        print("outputs: " + self.str_out)

        self.N_atm = len(self.extra_in) # Number of atmosphere variables
        self.N_in = len(self.inputs) 
        self.N_extra = self.N_atm + self.N_in # Number of atmosphere variables + Lateral boundary variables
        self.N_out = len(self.outputs)

        self.num_in = int((args.hist+1)*self.N_in + self.N_extra)

        print("Number of inputs: ", self.num_in) # 3 (ocean speeds + ocean temp)(t) +
                                            # 3 (atm wind stresses + atm temp)(t) +
                                            # 3 (boundary ocean speeds + boundary ocean temp)(t) -> 3 (ocean speeds + ocean temp)(t+1)
        print("Number of outputs: ", self.N_out) # 3

        # Post-fix strings
        self.str_video = 'steps_'+str(args.steps)+'_'+args.region+'_Test_in_' + self.str_in + 'ext_' + self.str_ext +'_out'+ self.str_out + 'N_train_' + str(args.N_samples) + "_Lateral_Data_025_no_smooth"
        self.str_save = 'steps_'+str(args.steps)+'_'+args.region+"_in_"+self.str_in+"ext_"+self.str_ext+"N_samples_"+str(args.N_samples)
        self.post_model_name = args.region+"_Test_in_"+self.str_in+"ext_"+self.str_ext+"_out"+self.str_in+"N_train_"+str(args.N_samples)+"_Lateral_Data_025_no_smooth"
        self.post_pred_name = args.region+"_in_"+self.str_in+"ext_"+self.str_ext+"N_samples_"+str(args.N_samples)

        # Getting start and end indices of train and test
        s_train, e_train, e_test = get_train_test_ranges(args.N_samples, args.N_val, args.lag, args.hist, args.interval)

        # Saving data
        print("Getting inputs")
        if "global" in args.region:
            inputs, extra_in, outputs = gen_data_global(self.inputs,self.extra_in,self.outputs,args.lag)
        else:
            # inputs, extra_in, outputs = gen_data_025_lateral(self.inputs,self.extra_in,self.outputs,args.lag,REGIONS[args.region]["lat"], REGIONS[args.region]["lon"],args.Nb,filter_T = True) # compare vary
            inputs, extra_in, outputs = gen_data_025_lateral(self.inputs,self.extra_in,self.outputs,args.lag,REGIONS[args.region]["lat"], REGIONS[args.region]["lon"],args.Nb)
        
        print("Calculating mask tensors")
        self.wet, self.wet_nan = get_wet_mask(inputs, "cpu")
        self.wet_bool = np.array(self.wet.cpu()).astype(bool)
        wet_lap = compute_laplacian_wet(self.wet_nan, args.Nb)
        wet_lap = xr.where(wet_lap==0,1,np.nan)
        self.wet_lap = np.nan_to_num(wet_lap)

        self.time_vec = inputs[0].time.data

        self.time_test = self.time_vec[e_test:(e_test+args.lag*args.N_test)]
        
        if args.save_test_data:
            print("Saving data")
            data_in_test = gen_data_in_test(0, e_test, args.N_test, args.lag, args.hist, inputs, extra_in)
            data_out_test = gen_data_out_test(0, e_test, args.N_test, args.lag, args.hist, outputs)
            self.test_data = data_CNN_Lateral(data_in_test, data_out_test, self.wet.to(device = "cpu"), self.N_atm, args.Nb, device=args.device) 
            torch.save(self.test_data, Path(args.data_dir) / 'test_data_{0}.pt'.format(self.str_save))

        else:
            print("Loading test data")
            self.test_data = torch.load(Path(args.data_dir) / 'test_data_{0}.pt'.format(self.str_save))

        # Stats
        self.mean_out = self.test_data.norm_vals['m_out']  
        self.std_out = self.test_data.norm_vals['s_out']  
        self.mean_in = self.test_data.norm_vals['m_in']  
        self.std_in = self.test_data.norm_vals['s_in']  

        # clim
        print("Saving clim")
        clim = np.zeros((366,*self.wet.shape,3))
        for i in range(self.N_out):
            clim[:,:,:,i] = outputs[i].groupby('time.dayofyear').mean('time').data
        self.clim = clim

        # Getting area tensor
        print("Computing area tensor")
        self.grids = xr.open_dataset(args.grid_path)
        if "global" in args.region:
            self.grids = coarse_grid(self.grids,args.factor)

        else:
            self.grids = self.grids.sel({"yu_ocean":slice(*REGIONS[args.region]["lat"]),"xu_ocean":slice(*REGIONS[args.region]["lon"])})

        self.area = torch.from_numpy(self.grids["area_C"].to_numpy()).to(device="cpu")
        self.dx = self.grids["dxu"].to_numpy()
        self.dy = self.grids["dyu"].to_numpy()

        # Model
        if args.network == "ViT":
            model = instantiate(args.vit, in_channels=self.num_in, output_channels=self.N_in, 
                                img_size=[*self.test_data[0][0].shape[1:]])

        model = model.to(args.device)
        self.full_model_name = (args.short_model_name + self.post_model_name)
        full_model_path = Path(args.nets_dir) / (args.short_model_name + self.post_model_name + '.pt')
        model.load_state_dict(torch.load(full_model_path,map_location=torch.device(args.device)))

        self.model = model

        # Pred model path dir
        self.pred_model_path = Path(args.path_dir) / self.full_model_name
        if not os.path.isdir(self.pred_model_path):
            os.makedirs(self.pred_model_path)
        

        self.Nb = args.Nb
        self.hist = args.hist
        self.lag = args.lag
        self.N_test = args.N_test
        self.output_dir = args.output_dir
        self.region = args.region
        self.unet_path = args.unet_path
        self.steps = args.steps
    
    def generate_pred_lateral(self):
        print("Generation Pred begin...")
        for ns in [4000]:
            for rand_ind in range(1,4):
                print(ns,rand_ind)
                model_pred = recur_pred_lateral(self.N_test, self.test_data, self.model, self.hist, self.N_in, self.N_extra, self.Nb)
                print("data_gen")
                da = xr.DataArray(
                    data=model_pred,
                    dims=["time","x", "y","var"],
                )

                da.to_zarr(self.pred_model_path / ("Pred_lateral_Fast_Data_025_" + self.post_pred_name + "_rand_seed_"+str(rand_ind)+".zarr"),mode="w")

    
    def generate_short_pred_lateral(self):
        print("Generation Short Pred begin...")
        N_run = 5
        len_run = 200

        for ns in [4000]:
            for rand_ind in range(1,4):
                data_shape = self.test_data[0][1].shape
                model_pred = np.zeros((int(N_run*len_run), data_shape[1], data_shape[2], data_shape[0]))        

                for i in range(N_run):
                    print(ns,rand_ind)
                    temp = copy.deepcopy(self.test_data)   
                    temp.input = temp.input[int(i*len_run):int((i+1)*len_run)]
                    temp.output = temp.output[int(i*len_run):int((i+1)*len_run)]
                    temp.size = len_run

                    model_pred_temp = recur_pred_lateral(len_run,temp,self.model, self.hist, self.N_in, self.N_extra, self.Nb)
                    print("data_gen")
                    model_pred[int(i*len_run):int((i+1)*len_run)] = model_pred_temp

                da = xr.DataArray(
                    data=model_pred,
                    dims=["time","x", "y","var"],
                )

                da.to_zarr(self.pred_model_path / ("Pred_Short_Data_025_" + self.post_pred_name +"_rand_seed_" + str(rand_ind) + ".zarr"),mode="w")
    
    ### Need to Refactor the following functions
    def compare_pred_lateral(self):
        def get_stats(zarr_path, region, rand_int, str_in, str_ext, test_data, area, wet_bool, N_mean, lag):
            try:
                model_pred_atm = xr.open_zarr(Path(zarr_path) / ("Pred_lateral_Fast_Data_025_"+region+"_in_"+str_in+"ext_"+str_ext+"N_samples_"+str(4000)+"_rand_seed_"+str(rand_int)+".zarr")).sel(time=slice(0,N_mean)).to_array().to_numpy().squeeze()
            except:
                print("Path does not exist. Make sure to set run_gen_pred to True in config.")
            mean_atm, auto_mean = compute_mean(N_mean,test_data,model_pred_atm,area.cpu(),wet_bool)
            var_atm, auto_var = compute_var(N_mean,test_data,model_pred_atm,area.cpu(),wet_bool)    
            rmse_atm, auto_rmse = compute_rmse(np.min((500,N_mean)),test_data,model_pred_atm,area.cpu(),wet_bool)
            corrs_atm, auto_corrs = compute_corrs(np.min((500,N_mean)),test_data,model_pred_atm,wet_bool)
            KE, auto_KE = compute_KE(N_mean,test_data,model_pred_atm,area,wet_bool)
            freqs,FFT,auto_FFT = compute_time_spec(N_mean,auto_mean,mean_atm,lag)

            return model_pred_atm, mean_atm, auto_mean, rmse_atm, auto_rmse, corrs_atm, auto_corrs, KE, auto_KE, freqs, FFT, auto_FFT, var_atm, auto_var

        def get_spred(zarr_path, region, num_IC, str_in, str_ext, test_data, area, wet_bool, N_mean, lag):
            mean = np.zeros((num_IC,N_mean,3))
            var = np.zeros((num_IC,N_mean,3))    
            KE = np.zeros((num_IC,N_mean))    
            rmse = np.zeros((num_IC,np.min((500,N_mean)),3))
            corrs = np.zeros((num_IC,np.min((500,N_mean)),3))
            FFTs = np.zeros((num_IC,int(N_mean/2+1),3))
            
            for i in range(0,num_IC):
                out, mean_1, out, rmse_1, out, corrs_1, out, KE_1, out, freqs, FFT_1, out, var_1, out = get_stats(zarr_path, region,i+1,str_in,str_ext,test_data,area,wet_bool,N_mean,lag)
                KE[i] = KE_1
                mean[i] = mean_1
                rmse[i] = rmse_1
                corrs[i] = corrs_1
                FFTs[i] = FFT_1    
                var[i] = var_1
            return mean, rmse, corrs, KE, FFTs, freqs, var

        print("Long time stats compute begin...")
        mean_vit, rmse_vit, corrs_vit,KE_vit,FFTs_vit,freqs,var_vit = get_spred(self.pred_model_path, self.region, 3, self.str_in, self.str_ext, self.test_data, self.area, self.wet_bool, 3000, self.lag)
        model_pred_vit, m_vit, auto_mean, r_vit, auto_rmse, c_vit, auto_corrs, K_vit, auto_KE, freqs, F_vit, auto_FFT, v_vit, auto_var = get_stats(self.pred_model_path, self.region, 1, self.str_in, self.str_ext, self.test_data, self.area, self.wet_bool, 3000, self.lag) # zarr_path, region, rand_int, str_in, str_ext, test_data, area, wet_bool, N_mean, lag

        mean_unet, rmse_unet, corrs_unet,KE_unet,FFTs_unet,_,var_unet = get_spred(self.unet_path, self.region, 3, self.str_in, self.str_ext, self.test_data, self.area, self.wet_bool, 3000, self.lag)
        model_pred_unet, m_unet, _, r_unet, _, c_unet, _, K_unet, _, _, F_unet, _, v_unet, _ = get_stats(self.unet_path, self.region, 1, self.str_in, self.str_ext, self.test_data, self.area, self.wet_bool, 3000, self.lag)

        print("Long time stats plot begin...")
        plot_long_time_stats(self.region, self.str_save, self.output_dir, self.N_test, self.lag, freqs, auto_FFT, FFTs_unet, FFTs_vit,\
                        auto_mean, mean_unet, mean_vit)
    
    def compare_short_pred_lateral(self):
        def get_stats(zarr_path,region,N_IC,rand_int,str_in,str_ext,test_data,clim,time,area,wet_bool,N_mean): 
            try:
                model_pred_atm = xr.open_zarr(Path(zarr_path) / ("Pred_Short_Data_025_"+region+"_in_"+str_in+"ext_"+str_ext+"N_samples_"+str(4000)+"_rand_seed_"+str(rand_int)+".zarr")).to_array().to_numpy().squeeze()
            except:
                print("Path does not exist. Make sure to set run_gen_short_pred to True in config.")
            temp = copy.deepcopy(test_data)   
            temp.input = temp.input[int((N_IC-1)*N_mean):int((N_IC)*N_mean)]
            temp.output = temp.output[int((N_IC-1)*N_mean):int((N_IC+1)*N_mean)]
            temp.size = N_mean
            rmse_atm,auto_rmse = compute_rmse(N_mean,temp,model_pred_atm[int((N_IC-1)*N_mean):int((N_IC)*N_mean)],area.cpu(),wet_bool)
            corrs_atm,auto_corrs = compute_corrs_area(N_mean,temp,model_pred_atm[int((N_IC-1)*N_mean):int((N_IC)*N_mean)],area.cpu(),wet_bool)
            ACC_atm, auto_ACC = compute_ACC(N_mean,temp,model_pred_atm[int((N_IC-1)*N_mean):int((N_IC)*N_mean)],clim,time,area.cpu(),wet_bool)
            KE, auto_KE = compute_KE(N_mean,temp,model_pred_atm[int((N_IC-1)*N_mean):int((N_IC)*N_mean)],area,wet_bool)
            return rmse_atm, auto_rmse, corrs_atm, auto_corrs, ACC_atm, auto_ACC, KE, auto_KE
        
        def get_spred(zarr_path,region,N_IC,num_IC,str_in,str_ext,test_data,clim,time,area,wet_bool,N_mean):
            KE = np.zeros((int(num_IC*N_IC),N_mean))    
            rmse = np.zeros((int(num_IC*N_IC),N_mean,3))
            corrs = np.zeros((int(num_IC*N_IC),N_mean,3))
            ACC = np.zeros((int(num_IC*N_IC),N_mean,3))
            
            auto_KE = np.zeros((N_IC,N_mean))    
            auto_rmse = np.zeros((N_IC,N_mean,3))
            auto_corrs = np.zeros((N_IC,N_mean,3))   
            auto_ACC = np.zeros((N_IC,N_mean,3))    
            
            for i in range(0,num_IC):
                print(i)
                for j in range(0,N_IC):
                    rmse_1, auto_rmse_1, corrs_1, auto_corrs_1, ACC_1, auto_acc_1, KE_1, auto_KE_1 = get_stats(zarr_path,region,j+1,i+1,str_in,str_ext,test_data,clim,time,area,wet_bool,N_mean)
                    KE[int(i*N_IC+j)] = KE_1
                    rmse[int(i*N_IC+j)] = rmse_1
                    corrs[int(i*N_IC+j)] = corrs_1
                    ACC[int(i*N_IC+j)] = ACC_1
                    
                    if i ==0:
                        auto_rmse[j] = auto_rmse_1
                        auto_KE[j] = auto_KE_1
                        auto_corrs[j] = auto_corrs_1
                        auto_ACC[j] = auto_acc_1
                        
            return rmse, corrs, ACC, KE, auto_rmse, auto_corrs, auto_ACC,auto_KE
        
        print("Short time stats compute begin...")
        rmse_vit, corrs_vit, ACC_vit, KE_vit, auto_rmse,auto_corrs,auto_ACC,auto_KE = get_spred(self.pred_model_path, self.region, 5, 3, self.str_in, self.str_ext, self.test_data, self.clim, self.time_test, self.area, self.wet_bool, 200)
        rmse_unet, corrs_unet, ACC_unet, KE_unet, _,_,_,_ = get_spred(self.unet_path, self.region, 5, 3, self.str_in, self.str_ext, self.test_data, self.clim, self.time_test, self.area, self.wet_bool, 200)

        print("Short time stats plot begin...")
        plot_short_time_stats(self.region, self.str_save, self.output_dir, self.N_test, self.lag, auto_ACC, ACC_unet, ACC_vit,\
                          auto_rmse, rmse_unet, rmse_vit, auto_KE, KE_unet, KE_vit,\
                          auto_corrs, corrs_unet, corrs_vit)

    def plot_metrics(self):
        print("Plot metrics begin...")
        unet_path = Path(self.unet_path) / ("Pred_Short_Data_025_"+self.region+"_in_"+self.str_in+"ext_"+"tau_u_tau_v_t_ref_"+"steps_"+str(8)+"_rand_seed_"+str(1)+".zarr")
        model_pred_vit = xr.open_zarr(self.pred_model_path / ("Pred_Short_Data_025_" + self.post_pred_name +"_rand_seed_" + str(1) + ".zarr")).to_array().to_numpy().squeeze()
        model_pred_unet = xr.open_zarr(unet_path).to_array().to_numpy().squeeze()
        
        ### Short time scale metrics
        N_plot = 200

        # KE
        print("Getting KE stats...")
        KE_spec_vit, KE_spec_true = gen_KE_spectrum(N_plot, self.test_data, model_pred_vit, self.grids, self.wet)
        KE_spec_unet, KE_spec_true = gen_KE_spectrum(N_plot, self.test_data, model_pred_unet, self.grids, self.wet)

        KE_vit, KE_true = compute_KE(N_plot, self.test_data, model_pred_vit, self.area, self.wet_bool)
        KE_unet, KE_true = compute_KE(N_plot, self.test_data, model_pred_unet, self.area, self.wet_bool)
        
        # Enstrophy
        print("Getting Enstrophy stats...")
        enst_spec_vit, enst_spec_true = gen_enstrophy_spectrum(N_plot, self.test_data, model_pred_vit, self.grids, self.wet, self.wet_lap)
        enst_spec_unet, enst_spec_true = gen_enstrophy_spectrum(N_plot, self.test_data, model_pred_unet, self.grids, self.wet, self.wet_lap)
    
        enst_vit, enst_true = gen_enstrophy(N_plot, self.test_data, model_pred_vit, self.dx, self.dy, self.Nb, self.wet_lap)
        enst_vit = enst_vit.mean(axis=(1,2))

        enst_unet, enst_true = gen_enstrophy(N_plot, self.test_data, model_pred_unet, self.dx, self.dy, self.Nb, self.wet_lap)
        enst_unet = enst_unet.mean(axis=(1,2))

        enst_true = enst_true.mean(axis=(1,2))

        ### Spatial matching metrics
        print("Getting Spatial matching stats...")
        u_test = np.array(self.test_data[:][1][:,0]*self.std_out[0] +self.mean_out[0])
        v_test = np.array(self.test_data[:][1][:,1]*self.std_out[1] +self.mean_out[1])
        T_test = np.array(self.test_data[:][1][:,2]*self.std_out[2] +self.mean_out[2])

        # Corr
        print("Getting Corr stats...")
        N_eval = 200
        corr_T_vit, corr_T_true = compute_corrs_single(N_eval, T_test, model_pred_vit[:,:,:,2], self.area, self.wet_bool, self.std_out[2], self.mean_out[2])
        corr_T_unet, corr_T_true = compute_corrs_single(N_eval, T_test, model_pred_unet[:,:,:,2], self.area, self.wet_bool, self.std_out[2], self.mean_out[2])

        # RMSE
        print("Getting RMSE stats...")
        RMSE_T_vit, RMSE_T_true = compute_RMSE_single(N_eval, T_test, model_pred_vit[:,:,:,2],
                                                self.area, self.wet_bool)
        RMSE_T_unet, RMSE_T_true = compute_RMSE_single(N_eval, T_test, model_pred_unet[:,:,:,2],
                                                self.area, self.wet_bool)

        # ACC
        print("Getting ACC stats...")
        N_eval = 100
        ACC_T_vit, ACC_T_true = compute_ACC_single(N_eval, T_test, model_pred_vit[:,:,:,2],
                                                self.clim[:,:,:,2],self.time_test, self.area, self.wet_bool)
        ACC_T_unet, ACC_T_true = compute_ACC_single(N_eval, T_test, model_pred_unet[:,:,:,2],
                                                self.clim[:,:,:,2],self.time_test, self.area, self.wet_bool)

        print("Plotting everything...")
        plot_all_metrics(self.region, self.str_save, self.output_dir, self.lag, self.steps, KE_spec_true, KE_spec_unet, KE_spec_vit,\
                KE_true, KE_unet, KE_vit, enst_spec_true, enst_spec_unet, enst_spec_vit, corr_T_true, corr_T_unet, corr_T_vit,\
                enst_true, enst_unet, enst_vit, RMSE_T_true, RMSE_T_unet, RMSE_T_vit,\
                ACC_T_true, ACC_T_unet, ACC_T_vit)

    # Cant plot this without the right overlay
    def plot_animation(self):
        pass
    
    def send_data_to_cpu(self):
        self.test_data.set_device(device = "cpu")
        
def main(args):
    e = Eval(args)

    if args.run_gen_pred:
        e.generate_pred_lateral()
    else:
        print("Skipping pred generation")

    if args.run_gen_short_pred:
        e.generate_short_pred_lateral()
    else:
        print("Skipping short pred generation")
    
    # Sending the data back to cpu
    e.send_data_to_cpu()

    if args.run_full_pred:
        e.compare_pred_lateral()
    
    if args.run_short_pred:
        e.compare_short_pred_lateral()
    
    if args.run_plot_metrics:
        e.plot_metrics()
    
    if args.run_plot_animation:
        e.plot_animation()
        
