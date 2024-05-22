from constants import INPT_VARS, EXTRA_VARS, OUT_VARS, GLOBAL_COMBINED_STATS
import hydra
from hydra.utils import instantiate
from pathlib import Path
import os
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import logging

from utils.data_utils import (
    get_wet_mask,
    get_train_test_ranges,
    gen_data_in_test,
    gen_data_out_test,
    data_CNN_Lateral,
    data_CNN_Dynamic,
    gen_data_025_lateral,
    gen_data_global_new,
)
from utils.eval_utils import (
    generate_model_rollout,
    compute_mean,
    compute_var,
    compute_corrs_area,
    compute_rmse,
    compute_corrs,
    compute_KE,
    compute_time_spec,
    compute_ACC,
    compute_nino34,
    compute_amo,
    gen_KE_spectrum,
    gen_KE,
    gen_KE_range,
    gen_value_range,
    gen_enstrophy_spectrum,
    gen_enstrophy,
    compute_corrs_single,
    compute_ACC_single,
    compute_RMSE_single,
    compute_mean_single,
)
from utils.subgrid_utils import get_area_tensor
from utils.climate_utils import compute_laplacian_wet
from utils.plot_utils import (
    plot_short_time_stats,
    plot_long_time_stats,
    plot_map,
    plot_error_map,
    plot_both_error_map,
    plot_metrics_KE_spectrum,
    plot_metrics_KE,
    plot_metrics_enstrophy_spectrum,
    plot_metrics_entrophy,
    plot_metrics_corr,
    plot_metrics_rmse,
    plot_metrics_acc,
    plot_metrics_mean,
    plot_metrics_pdf,
    get_initial_snapshot_fig,
    plot_region_based_metric,
    plot_diff_map,
)

import numpy as np
import torch
import xarray as xr
import copy


class Eval:
    def __init__(self, args, no_train=False):
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

        self.N_atm = len(self.extra_in)  # Number of atmosphere variables
        self.N_in = len(self.inputs)
        if args.lateral:
            self.N_extra = (
                self.N_atm + self.N_in
            )  # Number of atmosphere variables + Lateral boundary variables
        else:
            self.N_extra = self.N_atm  # Number of atmosphere variables
        self.N_out = len(self.outputs)

        self.num_in = int((args.hist + 1) * self.N_in + self.N_extra)

        print("Number of inputs: ", self.num_in)  # 3 (ocean speeds + ocean temp)(t) +
        # 3 (atm wind stresses + atm temp)(t) +
        # 3 (boundary ocean speeds + boundary ocean temp)(t) -> 3 (ocean speeds + ocean temp)(t+1)
        print("Number of outputs: ", self.N_out)  # 3

        # Post-fix strings
        self.str_train = (
            "steps_"
            + str(args.steps)
            + "_"
            + args.train_region
            + "_Test_in_"
            + self.str_in
            + "ext_"
            + self.str_ext
            + "_out"
            + self.str_out
            + "N_train_4000"
            + "_Lateral_Data_025_no_smooth"
        )
        self.str_save = (
            "steps_"
            + str(args.steps)
            + "_"
            + args.train_region
            + "_"
            + args.region
            + "_in_"
            + self.str_in
            + "ext_"
            + self.str_ext
            + "N_samples_"
            + str(args.N_samples)
        )
        self.post_model_name = (
            "Train_" + args.train_region
            + "_Test_" + args.region
            + "_Test_in_"
            + self.str_in
            + "ext_"
            + self.str_ext
            + "_out"
            + self.str_in
            + "N_train_"
            + str(args.N_samples)
            + "_Lateral_Data_025_no_smooth"
        )
        self.post_pred_name = (
            args.region
            + "_in_"
            + self.str_in
            + "ext_"
            + self.str_ext
            + "N_samples_"
            + str(args.N_samples)
        )

        # Getting start and end indices of train and test
        s_train, e_train, e_test = get_train_test_ranges(
            args.N_samples, args.N_val, args.lag, args.hist, args.interval
        )

        # Saving data
        print("Getting inputs")
        if "global_1" == args.region:
            inputs, extra_in, outputs = gen_data_global_new(self.inputs, self.extra_in, self.outputs, args.lag)
        elif "global_2x" == args.region:
            inputs, extra_in, outputs = gen_data_global_new(self.inputs, self.extra_in, self.outputs, args.lag, run_type ="2x")
        elif "global_4x" == args.region:
            inputs, extra_in, outputs = gen_data_global_new(self.inputs, self.extra_in, self.outputs, args.lag, run_type ="4x")
        else:
            raise NotImplementedError

        print("Calculating mask tensors")
        self.wet, self.wet_nan = get_wet_mask(inputs, "cpu")
        self.wet_bool = np.array(self.wet.cpu()).astype(bool)
        wet_lap = compute_laplacian_wet(self.wet_nan, 4) # hardcoded
        wet_lap = xr.where(wet_lap == 0, 1, np.nan)
        self.wet_lap = np.nan_to_num(wet_lap)
        print("Wet resolution:", self.wet.shape)

        self.time_vec = inputs[0].time.data

        self.time_test = self.time_vec[e_test : (e_test + args.lag * args.N_test)]

        print("Loading Train data")
        if args.region == 'global_4x':
            str1_video = "steps_4_combined_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth"
            str2_video = "steps_4_combined_global_2x_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth"

            # Dataloaders
            train_data1 = torch.load(
                Path(args.data_dir) / "train_data_cnn_{0}.pt".format(str1_video),
                map_location=torch.device("cpu"),
            )

            train_data2 = torch.load(
                Path(args.data_dir) / "train_data_cnn_{0}.pt".format(str2_video),
                map_location=torch.device("cpu"),
            )
            
            train_data = torch.utils.data.ConcatDataset([train_data1, train_data2])
        else:
            train_data = torch.load(
                        Path(args.data_dir) / "train_data_cnn_{0}.pt".format(self.str_train),
                        map_location=torch.device("cpu"),
                    )
        if no_train:
            print("Deleting train data")
            del train_data
        else:
            print("Saving train data")
            self.train_data = train_data
        if args.save_test_data:
            print("Saving data")
            data_in_test = gen_data_in_test(
                0, e_test, args.N_test, args.lag, args.hist, inputs, extra_in
            )
            data_out_test = gen_data_out_test(
                0, e_test, args.N_test, args.lag, args.hist, outputs
            )
            if "global" in args.region:
                norm_vals = train_data.norm_vals
                if "combined" in args.train_region:
                    assert len(norm_vals) == len(GLOBAL_COMBINED_STATS) and all(np.array_equal(norm_vals[k], GLOBAL_COMBINED_STATS[k]) for k in norm_vals)
                self.test_data = data_CNN_Dynamic(
                    data_in_test,
                    data_out_test,
                    self.wet.to(device="cpu"),
                    norm_vals,
                    device=args.device,
                )
            else:
                raise NotImplementedError()
            torch.save(
                self.test_data,
                Path(args.data_dir) / "test_data_cnn_{0}.pt".format(self.str_save),
            )

        else:
            print("Loading test data")
            self.test_data = torch.load(
                Path(args.data_dir) / "test_data_cnn_{0}.pt".format(self.str_save)
            )

        full_model_path = args.ckpt_path
        self.full_model_name = args.network + "_" + self.post_model_name

        # Stats
        self.mean_out = self.test_data.norm_vals["m_out"]
        self.std_out = self.test_data.norm_vals["s_out"]
        self.mean_in = self.test_data.norm_vals["m_in"]
        self.std_in = self.test_data.norm_vals["s_in"]

        # clim
        self.clim = None
        if args.save_clim_data:
            print("Saving clim")
            clim = np.zeros((366, *self.wet.shape, 3))
            for i in range(self.N_out):
                clim[:, :, :, i] = (
                    outputs[i].groupby("time.dayofyear").mean("time").data
                )
            torch.save(
                clim,
                Path(args.data_dir) / "clim_cnn_{0}.pt".format(self.str_save),
            )

        else:
            print("Loading clim")
            clim = torch.load(
                Path(args.data_dir) / "clim_cnn_{0}.pt".format(self.str_save)
            )

        self.clim = clim

        # Getting area tensor
        print("Computing area tensor")
        self.grids = xr.open_dataset('/scratch/as15415/Data/CM2x_grids/Grid_New.nc').rename({"dx": "dxu", "dy": "dyu"})

        self.area = torch.from_numpy(self.grids["area_C"].to_numpy()).to(device="cpu")
        self.dx = self.grids["dxu"].to_numpy()
        self.dy = self.grids["dyu"].to_numpy()

        # Pred model path dir
        self.pred_model_path = Path(args.path_dir) / self.full_model_name
        if not os.path.isdir(self.pred_model_path):
            os.makedirs(self.pred_model_path)

        self.Nb = args.Nb
        self.hist = args.hist
        self.lag = args.lag
        self.N_test = args.N_test
        self.N_samples = args.N_samples
        self.output_dir = args.output_dir
        self.region = args.region
        self.steps = args.steps
        self.network = args.model_name_replace
        self.inputs = inputs

        self.pred_region = args.region
        self.pred_names = args.pred_names if args.pred_names else []
        self.pred_paths = args.pred_paths if args.pred_paths else []

        self.JUPYTER_MODE = False

    def load_long_data(self):
        print("Load long data...")
        model_pred_net = (
            xr.open_zarr(
                self.pred_model_path
                / (
                    "Pred_lateral_Fast_Data_025_"
                    + self.post_pred_name
                    + "_rand_seed_"
                    + str(1)
                    + ".zarr"
                )
            )
            .to_array()
            .to_numpy()
            .squeeze()
        )

        model_pred_saved_nets = []
        for model_pred_path in self.pred_paths:
            net_path = Path(model_pred_path) / (
                "Pred_lateral_Fast_Data_025_"
                + self.pred_region
                + "_in_"
                + self.str_in
                + "ext_"
                + "tau_u_tau_v_t_ref_"
                + "N_samples_"
                + str(self.N_samples)
                + "_rand_seed_"
                + str(1)
                + ".zarr"
            )

            model_pred_saved_nets.append(
                xr.open_zarr(net_path).to_array().to_numpy().squeeze()
            )
        
        return model_pred_net, model_pred_saved_nets
        
    def send_data_to_cpu(self):
        self.test_data.set_device(device="cpu")

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import copy
from datetime import datetime
import os

# G1, G1
with initialize_config_dir(version_base=None, config_dir="/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/configs"):
    args1 = compose(config_name="exp/eval_swin_global", overrides=[
        "output_dir=./notebooks/temp/{0}_datapdf".format(str(datetime.now())[:10]),
        "model_name_replace=Swin",
        "network=Foundation Swin Train1Eval1",
        "train_region=global_1",
        "region=global_1",
        "swin.embed_dim=60",
        "exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample",
        "ckpt_path=/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-05-11-foundation_train_swintrans60_global_1/swintrans60/saved_nets/swin_best_steps_4_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt",
        "pred_names=['UNet (Baseline)', 'ConvNext UNet']",
        "pred_paths=['/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/Preds/Foundation Adam UNet Train1Eval1_Train_global_1_Test_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth', '/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/Preds/Foundation ConvNext UNet Train1Eval1_Train_global_1_Test_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth']"
    ])
if not os.path.exists(args1.output_dir):
    os.mkdir(args1.output_dir)

e1 = Eval(args1)
e1.send_data_to_cpu()

# G1, G2x
with initialize_config_dir(version_base=None, config_dir="/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/configs"):
    args2 = compose(config_name="exp/eval_swin_global", overrides=[
        "output_dir=./notebooks/temp/{0}_datapdf".format(str(datetime.now())[:10]),
        "model_name_replace=Swin",
        "network=Foundation Swin Train1Eval2x",
        "train_region=global_1",
        "region=global_2x",
        "swin.embed_dim=60",
        "exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample",
        "ckpt_path=/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-05-11-foundation_train_swintrans60_global_1/swintrans60/saved_nets/swin_best_steps_4_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt",
        "pred_names=['UNet (Baseline)', 'ConvNext UNet']",
        "pred_paths=['/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/Preds/Foundation Adam UNet Train1Eval2x_Train_global_1_Test_global_2x_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth', '/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/Preds/Foundation ConvNext UNet Train1Eval2x_Train_global_1_Test_global_2x_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth']"
    ])

e2 = Eval(args2, no_train=True)
e2.send_data_to_cpu()

# G1_2x, G_4x
with initialize_config_dir(version_base=None, config_dir="/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/configs"):
    args3 = compose(config_name="exp/eval_swin_global", overrides=[
        "output_dir=./notebooks/temp/{0}_datapdf".format(str(datetime.now())[:10]),
        "model_name_replace=Swin",
        "network=Foundation Swin Train12xEval4x",
        "train_region=combined_global_1",
        "region=global_4x",
        "swin.embed_dim=60",
        "N_samples=0",
        "N_val=0",
        "N_test=2000",
        "exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample",
        "ckpt_path=/scratch/sg7761/m2lines/Ocean_Emulator/train/2024-05-13-foundation_train_swin_global_1_2x/foundationswin/saved_nets/swin_best_steps_4_global_1_2x_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt",
        "pred_names=['UNet (Baseline)', 'ConvNext UNet']",
        "pred_paths=['/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/Preds/Foundation Adam UNet Train12xEval4x_Train_combined_global_1_Test_global_4x_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_0_Lateral_Data_025_no_smooth', '/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/Preds/Foundation ConvNext UNet Train12xEval4x_Train_combined_global_1_Test_global_4x_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_0_Lateral_Data_025_no_smooth']"
    ])

e3 = Eval(args3)
e3.send_data_to_cpu()


def get_pdf(e, model_pred_net, model_pred_saved_nets, start=100, N_days=100, long=False):
    # PDF
    print("Getting PDF stats...")
    pdf = {}
    ind_plot = 2
    true_field = (
        e.test_data[start : start + N_days][1][
            :, ind_plot, e.wet_bool
        ].flatten()
        * e.std_out[ind_plot]
    ) + e.mean_out[ind_plot]
    true_pdf, bins_true = np.histogram(true_field, bins=150, density=True)
    bins_true = (bins_true[1:] + bins_true[:-1]) / 2

    field_net = model_pred_net[
        start : start + N_days, e.wet_bool, ind_plot
    ].flatten()
    pdf_net, bins_net = np.histogram(field_net, bins=150, density=True)
    bins_net = (bins_net[1:] + bins_net[:-1]) / 2

    pdf[ind_plot] = {
        "true_pdf": true_pdf,
        "true": [bins_true, true_pdf],
        e.network: [bins_net, pdf_net],
    }

    for i, model_pred_saved in enumerate(model_pred_saved_nets):
        field_i = model_pred_saved[
            start : start + N_days, e.wet_bool, ind_plot
        ].flatten()
        pdf_i, bins_i = np.histogram(field_i, bins=150, density=True)
        bins_i = (bins_i[1:] + bins_i[:-1]) / 2

        pdf[ind_plot][e.pred_names[i]] = [bins_i, pdf_i]

    return pdf
    

model_pred_net, model_pred_saved_nets = e1.load_long_data()
pdf1 = get_pdf(e1, model_pred_net, model_pred_saved_nets, start=1999, N_days=1000, long=True)
model_pred_net, model_pred_saved_nets = e2.load_long_data()
pdf2 = get_pdf(e2, model_pred_net, model_pred_saved_nets, start=1999, N_days=1000, long=True)
model_pred_net, model_pred_saved_nets = e3.load_long_data()
pdf3 = get_pdf(e3, model_pred_net, model_pred_saved_nets, start=999, N_days=1000, long=True)

del e2
del e1.test_data
del e3.test_data
# G2x, G2x
with initialize_config_dir(version_base=None, config_dir="/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/configs"):
    args4 = compose(config_name="exp/eval_swin_global", overrides=[
        "output_dir=./notebooks/temp/{0}_datapdf".format(str(datetime.now())[:10]),
        "model_name_replace=Swin",
        "network=Foundation Swin Train2xEval2x",
        "train_region=global_2x",
        "region=global_2x",
        "swin.embed_dim=60",
        "exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample",
        "ckpt_path=/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-05-20-foundation_train_swin60_global_2x/swin2x/saved_nets/swin_best_steps_4_global_2x_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt",
        "pred_names=['UNet (Baseline)', 'ConvNext UNet']",
        "pred_paths=['/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/Preds/Foundation Adam UNet Train2xEval2x_Train_global_2x_Test_global_2x_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth', '/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/Preds/Foundation ConvNext UNet Train2xEval2x_Train_global_2x_Test_global_2x_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth']"
    ])

e4 = Eval(args4)
e4.send_data_to_cpu()

def get_train_pdf(e1, e2, e3):
    print("Getting PDF stats...")
    pdf = {}
    ind_plot = 2
    total = len(e1.train_data)
    print(total)
    start = total - 1000 - 1
    end = start + 1000
    true_field1 = (
        e1.train_data[:][1][
            :, ind_plot, e1.wet_bool
        ].flatten()
        * e1.std_out[ind_plot]
    ) + e1.mean_out[ind_plot]
    true_pdf1, bins_true1 = np.histogram(true_field1.cpu(), bins=150, density=True)
    bins_true1 = (bins_true1[1:] + bins_true1[:-1]) / 2
    # del e1.train_data

    total = len(e2.train_data)
    print(total)
    start = total - 1000 - 1
    end = start + 1000
    true_field2_1 = (
        e2.train_data.datasets[0][:][1][
            :, ind_plot, e2.wet_bool
        ].flatten()
        * e2.std_out[ind_plot]
    ) + e2.mean_out[ind_plot]
    true_field2_2 = (
        e2.train_data.datasets[1][:][1][
            :, ind_plot, e2.wet_bool
        ].flatten()
        * e2.std_out[ind_plot]
    ) + e2.mean_out[ind_plot]
    true_field2 = torch.concat([true_field2_1,true_field2_2])
    true_pdf2, bins_true2 = np.histogram(true_field2.cpu(), bins=150, density=True)
    bins_true2 = (bins_true2[1:] + bins_true2[:-1]) / 2

   
    # del e2.train_data

    total = len(e3.train_data)
    print(total)
    start = total - 1000 - 1
    end = start + 1000
    true_field3 = (
        e3.train_data[:][1][
            :, ind_plot, e3.wet_bool
        ].flatten()
        * e3.std_out[ind_plot]
    ) + e3.mean_out[ind_plot]
    true_pdf3, bins_true3 = np.histogram(true_field3.cpu(), bins=150, density=True)
    bins_true3 = (bins_true3[1:] + bins_true3[:-1]) / 2
    # del e3.train_data

    pdf[ind_plot] = {
        "true_pdf1": true_pdf1,
        "true1": [bins_true1, true_pdf1],
        "true_pdfblended": true_pdf2,
        "trueblended": [bins_true2, true_pdf2],
        "true_pdf2x": true_pdf3,
        "true2x": [bins_true3, true_pdf3],
    }

    

    return pdf
    
ground_pdfs = get_train_pdf(e1, e3, e4)


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cmocean
from pathlib import Path
import numpy as np
import cartopy.crs as ccrs
import cartopy as cart
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def plot_temp_pdf(
    network_names,
    region,
    output_dir,
    pdf1,
    pdf2,
    pdf3,
    true_pdf,
    JUPYTER_MODE=False,
):
    plt.style.use("bmh")

    # Long KE
    plt.rcParams.update({"font.size": 12})
    fig, axs = plt.subplots(
        2,
        2,
        figsize=(12, 5),
        gridspec_kw={
            "width_ratios": [1, 1],
            "height_ratios": [1, 1],
            "wspace": 0.3,
            "hspace": 0.6,
        }
    )

    clist = ["#A00B41", "#3300EA", "#00DCDE", "#A6BD00"]

    # PDF
    var_list = {
        "1": r"$\overline{v}$ $( m/s )$",
        "0": r"$\overline{u}$ $( m/s )$",
        "2": r"$\overline{T}$ $( ^\circ C )$",
        "KE": r"$\overline{KE}$",
    }

    ind_plot = 2
    # Top left
    for i, network_name in enumerate(network_names):
        axs[0,0].semilogy(
            *pdf1[ind_plot][network_name],
            lw=2,
            color=clist[i],
            label=f"{network_name}",
        )

    axs[0,0].set_ylim(
        [
            0.01,
            pdf1[ind_plot]["true_pdf"].max(),
        ]
    )
    
    axs[0,0].set_xlim(
        [
            -3,31
        ]
    )

    axs[0,0].semilogy(*pdf1[ind_plot]["true"], lw=2, c="k", ls='--', label="CM2.6")
    axs[0,0].legend(bbox_to_anchor=(0, 1.2, 1, 0.2), loc="lower left", fancybox=True, ncol=len(pdf1[2].keys()))
    

    axs[0,0].set_xlabel(var_list[str(ind_plot)])
    
    axs[0,0].set_ylabel(r"${p(}$" + var_list[str(ind_plot)][:14] + "${)}$")
    axs[0,0].set_title('PI Data - PI Data')

    # # Top Right
    for i, network_name in enumerate(network_names):
        axs[0,1].semilogy(
            *pdf2[ind_plot][network_name],
            lw=2,
            color=clist[i],
            label=f"{network_name}",
        )

    axs[0,1].set_ylim(
        [
            0.01,
            pdf2[ind_plot]["true_pdf"].max(),
        ]
    )
    
    axs[0,1].set_xlim(
        [
            -3,32
        ]
    )

    axs[0,1].semilogy(*pdf2[ind_plot]["true"], lw=2, c="k", ls='--', label="CM2.6")
    

    axs[0,1].set_xlabel(var_list[str(ind_plot)])
    
    axs[0,1].set_ylabel(r"${p(}$" + var_list[str(ind_plot)][:14] + "${)}$")
    axs[0,1].set_title('PI Data - 2x CO2')

    # # Bottom Right
    for i, network_name in enumerate(network_names):
        axs[1,1].semilogy(
            *pdf3[ind_plot][network_name],
            lw=2,
            color=clist[i],
            label=f"{network_name}",
        )

    axs[1,1].set_ylim(
        [
            0.01,
            pdf3[ind_plot]["true_pdf"].max(),
        ]
    )
    
    axs[1,1].set_xlim(
        [
            -3,32
        ]
    )

    axs[1,1].semilogy(*pdf3[ind_plot]["true"], lw=2, c="k", ls='--', label="CM2.6")
    

    axs[1,1].set_xlabel(var_list[str(ind_plot)])
    
    axs[1,1].set_ylabel(r"${p(}$" + var_list[str(ind_plot)][:14] + "${)}$")
    axs[1,1].set_title('Blended Data - 4x CO2')

    # Bottom Left
    clist2 = ['#d7191c','#abd9e9','#2c7bb6','#fdae61']
    axs[1,0].semilogy(
        *true_pdf[ind_plot]["true1"],
        lw=2,
        color=clist2[0],
        label="PI Data",
    )
    axs[1,0].semilogy(
        *true_pdf[ind_plot]["true2x"],
        lw=2,
        color=clist2[1],
        label="2x CO2",
    )
    axs[1,0].semilogy(
        *true_pdf[ind_plot]["trueblended"],
        lw=2,
        color=clist2[2],
        label="Blended Data",
    )
    axs[1,0].semilogy(*pdf3[ind_plot]["true"], lw=2, c=clist2[3], label="4x CO2")

    axs[1,0].set_ylim(
        [
            0.01,
            true_pdf[ind_plot]["true_pdf1"].max(),
        ]
    )
    
    axs[1,0].set_xlim(
        [
            23,32
        ]
    )
    
    axs[1,0].set_xlabel(var_list[str(ind_plot)])
    
    axs[1,0].set_ylabel(r"${p(}$" + var_list[str(ind_plot)][:14] + "${)}$")
    axs[1,0].set_title('Ground Truth')
    axs[1,0].legend(loc='upper center', bbox_to_anchor=(0, -0.6, 1, 0.2), fancybox=True, ncol=4)


    # plt.show()

    plt.savefig(
        Path(output_dir) / ("PDF" + region + "_" + str(ind_plot) + ".png"),
        bbox_inches="tight",
    )
    plt.clf()

plot_temp_pdf(
    e1.pred_names + [e1.network],
    e1.region + '_Long_',
    e1.output_dir,
    pdf1,
    pdf2,
    pdf3,
    ground_pdfs,
    e1.JUPYTER_MODE
)