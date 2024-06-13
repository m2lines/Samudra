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
            "Train_"
            + args.train_region
            + "_Test_"
            + args.region
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
            inputs, extra_in, outputs = gen_data_global_new(
                self.inputs, self.extra_in, self.outputs, args.lag
            )
        elif "global_2x" == args.region:
            inputs, extra_in, outputs = gen_data_global_new(
                self.inputs, self.extra_in, self.outputs, args.lag, run_type="2x"
            )
        elif "global_4x" == args.region:
            inputs, extra_in, outputs = gen_data_global_new(
                self.inputs, self.extra_in, self.outputs, args.lag, run_type="4x"
            )
        else:
            raise NotImplementedError

        print("Calculating mask tensors")
        self.wet, self.wet_nan = get_wet_mask(inputs, "cpu")
        self.wet_bool = np.array(self.wet.cpu()).astype(bool)
        wet_lap = compute_laplacian_wet(self.wet_nan, 4)  # hardcoded
        wet_lap = xr.where(wet_lap == 0, 1, np.nan)
        self.wet_lap = np.nan_to_num(wet_lap)
        print("Wet resolution:", self.wet.shape)

        self.time_vec = inputs[0].time.data

        self.time_test = self.time_vec[e_test : (e_test + args.lag * args.N_test)]

        print("Loading Train data")
        train_data = torch.load(
            Path(args.data_dir) / "train_data_cnn_{0}.pt".format(self.str_train),
            map_location=torch.device("cpu"),
        )
        # self.train_data = train_data

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
                    assert len(norm_vals) == len(GLOBAL_COMBINED_STATS) and all(
                        np.array_equal(norm_vals[k], GLOBAL_COMBINED_STATS[k])
                        for k in norm_vals
                    )
                self.test_data = data_CNN_Dynamic(
                    data_in_test,
                    data_out_test,
                    self.wet.to(device="cpu"),
                    norm_vals,
                    device=args.device,
                )
                del train_data
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
        self.grids = xr.open_dataset(
            "/scratch/as15415/Data/CM2x_grids/Grid_New.nc"
        ).rename({"dx": "dxu", "dy": "dyu"})

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
with initialize_config_dir(
    version_base=None,
    config_dir="/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/configs",
):
    args1 = compose(
        config_name="exp/eval_swin_global",
        overrides=[
            "output_dir=./notebooks/temp/{0}_indices".format(str(datetime.now())[:10]),
            "model_name_replace=Swin",
            "network=Foundation Swin Train1Eval1",
            "train_region=global_1",
            "region=global_1",
            "swin.embed_dim=60",
            "exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample",
            "ckpt_path=/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-05-11-foundation_train_swintrans60_global_1/swintrans60/saved_nets/swin_best_steps_4_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt",
            "pred_names=['UNet (Baseline)', 'ConvNext UNet']",
            "pred_paths=['/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/Preds/Foundation Adam UNet Train1Eval1_Train_global_1_Test_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth', '/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/Preds/Foundation ConvNext UNet Train1Eval1_Train_global_1_Test_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth']",
        ],
    )
if not os.path.exists(args1.output_dir):
    os.mkdir(args1.output_dir)

e1 = Eval(args1)

# G1, G2x
with initialize_config_dir(
    version_base=None,
    config_dir="/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/configs",
):
    args2 = compose(
        config_name="exp/eval_swin_global",
        overrides=[
            "output_dir=./notebooks/temp/{0}_indices".format(str(datetime.now())[:10]),
            "model_name_replace=Swin",
            "network=Foundation Swin Train1Eval2x",
            "train_region=global_1",
            "region=global_2x",
            "swin.embed_dim=60",
            "exp/modules/blocks@swin.up_sampling_block=transposed_conv_upsample",
            "ckpt_path=/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train/2024-05-11-foundation_train_swintrans60_global_1/swintrans60/saved_nets/swin_best_steps_4_global_1_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth.pt",
            "pred_names=['UNet (Baseline)', 'ConvNext UNet']",
            "pred_paths=['/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/Preds/Foundation Adam UNet Train1Eval2x_Train_global_1_Test_global_2x_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth', '/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/Preds/Foundation ConvNext UNet Train1Eval2x_Train_global_1_Test_global_2x_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_4000_Lateral_Data_025_no_smooth']",
        ],
    )

e2 = Eval(args2)

# G1_2x, G_4x
with initialize_config_dir(
    version_base=None,
    config_dir="/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/configs",
):
    args3 = compose(
        config_name="exp/eval_swin_global",
        overrides=[
            "output_dir=./notebooks/temp/{0}_indices".format(str(datetime.now())[:10]),
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
            "pred_paths=['/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/Preds/Foundation Adam UNet Train12xEval4x_Train_combined_global_1_Test_global_4x_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_0_Lateral_Data_025_no_smooth', '/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/Preds/Foundation ConvNext UNet Train12xEval4x_Train_combined_global_1_Test_global_4x_Test_in_u_v_T_ext_tau_u_tau_v_t_ref__outu_v_T_N_train_0_Lateral_Data_025_no_smooth']",
        ],
    )

e3 = Eval(args3)


# Sending the data back to cpu
e1.send_data_to_cpu()
e2.send_data_to_cpu()
e3.send_data_to_cpu()


def get_indices(e, model_pred_net, model_pred_saved_nets, long=False):
    print("Getting Nino34...")
    nino_net, nino_true = compute_nino34(
        e.grids,
        e.inputs,
        model_pred_net,
        e.test_data,
        e.mean_out,
        e.std_out,
        e.time_test,
    )
    nino_saved = []
    for model_pred_saved in model_pred_saved_nets:
        nino_net_i, nino_true = compute_nino34(
            e.grids,
            e.inputs,
            model_pred_saved,
            e.test_data,
            e.mean_out,
            e.std_out,
            e.time_test,
        )
        nino_saved.append(nino_net_i)

    # print("Plotting Nino34...")
    # plot_region_based_metric(
    #     e.pred_names + [e.network],
    #     e.region if not long else e.region + '_Long_',
    #     e.str_save,
    #     e.output_dir,
    #     nino_true,
    #     nino_saved + [nino_net],
    #     e.JUPYTER_MODE,
    #     mode='nino34'
    # )

    print("Getting Amo...")
    amo_net, amo_true = compute_amo(
        e.grids,
        e.inputs,
        model_pred_net,
        e.test_data,
        e.mean_out,
        e.std_out,
        e.time_test,
    )
    amo_saved = []
    for model_pred_saved in model_pred_saved_nets:
        amo_net_i, amo_true = compute_amo(
            e.grids,
            e.inputs,
            model_pred_saved,
            e.test_data,
            e.mean_out,
            e.std_out,
            e.time_test,
        )
        amo_saved.append(amo_net_i)

    return nino_true, nino_saved + [nino_net], amo_true, amo_saved + [amo_net]


model_pred_net, model_pred_saved_nets = e1.load_long_data()
nino_true1, nino_saved1, amo_true1, amo_saved1 = get_indices(
    e1, model_pred_net, model_pred_saved_nets, True
)
model_pred_net, model_pred_saved_nets = e2.load_long_data()
nino_true2, nino_saved2, amo_true2, amo_saved2 = get_indices(
    e2, model_pred_net, model_pred_saved_nets, True
)
model_pred_net, model_pred_saved_nets = e3.load_long_data()
nino_true3, nino_saved3, amo_true3, amo_saved3 = get_indices(
    e3, model_pred_net, model_pred_saved_nets, True
)


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cmocean
from pathlib import Path
import numpy as np
import cartopy.crs as ccrs
import cartopy as cart
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def plot_both_region_based_metric(
    network_names,
    region,
    save_str,
    output_dir,
    true_nino1,
    indices_nino1,
    true_nino2,
    indices_nino2,
    true_nino3,
    indices_nino3,
    true_amo1,
    indices_amo1,
    true_amo2,
    indices_amo2,
    true_amo3,
    indices_amo3,
    JUPYTER_MODE=False,
):

    plt.style.use("bmh")

    # Long KE
    plt.rcParams.update({"font.size": 12})
    fig, axs = plt.subplots(
        2,
        3,
        figsize=(12, 5),
        gridspec_kw={
            "width_ratios": [1, 1, 1],
            "height_ratios": [1, 1],
            "wspace": 0.3,
            "hspace": 0.3,
        },
    )

    clist = ["#A00B41", "#3300EA", "#00DCDE", "#A6BD00"]

    y = "Nino 3.4 Index"

    # 1
    k = 0
    N_plot = len(indices_nino1[0])
    for i, indices_i in enumerate(indices_nino1):
        if indices_i is not None:
            axs[0, k].plot(
                np.arange(1, N_plot + 1),
                indices_i,
                c=clist[i],
                label=f"{network_names[i]}",
            )
    axs[0, k].plot(np.arange(1, N_plot + 1), true_nino1, "--k", label="CM2.6")
    # axs[0, k].set_xlabel(r"time $( days )$", fontsize="15")
    axs[0, k].set_ylabel(y, fontsize="15")
    axs[0, k].set_title("PI Data - PI Data")
    axs[0, k].legend(
        bbox_to_anchor=(0, 1.2, 1, 0.2),
        loc="lower left",
        fancybox=True,
        fontsize="15",
        ncol=len(indices_nino1) + 1,
    )

    # 2
    k = 1
    N_plot = len(indices_nino2[0])
    for i, indices_i in enumerate(indices_nino2):
        if indices_i is not None:
            axs[0, k].plot(
                np.arange(1, N_plot + 1),
                indices_i,
                c=clist[i],
                label=f"{network_names[i]}",
            )
    axs[0, k].plot(np.arange(1, N_plot + 1), true_nino2, "--k", label="CM2.6")
    axs[0, k].set_title("PI Data - 2x CO2")

    # 3
    k = 2
    N_plot = len(indices_nino3[0])
    for i, indices_i in enumerate(indices_nino3):
        if indices_i is not None:
            axs[0, k].plot(
                np.arange(1, N_plot + 1),
                indices_i,
                c=clist[i],
                label=f"{network_names[i]}",
            )
    axs[0, k].plot(np.arange(1, N_plot + 1), true_nino3, "--k", label="CM2.6")
    axs[0, k].set_title("Blended Data - 4x CO2")

    # AMO
    y = "AMO Index"
    # 1
    k = 0
    N_plot = len(indices_amo1[0])
    for i, indices_i in enumerate(indices_amo1):
        if indices_i is not None:
            axs[1, k].plot(
                np.arange(1, N_plot + 1),
                indices_i,
                c=clist[i],
                label=f"{network_names[i]}",
            )
    axs[1, k].plot(np.arange(1, N_plot + 1), true_amo1, "--k", label="CM2.6")
    axs[1, k].set_xlabel(r"time $( days )$", fontsize="15")
    axs[1, k].set_ylabel(y, fontsize="15")

    # 2
    k = 1
    N_plot = len(indices_amo2[0])
    for i, indices_i in enumerate(indices_amo2):
        if indices_i is not None:
            axs[1, k].plot(
                np.arange(1, N_plot + 1),
                indices_i,
                c=clist[i],
                label=f"{network_names[i]}",
            )
    axs[1, k].plot(np.arange(1, N_plot + 1), true_amo2, "--k", label="CM2.6")
    axs[1, k].set_xlabel(r"time $( days )$", fontsize="15")
    # axs[1, k].set_ylabel(y, fontsize="15")

    # 3
    k = 2
    N_plot = len(indices_amo3[0])
    for i, indices_i in enumerate(indices_amo3):
        if indices_i is not None:
            axs[1, k].plot(
                np.arange(1, N_plot + 1),
                indices_i,
                c=clist[i],
                label=f"{network_names[i]}",
            )
    axs[1, k].plot(np.arange(1, N_plot + 1), true_amo3, "--k", label="CM2.6")
    axs[1, k].set_xlabel(r"time $( days )$", fontsize="15")
    # axs[1, k].set_ylabel(y, fontsize="15")

    # plt.show()

    plt.savefig(
        Path(output_dir) / ("Indexplots" + "_" + region + "_" + save_str + ".png"),
        bbox_inches="tight",
    )
    plt.clf()


# Plotting ENSO
plot_both_region_based_metric(
    e1.pred_names + [e1.network],
    e1.region + "_Long_",
    e1.str_save,
    e1.output_dir,
    nino_true1,
    nino_saved1,
    nino_true2,
    nino_saved2,
    nino_true3,
    nino_saved3,
    amo_true1,
    amo_saved1,
    amo_true2,
    amo_saved2,
    amo_true3,
    amo_saved3,
    e1.JUPYTER_MODE,
)
