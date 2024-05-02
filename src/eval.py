from constants import INPT_VARS, EXTRA_VARS, OUT_VARS, REGIONS
import hydra
from hydra.utils import instantiate
from pathlib import Path
import os
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

from utils.data_utils import (
    get_wet_mask,
    get_train_test_ranges,
    gen_data_in_test,
    gen_data_out_test,
    data_CNN_Lateral,
    data_CNN_Dynamic,
    gen_data_025_lateral,
    gen_data_global,
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
    gen_KE_spectrum,
    gen_KE,
    gen_KE_range,
    gen_enstrophy_spectrum,
    gen_enstrophy,
    compute_corrs_single,
    compute_ACC_single,
    compute_RMSE_single,
)
from utils.subgrid_utils import coarse_grid, get_area_tensor
from utils.climate_utils import compute_laplacian_wet
from utils.plot_utils import (
    plot_short_time_stats,
    plot_long_time_stats,
    plot_long_KE,
    plot_metrics_KE_spectrum,
    plot_metrics_KE,
    plot_metrics_enstrophy_spectrum,
    plot_metrics_entrophy,
    plot_metrics_corr,
    plot_metrics_rmse,
    plot_metrics_acc,
    plot_metrics_pdf,
    get_initial_snapshot_fig,
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
        self.str_video = (
            "steps_"
            + str(args.steps)
            + "_"
            + args.region
            + "_Test_in_"
            + self.str_in
            + "ext_"
            + self.str_ext
            + "_out"
            + self.str_out
            + "N_train_"
            + str(args.N_samples)
            + "_Lateral_Data_025_no_smooth"
        )
        self.str_save = (
            "steps_"
            + str(args.steps)
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
            args.region
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
        if "global" in args.region:
            inputs, extra_in, outputs = gen_data_global(
                self.inputs, self.extra_in, self.outputs, args.lag
            )
        else:
            inputs, extra_in, outputs = gen_data_025_lateral(
                self.inputs,
                self.extra_in,
                self.outputs,
                args.lag,
                REGIONS[args.region]["lat"],
                REGIONS[args.region]["lon"],
                args.Nb,
            )

        print("Calculating mask tensors")
        self.wet, self.wet_nan = get_wet_mask(inputs, "cpu")
        self.wet_bool = np.array(self.wet.cpu()).astype(bool)
        wet_lap = compute_laplacian_wet(self.wet_nan, args.Nb)
        wet_lap = xr.where(wet_lap == 0, 1, np.nan)
        self.wet_lap = np.nan_to_num(wet_lap)
        print("Wet resolution:", self.wet.shape)

        self.time_vec = inputs[0].time.data

        self.time_test = self.time_vec[e_test : (e_test + args.lag * args.N_test)]

        if args.save_test_data:
            print("Saving data")
            data_in_test = gen_data_in_test(
                0, e_test, args.N_test, args.lag, args.hist, inputs, extra_in
            )
            data_out_test = gen_data_out_test(
                0, e_test, args.N_test, args.lag, args.hist, outputs
            )
            if "global" in args.region:
                self.test_data = data_CNN_Dynamic(
                    data_in_test,
                    data_out_test,
                    self.wet.to(device="cpu"),
                    device=args.device,
                )
            else:
                self.test_data = data_CNN_Lateral(
                    data_in_test,
                    data_out_test,
                    self.wet.to(device="cpu"),
                    self.N_atm,
                    args.Nb,
                    device=args.device,
                )
            torch.save(
                self.test_data,
                Path(args.data_dir) / "test_data_cnn_{0}.pt".format(self.str_save),
            )

        else:
            print("Loading test data")
            self.test_data = torch.load(
                Path(args.data_dir) / "test_data_cnn_{0}.pt".format(self.str_save)
            )

        # Model
        if "swin" in args.network.lower():
            print("Loading model swin")
            model = instantiate(
                args.swin,
                in_channels=self.num_in,
                output_channels=self.N_in,
                pretrain_img_size=[*self.test_data[0][0].shape[1:]],
            )
        elif "unet" in args.network.lower():
            print("Loading model unet")
            model = instantiate(
                args.unet, input_channels=self.num_in, output_channels=self.N_in
            )
            model.set_input_size([*self.test_data[0][0].shape[1:]])

        full_model_path = args.ckpt_path
        self.full_model_name = args.network + "_" + self.post_model_name
        self.output_channels = model.output_channels

        # from torchinfo import summary
        # # summary(model)
        # i = [torch.zeros(1, 9, 128, 192)] * 2
        # summary(model, input_data=[i], col_names=["output_size", "num_params"], depth=4)
        # import pdb; pdb.set_trace()

        model = model.to(args.device)
        model.load_state_dict(
            torch.load(full_model_path, map_location=torch.device(args.device))
        )

        self.model = model

        # Stats
        self.mean_out = self.test_data.norm_vals["m_out"]
        self.std_out = self.test_data.norm_vals["s_out"]
        self.mean_in = self.test_data.norm_vals["m_in"]
        self.std_in = self.test_data.norm_vals["s_in"]

        # clim
        self.clim = None
        if args.run_short_pred or args.run_plot_metrics or args.run_long_metrics:
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
        self.grids = xr.open_dataset(args.grid_path)
        if "global" in args.region:
            self.grids = coarse_grid(self.grids, args.factor)

        else:
            self.grids = self.grids.sel(
                {
                    "yu_ocean": slice(*REGIONS[args.region]["lat"]),
                    "xu_ocean": slice(*REGIONS[args.region]["lon"]),
                }
            )

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
        self.output_dir = args.output_dir
        self.region = args.region
        self.steps = args.steps
        self.network = args.network

        self.pred_region = args.pred_region
        self.pred_names = args.pred_names
        self.pred_paths = args.pred_paths

    def generate_pred_lateral(self):
        print("Generation Pred begin...")
        model_pred = None
        old_pred = None
        for ns in [4000]:
            for rand_ind in range(1, 4):
                print(ns, rand_ind)
                # torch.manual_seed(rand_ind)
                # torch.cuda.manual_seed(rand_ind)
                # import numpy as np
                # np.random.seed(rand_ind)
                # new_test_data = copy.deepcopy(self.test_data)
                old_pred = model_pred
                model_pred = generate_model_rollout(
                    self.N_test,
                    self.test_data,
                    self.model,
                    self.hist,
                    self.N_in,
                    self.N_extra,
                    self.Nb,
                    self.region
                )

                print("data_gen")
                da = xr.DataArray(
                    data=model_pred,
                    dims=["time", "x", "y", "var"],
                )

                da.to_zarr(
                    self.pred_model_path
                    / (
                        "Pred_lateral_Fast_Data_025_"
                        + self.post_pred_name
                        + "_rand_seed_"
                        + str(rand_ind)
                        + ".zarr"
                    ),
                    mode="w",
                )
                print(f"Model pred shape {model_pred.shape}")
                # del model_pred
        print((old_pred == model_pred).all())

    def generate_short_pred_lateral(self):
        print("Generation Short Pred begin...")
        N_run = 5
        len_run = 200

        model_pred = None
        old_pred = None
        for ns in [4000]:
            for rand_ind in range(1, 4):
                data_shape = self.test_data[0][1].shape
                old_pred = model_pred
                model_pred = np.zeros(
                    (int(N_run * len_run), data_shape[1], data_shape[2], data_shape[0])
                )

                for i in range(N_run):
                    print(ns, rand_ind)
                    temp = copy.deepcopy(self.test_data)
                    temp.input = temp.input[int(i * len_run) : int((i + 1) * len_run)]
                    temp.output = temp.output[int(i * len_run) : int((i + 1) * len_run)]
                    temp.size = len_run

                    model_pred_temp = generate_model_rollout(
                        len_run,
                        temp,
                        self.model,
                        self.hist,
                        self.N_in,
                        self.N_extra,
                        self.Nb,
                        self.region
                    )
                    print("data_gen")
                    model_pred[int(i * len_run) : int((i + 1) * len_run)] = (
                        model_pred_temp
                    )

                da = xr.DataArray(
                    data=model_pred,
                    dims=["time", "x", "y", "var"],
                )

                da.to_zarr(
                    self.pred_model_path
                    / (
                        "Pred_Short_Data_025_"
                        + self.post_pred_name
                        + "_rand_seed_"
                        + str(rand_ind)
                        + ".zarr"
                    ),
                    mode="w",
                )
                print(f"Model pred shape {model_pred.shape}")
        print((old_pred == model_pred).all())

    ### Need to Refactor the following functions
    def compare_pred_lateral(self):
        def get_stats(
            zarr_path,
            region,
            rand_int,
            str_in,
            str_ext,
            test_data,
            area,
            wet_bool,
            N_mean,
            lag,
        ):
            try:
                model_pred_atm = (
                    xr.open_zarr(
                        Path(zarr_path)
                        / (
                            "Pred_lateral_Fast_Data_025_"
                            + region
                            + "_in_"
                            + str_in
                            + "ext_"
                            + str_ext
                            + "N_samples_"
                            + str(4000)
                            + "_rand_seed_"
                            + str(rand_int)
                            + ".zarr"
                        )
                    )
                    .sel(time=slice(0, N_mean))
                    .to_array()
                    .to_numpy()
                    .squeeze()
                )
            except Exception as error:
                print(error)
                print(zarr_path)
                # raise Exception(
                #     f"Path in {zarr_path} does not exist. Make sure to set run_gen_pred to True in config."
                # )

            mean_atm, auto_mean = compute_mean(
                N_mean, test_data, model_pred_atm, area.cpu(), wet_bool
            )
            var_atm, auto_var = compute_var(
                N_mean, test_data, model_pred_atm, area.cpu(), wet_bool
            )
            rmse_atm, auto_rmse = compute_rmse(
                np.min((500, N_mean)), test_data, model_pred_atm, area.cpu(), wet_bool
            )
            corrs_atm, auto_corrs = compute_corrs(
                np.min((500, N_mean)), test_data, model_pred_atm, wet_bool
            )
            KE, auto_KE = compute_KE(N_mean, test_data, model_pred_atm, area, wet_bool)
            freqs, FFT, auto_FFT = compute_time_spec(N_mean, auto_mean, mean_atm, lag)

            return (
                model_pred_atm,
                mean_atm,
                auto_mean,
                rmse_atm,
                auto_rmse,
                corrs_atm,
                auto_corrs,
                KE,
                auto_KE,
                freqs,
                FFT,
                auto_FFT,
                var_atm,
                auto_var,
            )

        def get_spred(
            zarr_path,
            region,
            num_IC,
            str_in,
            str_ext,
            test_data,
            area,
            wet_bool,
            N_mean,
            lag,
        ):
            mean = np.zeros((num_IC, N_mean, 3))
            var = np.zeros((num_IC, N_mean, 3))
            KE = np.zeros((num_IC, N_mean))
            rmse = np.zeros((num_IC, np.min((500, N_mean)), 3))
            corrs = np.zeros((num_IC, np.min((500, N_mean)), 3))
            FFTs = np.zeros((num_IC, int(N_mean / 2 + 1), 3))

            for i in range(0, num_IC):
                (
                    out,
                    mean_1,
                    out,
                    rmse_1,
                    out,
                    corrs_1,
                    out,
                    KE_1,
                    out,
                    freqs,
                    FFT_1,
                    out,
                    var_1,
                    out,
                ) = get_stats(
                    zarr_path,
                    region,
                    i + 1,
                    str_in,
                    str_ext,
                    test_data,
                    area,
                    wet_bool,
                    N_mean,
                    lag,
                )
                KE[i] = KE_1
                mean[i] = mean_1
                rmse[i] = rmse_1
                corrs[i] = corrs_1
                FFTs[i] = FFT_1
                var[i] = var_1
            return mean, rmse, corrs, KE, FFTs, freqs, var

        print("Long time stats compute begin...")
        mean_net, rmse_net, corrs_net, KE_net, FFTs_net, freqs, var_net = get_spred(
            self.pred_model_path,
            self.region,
            3,
            self.str_in,
            self.str_ext,
            self.test_data,
            self.area,
            self.wet_bool,
            self.N_test,
            self.lag,
        )
        (
            model_pred_net,
            m_net,
            auto_mean,
            r_net,
            auto_rmse,
            c_net,
            auto_corrs,
            K_net,
            auto_KE,
            freqs,
            F_net,
            auto_FFT,
            v_net,
            auto_var,
        ) = get_stats(
            self.pred_model_path,
            self.region,
            1,
            self.str_in,
            self.str_ext,
            self.test_data,
            self.area,
            self.wet_bool,
            self.N_test,
            self.lag,
        )  # zarr_path, region, rand_int, str_in, str_ext, test_data, area, wet_bool, N_mean, lag

        # model_pred_saved_nets = []
        FFTs_saved = []
        means_saved = []
        for model_pred_path in self.pred_paths:
            mean_i, _, _, _, FFTs_i, freqs, _ = get_spred(
                model_pred_path,
                self.pred_region,
                3,
                self.str_in,
                self.str_ext,
                self.test_data,
                self.area,
                self.wet_bool,
                self.N_test,
                self.lag,
            )
            FFTs_saved.append(FFTs_i)
            means_saved.append(mean_i)

        print("Long time stats plot begin...")

        plot_long_time_stats(
            self.pred_names + [self.network],
            self.region,
            self.str_save,
            self.output_dir,
            self.N_test,
            self.lag,
            freqs,
            auto_FFT,
            FFTs_saved + [FFTs_net],
            auto_mean,
            means_saved + [mean_net],
        )

    def compare_short_pred_lateral(self):
        def get_stats(
            zarr_path,
            region,
            N_IC,
            rand_int,
            str_in,
            str_ext,
            test_data,
            clim,
            time,
            area,
            wet_bool,
            N_mean,
        ):
            try:
                model_pred_atm = (
                    xr.open_zarr(
                        Path(zarr_path)
                        / (
                            "Pred_Short_Data_025_"
                            + region
                            + "_in_"
                            + str_in
                            + "ext_"
                            + str_ext
                            + "N_samples_"
                            + str(4000)
                            + "_rand_seed_"
                            + str(rand_int)
                            + ".zarr"
                        )
                    )
                    .to_array()
                    .to_numpy()
                    .squeeze()
                )
            except:
                print(
                    "Path does not exist. Make sure to set run_gen_short_pred to True in config."
                )
            temp = copy.deepcopy(test_data)
            temp.input = temp.input[int((N_IC - 1) * N_mean) : int((N_IC) * N_mean)]
            temp.output = temp.output[
                int((N_IC - 1) * N_mean) : int((N_IC + 1) * N_mean)
            ]
            temp.size = N_mean
            rmse_atm, auto_rmse = compute_rmse(
                N_mean,
                temp,
                model_pred_atm[int((N_IC - 1) * N_mean) : int((N_IC) * N_mean)],
                area.cpu(),
                wet_bool,
            )
            corrs_atm, auto_corrs = compute_corrs_area(
                N_mean,
                temp,
                model_pred_atm[int((N_IC - 1) * N_mean) : int((N_IC) * N_mean)],
                area.cpu(),
                wet_bool,
            )
            ACC_atm, auto_ACC = compute_ACC(
                N_mean,
                temp,
                model_pred_atm[int((N_IC - 1) * N_mean) : int((N_IC) * N_mean)],
                clim,
                time,
                area.cpu(),
                wet_bool,
            )
            KE, auto_KE = compute_KE(
                N_mean,
                temp,
                model_pred_atm[int((N_IC - 1) * N_mean) : int((N_IC) * N_mean)],
                area,
                wet_bool,
            )
            return (
                rmse_atm,
                auto_rmse,
                corrs_atm,
                auto_corrs,
                ACC_atm,
                auto_ACC,
                KE,
                auto_KE,
            )

        def get_spred(
            zarr_path,
            region,
            N_IC,
            num_IC,
            str_in,
            str_ext,
            test_data,
            clim,
            time,
            area,
            wet_bool,
            N_mean,
        ):
            KE = np.zeros((int(num_IC * N_IC), N_mean))
            rmse = np.zeros((int(num_IC * N_IC), N_mean, 3))
            corrs = np.zeros((int(num_IC * N_IC), N_mean, 3))
            ACC = np.zeros((int(num_IC * N_IC), N_mean, 3))

            auto_KE = np.zeros((N_IC, N_mean))
            auto_rmse = np.zeros((N_IC, N_mean, 3))
            auto_corrs = np.zeros((N_IC, N_mean, 3))
            auto_ACC = np.zeros((N_IC, N_mean, 3))

            for i in range(0, num_IC):
                print(i)
                for j in range(0, N_IC):
                    (
                        rmse_1,
                        auto_rmse_1,
                        corrs_1,
                        auto_corrs_1,
                        ACC_1,
                        auto_acc_1,
                        KE_1,
                        auto_KE_1,
                    ) = get_stats(
                        zarr_path,
                        region,
                        j + 1,
                        i + 1,
                        str_in,
                        str_ext,
                        test_data,
                        clim,
                        time,
                        area,
                        wet_bool,
                        N_mean,
                    )
                    KE[int(i * N_IC + j)] = KE_1
                    rmse[int(i * N_IC + j)] = rmse_1
                    corrs[int(i * N_IC + j)] = corrs_1
                    ACC[int(i * N_IC + j)] = ACC_1

                    if i == 0:
                        auto_rmse[j] = auto_rmse_1
                        auto_KE[j] = auto_KE_1
                        auto_corrs[j] = auto_corrs_1
                        auto_ACC[j] = auto_acc_1

            return rmse, corrs, ACC, KE, auto_rmse, auto_corrs, auto_ACC, auto_KE

        print("Short time stats compute begin...")

        (
            rmse_net,
            corrs_net,
            ACC_net,
            KE_net,
            auto_rmse,
            auto_corrs,
            auto_ACC,
            auto_KE,
        ) = get_spred(
            self.pred_model_path,
            self.region,
            5,
            3,
            self.str_in,
            self.str_ext,
            self.test_data,
            self.clim,
            self.time_test,
            self.area,
            self.wet_bool,
            200,
        )

        ACC_saved = []
        RMSE_saved = []
        Corrs_saved = []
        KE_saved = []
        for model_pred_path in self.pred_paths:
            (
                rmse_i,
                corrs_i,
                ACC_i,
                KE_i,
                auto_rmse,
                auto_corrs,
                auto_ACC,
                auto_KE,
            ) = get_spred(
                model_pred_path,
                self.pred_region,
                5,
                3,
                self.str_in,
                self.str_ext,
                self.test_data,
                self.clim,
                self.time_test,
                self.area,
                self.wet_bool,
                200,
            )
            ACC_saved.append(ACC_i)
            RMSE_saved.append(rmse_i)
            Corrs_saved.append(corrs_i)
            KE_saved.append(KE_i)

        print("Short time stats plot begin...")
        plot_short_time_stats(
            self.pred_names + [self.network],
            self.region,
            self.str_save,
            self.output_dir,
            self.N_test,
            self.lag,
            auto_ACC,
            ACC_saved + [ACC_net],
            auto_rmse,
            RMSE_saved + [rmse_net],
            auto_KE,
            KE_saved + [KE_net],
            auto_corrs,
            Corrs_saved + [corrs_net],
        )

    def plot_metrics(self):
        print("Plot metrics begin...")
        model_pred_net = None
        model_pred_net = (
            xr.open_zarr(
                self.pred_model_path
                / (
                    "Pred_Short_Data_025_"
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
                "Pred_Short_Data_025_"
                + self.pred_region
                + "_in_"
                + self.str_in
                + "ext_"
                + "tau_u_tau_v_t_ref_"
                + "N_samples_"
                + str(4000)
                + "_rand_seed_"
                + str(1)
                + ".zarr"
            )

            model_pred_saved_nets.append(
                xr.open_zarr(net_path).to_array().to_numpy().squeeze()
            )

        ### Long time KE
        print("Getting mean KE stats...")
        start = 0
        N_plot = 1000

        long_KE_net, long_KE_true = gen_KE_range(
            start, N_plot, self.test_data, model_pred_net
        )
        long_KE_net = long_KE_net.mean(0)
        long_KE_true = long_KE_true.mean(0)

        long_KE_saved = []
        for model_pred_saved in model_pred_saved_nets:
            long_KE_savedi, long_KE_true = gen_KE_range(
                start, N_plot, self.test_data, model_pred_saved
            )
            long_KE_savedi = long_KE_savedi.mean(0)
            long_KE_true = long_KE_true.mean(0)
            long_KE_saved.append(long_KE_savedi)

        print("Plotting mean KE...")
        plot_long_KE(
            self.pred_names + [self.network],
            self.region,
            self.str_save,
            self.output_dir,
            self.grids,
            self.Nb,
            self.wet_nan,
            long_KE_true,
            long_KE_saved + [long_KE_net],
        )

        ### Short time scale metrics
        N_plot = 200

        # KE
        print("Getting Short KE stats...")
        KE_spec_net, KE_spec_true = gen_KE_spectrum(
            N_plot, self.test_data, model_pred_net, self.grids, self.wet
        )

        KE_spec_saved = []
        for model_pred_saved in model_pred_saved_nets:
            KE_speci, KE_spec_true = gen_KE_spectrum(
                N_plot, self.test_data, model_pred_saved, self.grids, self.wet
            )
            KE_spec_saved.append(KE_speci)

        print("Plotting KE Spectrum...")
        plot_metrics_KE_spectrum(
            self.pred_names + [self.network],
            self.region,
            self.str_save,
            self.output_dir,
            KE_spec_true,
            KE_spec_saved + [KE_spec_net],
        )

        KE_net, KE_true = compute_KE(
            N_plot, self.test_data, model_pred_net, self.area, self.wet_bool
        )

        KE_saved = []
        for model_pred_saved in model_pred_saved_nets:
            KE_neti, KE_true = compute_KE(
                N_plot, self.test_data, model_pred_saved, self.area, self.wet_bool
            )
            KE_saved.append(KE_neti)

        print("Plotting KE...")
        plot_metrics_KE(
            self.pred_names + [self.network],
            self.region,
            self.str_save,
            self.output_dir,
            KE_true,
            KE_saved + [KE_net],
        )

        # Enstrophy
        print("Getting Enstrophy stats...")
        enst_spec_net, enst_spec_true = gen_enstrophy_spectrum(
            N_plot,
            self.test_data,
            model_pred_net,
            self.grids,
            self.wet,
            self.wet_lap,
        )

        enst_spec_saved = []
        for model_pred_saved in model_pred_saved_nets:
            enst_speci, enst_spec_true = gen_enstrophy_spectrum(
                N_plot,
                self.test_data,
                model_pred_saved,
                self.grids,
                self.wet,
                self.wet_lap,
            )
            enst_spec_saved.append(enst_speci)

        print("Plotting Enstrophy spectrum...")
        plot_metrics_enstrophy_spectrum(
            self.pred_names + [self.network],
            self.region,
            self.str_save,
            self.output_dir,
            enst_spec_true,
            enst_spec_saved + [enst_spec_net],
        )

        enst_net, enst_true = gen_enstrophy(
            N_plot,
            self.test_data,
            model_pred_net,
            self.dx,
            self.dy,
            self.Nb,
            self.wet_lap,
        )
        enst_net = enst_net.mean(axis=(1, 2))

        enst_saved = []
        for model_pred_saved in model_pred_saved_nets:
            enst_i, enst_true = gen_enstrophy(
                N_plot,
                self.test_data,
                model_pred_saved,
                self.dx,
                self.dy,
                self.Nb,
                self.wet_lap,
            )
            enst_i = enst_i.mean(axis=(1, 2))
            enst_saved.append(enst_i)

        enst_true = enst_true.mean(axis=(1, 2))

        print("Plotting Enstrophy...")
        plot_metrics_entrophy(
            self.pred_names + [self.network],
            self.region,
            self.str_save,
            self.output_dir,
            enst_true,
            enst_saved + [enst_net],
        )

        ### Spatial matching metrics
        print("Getting Spatial matching stats...")
        u_test = np.array(
            self.test_data[:][1][:, 0] * self.std_out[0] + self.mean_out[0]
        )
        v_test = np.array(
            self.test_data[:][1][:, 1] * self.std_out[1] + self.mean_out[1]
        )
        T_test = np.array(
            self.test_data[:][1][:, 2] * self.std_out[2] + self.mean_out[2]
        )

        # Corr
        print("Getting Corr stats...")
        N_eval = 200
        corr_T_net, corr_T_true = compute_corrs_single(
            N_eval,
            T_test,
            model_pred_net[:, :, :, 2],
            self.area,
            self.wet_bool,
            self.std_out[2],
            self.mean_out[2],
        )
        corr_T_saved = []
        for model_pred_saved in model_pred_saved_nets:
            corr_T_i, corr_T_true = compute_corrs_single(
                N_eval,
                T_test,
                model_pred_saved[:, :, :, 2],
                self.area,
                self.wet_bool,
                self.std_out[2],
                self.mean_out[2],
            )
            corr_T_saved.append(corr_T_i)

        print("Plotting Corr...")
        plot_metrics_corr(
            self.pred_names + [self.network],
            self.region,
            self.str_save,
            self.output_dir,
            corr_T_true,
            corr_T_saved + [corr_T_net],
        )

        # RMSE
        print("Getting RMSE stats...")
        RMSE_T_net, RMSE_T_true = compute_RMSE_single(
            N_eval, T_test, model_pred_net[:, :, :, 2], self.area, self.wet_bool
        )

        RMSE_T_saved = []
        for model_pred_saved in model_pred_saved_nets:
            RMSE_T_i, RMSE_T_true = compute_RMSE_single(
                N_eval, T_test, model_pred_saved[:, :, :, 2], self.area, self.wet_bool
            )
            RMSE_T_saved.append(RMSE_T_i)

        print("Plotting RMSE...")
        plot_metrics_rmse(
            self.pred_names + [self.network],
            self.region,
            self.str_save,
            self.output_dir,
            RMSE_T_true,
            RMSE_T_saved + [RMSE_T_net],
        )

        # ACC
        print("Getting ACC stats...")
        N_eval = 100
        ACC_T_net, ACC_T_true = compute_ACC_single(
            N_eval,
            T_test,
            model_pred_net[:, :, :, 2],
            self.clim[:, :, :, 2],
            self.time_test,
            self.area,
            self.wet_bool,
        )

        ACC_T_saved = []
        for model_pred_saved in model_pred_saved_nets:
            ACC_T_i, ACC_T_true = compute_ACC_single(
                N_eval,
                T_test,
                model_pred_saved[:, :, :, 2],
                self.clim[:, :, :, 2],
                self.time_test,
                self.area,
                self.wet_bool,
            )
            ACC_T_saved.append(ACC_T_i)

        print("Plotting ACC...")
        plot_metrics_acc(
            self.pred_names + [self.network],
            self.region,
            self.str_save,
            self.output_dir,
            ACC_T_true,
            ACC_T_saved + [ACC_T_net],
        )

        # PDF
        print("Getting PDF stats...")
        N_days = 100
        day_start = 100  # Last 100 days
        pdf = {}
        for ind_plot in range(3):
            true_field = (
                self.test_data[day_start : day_start + N_days][1][
                    :, ind_plot, self.wet_bool
                ].flatten()
                * self.std_out[ind_plot]
            ) + self.mean_out[ind_plot]
            true_pdf, bins_true = np.histogram(true_field, bins=150, density=True)
            bins_true = (bins_true[1:] + bins_true[:-1]) / 2

            field_net = model_pred_net[
                day_start : day_start + N_days, self.wet_bool, ind_plot
            ].flatten()
            pdf_net, bins_net = np.histogram(field_net, bins=150, density=True)
            bins_net = (bins_net[1:] + bins_net[:-1]) / 2

            pdf[ind_plot] = {
                "true_pdf": true_pdf,
                "true": [bins_true, true_pdf],
                self.network: [bins_net, pdf_net],
            }

            for i, model_pred_saved in enumerate(model_pred_saved_nets):
                field_i = model_pred_saved[
                    day_start : day_start + N_days, self.wet_bool, ind_plot
                ].flatten()
                pdf_i, bins_i = np.histogram(field_i, bins=150, density=True)
                bins_i = (bins_i[1:] + bins_i[:-1]) / 2

                pdf[ind_plot][self.pred_names[i]] = [bins_i, pdf_i]

        print("Plotting pdf...")
        plot_metrics_pdf(
            self.pred_names + [self.network],
            self.region,
            self.output_dir,
            pdf,
        )

    def plot_animation(self):
        def compute_rmse_snapshot(test_data, model_pred, area, wet, mean, std, index):
            area_flat = np.array(area[wet].flatten())

            truth = test_data[index, wet] * std[index] + mean[index]

            truth = np.array(truth.cpu())

            rmse_u = np.sqrt(
                (
                    area_flat
                    * (model_pred[wet, index].flatten() - truth.flatten()) ** 2
                ).sum()
                / area_flat.sum()
            )

            return rmse_u

        def get_stats(
            zarr_path,
            region,
            str_in,
            str_ext,
            test_data,
            area,
            wet_bool,
            N_mean,
            index,
        ):
            mean_out = test_data.norm_vals["m_out"]
            std_out = test_data.norm_vals["s_out"]
            rmse = 1000
            test_time = 25
            for rand_int in range(1, 4):
                model_pred_temp = (
                    xr.open_zarr(
                        zarr_path
                        / (
                            "Pred_lateral_Fast_Data_025_"
                            + region
                            + "_in_"
                            + str_in
                            + "ext_"
                            + str_ext
                            + "N_samples_"
                            + str(4000)
                            + "_rand_seed_"
                            + str(rand_int)
                            + ".zarr"
                        )
                    )
                    .sel(time=slice(test_time - 1, test_time))
                    .to_array()
                    .to_numpy()
                    .squeeze()
                )
                rmse_temp = compute_rmse_snapshot(
                    test_data[test_time - 1][1],
                    model_pred_temp,
                    area,
                    wet_bool,
                    mean_out,
                    std_out,
                    index,
                )
                if rmse_temp < rmse:
                    rmse = rmse_temp
                    rand_best = rand_int
                    print("RMSE: ", rmse)
            model_pred_atm = (
                xr.open_zarr(
                    zarr_path
                    / (
                        "Pred_lateral_Fast_Data_025_"
                        + region
                        + "_in_"
                        + str_in
                        + "ext_"
                        + str_ext
                        + "N_samples_"
                        + str(4000)
                        + "_rand_seed_"
                        + str(rand_best)
                        + ".zarr"
                    )
                )
                .sel(time=slice(0, N_mean))
                .to_array()
                .to_numpy()
                .squeeze()
            )
            return model_pred_atm

        print("Plot animation begin...")

        N_plot = 1000

        for ind_plot in range(3):

            model_pred_net = get_stats(
                self.pred_model_path,
                self.region,
                self.str_in,
                self.str_ext,
                self.test_data,
                self.area,
                self.wet_bool,
                N_plot,
                ind_plot,
            )

            model_pred_saved_nets = []
            for model_pred_path in self.pred_paths:
                model_pred_i = get_stats(
                    Path(model_pred_path),
                    self.pred_region,
                    self.str_in,
                    self.str_ext,
                    self.test_data,
                    self.area,
                    self.wet_bool,
                    N_plot,
                    ind_plot,
                )
                model_pred_saved_nets.append(model_pred_i)

                # torch.save(model_pred_unet, f'model_pred_{ind_plot}.pt')

            model_pred_saved_nets.append(model_pred_net)
            var_list = {"1": r"v", "0": r"u", "2": r"T"}
            fig, plts, a = get_initial_snapshot_fig(
                self.pred_names + [self.network],
                N_plot,
                self.region,
                self.grids,
                self.test_data,
                self.wet_nan,
                model_pred_saved_nets,
                self.mean_out,
                self.std_out,
                ind_plot,
                self.Nb,
            )
            plt.savefig(Path(self.output_dir) / "initial_snapshot.png")

            def update_snapshot(i):
                if 'global' in self.region:
                    plts[0].set_array(
                        (
                            self.test_data[i][1][ind_plot].cpu()
                            * self.wet_nan
                            * self.std_out[ind_plot]
                            + self.mean_out[ind_plot]
                        ).flatten()
                    )
                    for j, model_pred in enumerate(model_pred_saved_nets):
                        plts[j + 1].set_array(
                            (
                                model_pred[
                                    i, :, :, ind_plot
                                ]
                                * self.wet_nan
                            ).flatten()
                        )
                else:
                    plts[0].set_array(
                        (
                            self.test_data[i][1][
                                ind_plot, self.Nb : -self.Nb, self.Nb : -self.Nb
                            ].cpu()
                            * self.wet_nan[self.Nb : -self.Nb, self.Nb : -self.Nb]
                            * self.std_out[ind_plot]
                            + self.mean_out[ind_plot]
                        ).flatten()
                    )
                    for j, model_pred in enumerate(model_pred_saved_nets):
                        plts[j + 1].set_array(
                            (
                                model_pred[
                                    i, self.Nb : -self.Nb, self.Nb : -self.Nb, ind_plot
                                ]
                                * self.wet_nan[self.Nb : -self.Nb, self.Nb : -self.Nb]
                            ).flatten()
                        )
                        

                a.set_text(r"$t = " + str(i + 1) + "$ days ")

            anim = FuncAnimation(
                fig, update_snapshot, interval=100, frames=range(0, 1000, 2)
            )
            anim.save(
                Path(self.output_dir)
                / (
                    self.post_model_name
                    + "_"
                    + self.region
                    + "_"
                    + var_list[str(ind_plot)]
                    + ".gif"
                )
            )

    def plot_long_metrics(self):
        print("Plot Long metrics begin...")
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
                + str(4000)
                + "_rand_seed_"
                + str(1)
                + ".zarr"
            )

            model_pred_saved_nets.append(
                xr.open_zarr(net_path).to_array().to_numpy().squeeze()
            )

        ### Long time KE
        print("Getting Long mean KE stats...")
        start = 1999
        N_plot = 2999

        long_KE_net, long_KE_true = gen_KE_range(
            start, N_plot, self.test_data, model_pred_net
        )
        long_KE_net = long_KE_net.mean(0)
        long_KE_true = long_KE_true.mean(0)

        long_KE_saved = []
        for model_pred_saved in model_pred_saved_nets:
            long_KE_savedi, long_KE_true = gen_KE_range(
                start, N_plot, self.test_data, model_pred_saved
            )
            long_KE_savedi = long_KE_savedi.mean(0)
            long_KE_true = long_KE_true.mean(0)
            long_KE_saved.append(long_KE_savedi)

        print("Plotting Long mean KE...")
        plot_long_KE(
            self.pred_names + [self.network],
            self.region + "_Long_",
            self.str_save,
            self.output_dir,
            self.grids,
            self.Nb,
            self.wet_nan,
            long_KE_true,
            long_KE_saved + [long_KE_net],
        )

        ### Short time scale metrics
        N_plot = 1000

        # KE
        print("Getting Long KE stats...")
        KE_spec_net, KE_spec_true = gen_KE_spectrum(
            N_plot, self.test_data, model_pred_net, self.grids, self.wet
        )

        KE_spec_saved = []
        for model_pred_saved in model_pred_saved_nets:
            KE_speci, KE_spec_true = gen_KE_spectrum(
                N_plot, self.test_data, model_pred_saved, self.grids, self.wet
            )
            KE_spec_saved.append(KE_speci)

        print("Plotting Long KE Spectrum...")
        plot_metrics_KE_spectrum(
            self.pred_names + [self.network],
            self.region + "_Long_",
            self.str_save,
            self.output_dir,
            KE_spec_true,
            KE_spec_saved + [KE_spec_net],
        )

        KE_net, KE_true = compute_KE(
            N_plot, self.test_data, model_pred_net, self.area, self.wet_bool
        )

        KE_saved = []
        for model_pred_saved in model_pred_saved_nets:
            KE_neti, KE_true = compute_KE(
                N_plot, self.test_data, model_pred_saved, self.area, self.wet_bool
            )
            KE_saved.append(KE_neti)

        print("Plotting Long KE...")
        plot_metrics_KE(
            self.pred_names + [self.network],
            self.region + "_Long_",
            self.str_save,
            self.output_dir,
            KE_true,
            KE_saved + [KE_net],
        )

        # Enstrophy
        print("Getting Long Enstrophy stats...")
        enst_spec_net, enst_spec_true = gen_enstrophy_spectrum(
            N_plot,
            self.test_data,
            model_pred_net,
            self.grids,
            self.wet,
            self.wet_lap,
        )

        enst_spec_saved = []
        for model_pred_saved in model_pred_saved_nets:
            enst_speci, enst_spec_true = gen_enstrophy_spectrum(
                N_plot,
                self.test_data,
                model_pred_saved,
                self.grids,
                self.wet,
                self.wet_lap,
            )
            enst_spec_saved.append(enst_speci)

        print("Plotting Long Enstrophy spectrum...")
        plot_metrics_enstrophy_spectrum(
            self.pred_names + [self.network],
            self.region + "_Long_",
            self.str_save,
            self.output_dir,
            enst_spec_true,
            enst_spec_saved + [enst_spec_net],
        )

        enst_net, enst_true = gen_enstrophy(
            N_plot,
            self.test_data,
            model_pred_net,
            self.dx,
            self.dy,
            self.Nb,
            self.wet_lap,
        )
        enst_net = enst_net.mean(axis=(1, 2))

        enst_saved = []
        for model_pred_saved in model_pred_saved_nets:
            enst_i, enst_true = gen_enstrophy(
                N_plot,
                self.test_data,
                model_pred_saved,
                self.dx,
                self.dy,
                self.Nb,
                self.wet_lap,
            )
            enst_i = enst_i.mean(axis=(1, 2))
            enst_saved.append(enst_i)

        enst_true = enst_true.mean(axis=(1, 2))

        print("Plotting Long Enstrophy...")
        plot_metrics_entrophy(
            self.pred_names + [self.network],
            self.region + "_Long_",
            self.str_save,
            self.output_dir,
            enst_true,
            enst_saved + [enst_net],
        )

        ### Spatial matching metrics
        print("Getting Spatial matching stats...")
        u_test = np.array(
            self.test_data[:][1][:, 0] * self.std_out[0] + self.mean_out[0]
        )
        v_test = np.array(
            self.test_data[:][1][:, 1] * self.std_out[1] + self.mean_out[1]
        )
        T_test = np.array(
            self.test_data[:][1][:, 2] * self.std_out[2] + self.mean_out[2]
        )

        # # Corr
        # print("Getting Long Corr stats...")
        # N_eval = 1000
        # corr_T_net, corr_T_true = compute_corrs_single(
        #     N_eval,
        #     T_test,
        #     model_pred_net[:, :, :, 2],
        #     self.area,
        #     self.wet_bool,
        #     self.std_out[2],
        #     self.mean_out[2],
        # )
        # corr_T_saved = []
        # for model_pred_saved in model_pred_saved_nets:
        #     corr_T_i, corr_T_true = compute_corrs_single(
        #         N_eval,
        #         T_test,
        #         model_pred_saved[:, :, :, 2],
        #         self.area,
        #         self.wet_bool,
        #         self.std_out[2],
        #         self.mean_out[2],
        #     )
        #     corr_T_saved.append(corr_T_i)

        # print("Plotting Long Corr...")
        # plot_metrics_corr(
        #     self.pred_names + [self.network],
        #     self.region + "_Long_",
        #     self.str_save,
        #     self.output_dir,
        #     corr_T_true,
        #     corr_T_saved + [corr_T_net],
        # )

        # # RMSE
        # print("Getting Long RMSE stats...")
        # RMSE_T_net, RMSE_T_true = compute_RMSE_single(
        #     N_eval, T_test, model_pred_net[:, :, :, 2], self.area, self.wet_bool
        # )

        # RMSE_T_saved = []
        # for model_pred_saved in model_pred_saved_nets:
        #     RMSE_T_i, RMSE_T_true = compute_RMSE_single(
        #         N_eval, T_test, model_pred_saved[:, :, :, 2], self.area, self.wet_bool
        #     )
        #     RMSE_T_saved.append(RMSE_T_i)

        # print("Plotting Long RMSE...")
        # plot_metrics_rmse(
        #     self.pred_names + [self.network],
        #     self.region + "_Long_",
        #     self.str_save,
        #     self.output_dir,
        #     RMSE_T_true,
        #     RMSE_T_saved + [RMSE_T_net],
        # )

        # # ACC
        # print("Getting Long ACC stats...")
        # N_eval = 1000
        # ACC_T_net, ACC_T_true = compute_ACC_single(
        #     N_eval,
        #     T_test,
        #     model_pred_net[:, :, :, 2],
        #     self.clim[:, :, :, 2],
        #     self.time_test,
        #     self.area,
        #     self.wet_bool,
        # )

        # ACC_T_saved = []
        # for model_pred_saved in model_pred_saved_nets:
        #     ACC_T_i, ACC_T_true = compute_ACC_single(
        #         N_eval,
        #         T_test,
        #         model_pred_saved[:, :, :, 2],
        #         self.clim[:, :, :, 2],
        #         self.time_test,
        #         self.area,
        #         self.wet_bool,
        #     )
        #     ACC_T_saved.append(ACC_T_i)

        # print("Plotting Long ACC...")
        # plot_metrics_acc(
        #     self.pred_names + [self.network],
        #     self.region + "_Long_",
        #     self.str_save,
        #     self.output_dir,
        #     ACC_T_true,
        #     ACC_T_saved + [ACC_T_net],
        # )

        # PDF
        print("Getting Long PDF stats...")
        N_days = 1000
        day_start = 1999  # Last 100 days
        pdf = {}
        for ind_plot in range(3):
            true_field = (
                self.test_data[day_start : day_start + N_days][1][
                    :, ind_plot, self.wet_bool
                ].flatten()
                * self.std_out[ind_plot]
            ) + self.mean_out[ind_plot]
            true_pdf, bins_true = np.histogram(true_field, bins=150, density=True)
            bins_true = (bins_true[1:] + bins_true[:-1]) / 2

            field_net = model_pred_net[
                day_start : day_start + N_days, self.wet_bool, ind_plot
            ].flatten()
            pdf_net, bins_net = np.histogram(field_net, bins=150, density=True)
            bins_net = (bins_net[1:] + bins_net[:-1]) / 2

            pdf[ind_plot] = {
                "true_pdf": true_pdf,
                "true": [bins_true, true_pdf],
                self.network: [bins_net, pdf_net],
            }

            for i, model_pred_saved in enumerate(model_pred_saved_nets):
                field_i = model_pred_saved[
                    day_start : day_start + N_days, self.wet_bool, ind_plot
                ].flatten()
                pdf_i, bins_i = np.histogram(field_i, bins=150, density=True)
                bins_i = (bins_i[1:] + bins_i[:-1]) / 2

                pdf[ind_plot][self.pred_names[i]] = [bins_i, pdf_i]

        print("Plotting Long pdf...")
        plot_metrics_pdf(
            self.pred_names + [self.network],
            self.region + "_Long_",
            self.output_dir,
            pdf,
        )

    def send_data_to_cpu(self):
        self.test_data.set_device(device="cpu")


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

    if args.run_long_metrics:
        e.plot_long_metrics()

    if args.run_plot_animation:
        e.plot_animation()
