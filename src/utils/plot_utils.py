import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cmocean
from pathlib import Path
import numpy as np
import cartopy.crs as ccrs
import cartopy as cart
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def plot_time_spec(
    network_names,
    axs,
    plt_index,
    index,
    N_test,
    freqs,
    auto_FFT,
    FFTs,
    clist,
    legend=True,
):
    T_plot = 200

    N_int = int(T_plot)
    N_true = min(N_test, N_int)

    var_list = {
        "1": r"$\widehat{\overline{v}}$ $( m/s )$",
        "0": r"$\widehat{\overline{u}}$ $( m/s )$",
        "2": r"$\widehat{\overline{T}}$ $( ^\circ C )$",
    }

    axs[plt_index].semilogx(
        freqs[:N_int], auto_FFT[:N_int, index], "--k", label="CM2.6", zorder=5
    )

    for i, FFT_i in enumerate(FFTs):
        if FFT_i is not None:
            axs[plt_index].plot(
                freqs[:N_int],
                FFT_i.mean(axis=0)[:N_int, index],
                color=clist[i],
                label=network_names[i],
            )
            axs[plt_index].fill_between(
                freqs[:N_int],
                FFT_i.mean(axis=0)[:N_int, index] - FFT_i.std(axis=0)[:N_int, index],
                FFT_i.mean(axis=0)[:N_int, index] + FFT_i.std(axis=0)[:N_int, index],
                ls="--",
                color=clist[i],
                alpha=0.25,
            )

    axs[plt_index].set_ylabel(r"" + var_list[str(index)])
    axs[plt_index].set_xlabel(r"Frequency $( 1/day )$")

    axs[plt_index].set_xlim([0, freqs[T_plot]])
    axs[plt_index].set_ylim([0, auto_FFT[1:N_int, index].max() * 2])

    if legend:
        axs[plt_index].legend(ncol=1, loc="upper right")


def plot_var(
    network_names,
    axs,
    plt_index,
    index,
    N_test,
    lag,
    auto_var,
    vars,
    clist,
):
    T_plot = 1098

    N_int = int(T_plot / lag)
    N_true = min(N_test, N_int)

    var_list = {
        "1": r"$\mathrm{Var}(\overline{v})$",
        "0": r"$\mathrm{Var}(\overline{u})$",
        "2": r"$\mathrm{Var}(\overline{T})$",
    }

    axs[plt_index].plot(
        (np.arange(N_int) * lag) / 366,
        auto_var[:N_int, index],
        "--k",
        label="CM2.6",
        zorder=5,
    )

    for i, vars_i in enumerate(vars):
        if vars_i is not None:
            axs[plt_index].plot(
                (np.arange(N_int) * lag) / 366,
                vars_i.mean(axis=0)[:N_int, index],
                color=clist[i],
                label=network_names[i],
            )
            axs[plt_index].fill_between(
                (np.arange(N_int) * lag) / 366,
                vars_i.mean(axis=0)[:N_int, index] - vars_i.std(axis=0)[:N_int, index],
                vars_i.mean(axis=0)[:N_int, index] + vars_i.std(axis=0)[:N_int, index],
                ls="--",
                color=clist[i],
                alpha=0.25,
            )

    axs[plt_index].set_ylabel(r"" + var_list[str(index)])
    axs[plt_index].set_xlabel(r"Time $( years )$")

    axs[plt_index].set_xlim([0, T_plot / 366])
    axs[plt_index].yaxis.set_major_formatter(
        ticker.ScalarFormatter(useMathText=True, useOffset=False)
    )
    axs[plt_index].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[plt_index].xaxis.set_major_locator(
        ticker.MultipleLocator(base=0.5)
    )  # Adjust base as needed


def plot_mean(
    network_names,
    axs,
    plt_index,
    index,
    N_test,
    lag,
    auto_mean,
    means,
    clist,
):

    T_plot = N_test

    N_int = int(T_plot / lag)
    N_true = min(N_test, N_int)

    var_list = {
        "1": r"$\overline{v}$ $( m/s )$",
        "0": r"$\overline{u}$ $( m/s )$",
        "2": r"$\overline{T}$ $( ^\circ C )$",
    }

    axs[plt_index].plot(
        (np.arange(N_int) * lag) / 366,
        auto_mean[:N_int, index],
        "--k",
        label="CM2.6",
        zorder=5,
    )

    for i, means_i in enumerate(means):
        if means_i is not None:
            axs[plt_index].plot(
                (np.arange(N_int) * lag) / 366,
                means_i.mean(axis=0)[:N_int, index],
                color=clist[i],
                label=network_names[i],
            )
            axs[plt_index].fill_between(
                (np.arange(N_int) * lag) / 366,
                means_i.mean(axis=0)[:N_int, index]
                - means_i.std(axis=0)[:N_int, index],
                means_i.mean(axis=0)[:N_int, index]
                + means_i.std(axis=0)[:N_int, index],
                ls="--",
                color=clist[i],
                alpha=0.25,
            )

    axs[plt_index].set_ylabel(r"" + var_list[str(index)])
    axs[plt_index].set_xlabel(r"Time $( years )$")

    min_val = auto_mean[:N_int, index].min()
    max_val = auto_mean[:N_int, index].max()

    if min_val > 0:
        axs[plt_index].set_ylim([min_val * 0.8, max_val * 1.1])
    elif min_val < 0 and max_val > 0:
        axs[plt_index].set_ylim([min_val * 1.1, max_val * 1.1])
    else:
        axs[plt_index].set_ylim([min_val * 1.1, 0])

    if index == 2:
        axs[plt_index].set_xlim([4, 8])
        axs[plt_index].xaxis.set_major_locator(
            ticker.MultipleLocator(base=1)
        )  # Adjust base as needed

    #     axs[plt_index].set_ylim([22,28])
    else:
        axs[plt_index].set_xlim([7, 8])
        axs[plt_index].yaxis.set_major_formatter(
            ticker.ScalarFormatter(useMathText=True, useOffset=False)
        )
        axs[plt_index].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axs[plt_index].xaxis.set_major_locator(
            ticker.MultipleLocator(base=0.5)
        )  # Adjust base as needed


def plot_acc(
    network_names,
    axs,
    plt_ind_acc,
    index,
    N_test,
    lag,
    auto_ACC,
    ACCs,
    clist,
):
    T_plot = 100

    N_int = int(T_plot / lag)
    N_true = min(N_test, N_int)

    var_list = {
        "1": r"$\overline{v}$ $( m/s )$",
        "0": r"$\overline{u}$ $( m/s )$",
        "2": r"$\overline{T}$ $( ^\circ C )$",
    }

    axs[plt_ind_acc].plot(
        (np.arange(N_int) * lag),
        auto_ACC.mean(axis=0)[:N_int, index],
        color="dimgrey",
        label="$\mathbf{\Phi}(t=0)$",
    )
    # axs[plt_ind_acc].fill_between(
    #     (np.arange(N_int) * lag),
    #     auto_ACC.mean(axis=0)[:N_int, index] - auto_ACC.std(axis=0)[:N_int, index],
    #     auto_ACC.mean(axis=0)[:N_int, index] + auto_ACC.std(axis=0)[:N_int, index],
    #     ls="-",
    #     color="dimgrey",
    #     alpha=0.2,
    # )

    for i, ACC_i in enumerate(ACCs):
        if ACC_i is not None:
            axs[plt_ind_acc].plot(
                (np.arange(N_int) * lag),
                ACC_i.mean(axis=0)[:N_int, index],
                color=clist[i],
                label=network_names[i],
            )
            # axs[plt_ind_acc].fill_between(
            #     (np.arange(N_int) * lag),
            #     ACC_i.mean(axis=0)[:N_int, index] - ACC_i.std(axis=0)[:N_int, index],
            #     ACC_i.mean(axis=0)[:N_int, index] + ACC_i.std(axis=0)[:N_int, index],
            #     ls="-",
            #     color=clist[i],
            #     alpha=0.2,
            # )

    axs[plt_ind_acc].set_ylabel(r"ACC " + var_list[str(index)])

    axs[plt_ind_acc].set_xlabel("Time (days)")

    axs[plt_ind_acc].set_ylim([0, 1])
    axs[plt_ind_acc].set_xlim([0, T_plot])


def plot_corr(
    network_names,
    axs,
    plt_ind_acc,
    index,
    N_test,
    lag,
    auto_corrs,
    corrs,
    clist,
):

    T_plot = 100

    N_int = int(T_plot / lag)
    N_true = min(N_test, N_int)

    var_list = {
        "1": r"$\overline{v}$ $( m/s )$",
        "0": r"$\overline{u}$ $( m/s )$",
        "2": r"$\overline{T}$ $( ^\circ C )$",
    }

    axs[plt_ind_acc].plot(
        (np.arange(N_int) * lag),
        auto_corrs.mean(axis=0)[:N_int, index],
        color="dimgrey",
        label="$\mathbf{\Phi}(t=0)$",
    )
    axs[plt_ind_acc].fill_between(
        (np.arange(N_int) * lag),
        auto_corrs.mean(axis=0)[:N_int, index] - auto_corrs.std(axis=0)[:N_int, index],
        auto_corrs.mean(axis=0)[:N_int, index] + auto_corrs.std(axis=0)[:N_int, index],
        ls="-",
        color="dimgrey",
        alpha=0.2,
    )

    for i, corrs_i in enumerate(corrs):
        if corrs_i is not None:
            axs[plt_ind_acc].plot(
                (np.arange(N_int) * lag),
                corrs_i.mean(axis=0)[:N_int, index],
                color=clist[i],
                label=network_names[i],
            )
            axs[plt_ind_acc].fill_between(
                (np.arange(N_int) * lag),
                corrs_i.mean(axis=0)[:N_int, index]
                - corrs_i.std(axis=0)[:N_int, index],
                corrs_i.mean(axis=0)[:N_int, index]
                + corrs_i.std(axis=0)[:N_int, index],
                ls="-",
                color=clist[i],
                alpha=0.2,
            )

    axs[plt_ind_acc].set_ylabel(r"Correlation " + var_list[str(index)])
    axs[plt_ind_acc].set_xlabel(r"Time $( days )$")

    axs[plt_ind_acc].set_ylim([0, 1])
    axs[plt_ind_acc].set_xlim([0, T_plot])


def plot_KE(
    network_names,
    axs,
    plt_ind_acc,
    N_test,
    lag,
    auto_KE,
    KEs,
    clist,
):

    T_plot = 200

    N_int = int(T_plot / lag)
    N_true = min(N_test, N_int)

    var_list = {
        "1": r"$\overline{v}$ $( m/s )$",
        "0": r"$\overline{u}$ $( m/s )$",
        "2": r"$\overline{T}$ $( ^\circ C )$",
    }

    axs[plt_ind_acc].plot(
        (np.arange(N_int) * lag) / 366,
        auto_KE[:N_int].mean(axis=0),
        color="dimgrey",
        label="$\mathbf{\Phi}(t=0)$",
    )
    axs[plt_ind_acc].fill_between(
        (np.arange(N_int) * lag) / 366,
        auto_KE.mean(axis=0)[:N_int] - auto_KE.std(axis=0)[:N_int],
        auto_KE.mean(axis=0)[:N_int] + auto_KE.std(axis=0)[:N_int],
        ls="-",
        color="dimgrey",
        alpha=0.2,
    )

    for i, KE_i in enumerate(KEs):
        if KE_i is not None:
            axs[plt_ind_acc].plot(
                (np.arange(N_int) * lag) / 366,
                KE_i.mean(axis=0)[:N_int],
                color=clist[i],
                label=network_names[i],
            )
            axs[plt_ind_acc].fill_between(
                (np.arange(N_int) * lag) / 366,
                KE_i.mean(axis=0)[:N_int] - KE_i.std(axis=0)[:N_int],
                KE_i.mean(axis=0)[:N_int] + KE_i.std(axis=0)[:N_int],
                ls="-",
                color=clist[i],
                alpha=0.2,
            )

    axs[plt_ind_acc].set_ylabel(r"KE")
    axs[plt_ind_acc].set_xlabel(r"Time $( days )$")

    axs[plt_ind_acc].set_ylim([0, 0.05])


def plot_rmse(
    network_names,
    axs,
    plt_ind_acc,
    index,
    N_test,
    lag,
    auto_rmse,
    rmses,
    clist,
    legend=False,
):
    T_plot = 200

    N_int = int(T_plot / lag)
    N_true = min(N_test, N_int)

    var_list = {
        "1": r"$\overline{v}$ $( m/s )$",
        "0": r"$\overline{u}$ $( m/s )$",
        "2": r"$\overline{T}$ $( ^\circ C )$",
    }

    axs[plt_ind_acc].plot(
        (np.arange(N_int) * lag),
        auto_rmse.mean(axis=0)[:N_int, index],
        color="dimgrey",
        label="$\mathbf{\Phi}(t=0)$",
    )
    # axs[plt_ind_acc].fill_between(
    #     (np.arange(N_int) * lag),
    #     auto_rmse.mean(axis=0)[:N_int, index] - auto_rmse.std(axis=0)[:N_int, index],
    #     auto_rmse.mean(axis=0)[:N_int, index] + auto_rmse.std(axis=0)[:N_int, index],
    #     ls="-",
    #     color="dimgrey",
    #     alpha=0.2,
    # )

    for i, rmse_i in enumerate(rmses):
        if rmse_i is not None:
            axs[plt_ind_acc].plot(
                (np.arange(N_int) * lag),
                rmse_i.mean(axis=0)[:N_int, index],
                color=clist[i],
                label=network_names[i],
            )
            # axs[plt_ind_acc].fill_between(
            #     (np.arange(N_int) * lag),
            #     rmse_i.mean(axis=0)[:N_int, index] - rmse_i.std(axis=0)[:N_int, index],
            #     rmse_i.mean(axis=0)[:N_int, index] + rmse_i.std(axis=0)[:N_int, index],
            #     ls="-",
            #     color=clist[i],
            #     alpha=0.2,
            # )

    axs[plt_ind_acc].set_ylabel(r"RMSE " + var_list[str(index)])
    axs[plt_ind_acc].set_xlabel(r"Time $( days )$")

    axs[plt_ind_acc].set_xlim([0, T_plot])
    if legend:
        axs[plt_ind_acc].legend(ncol=2)
    if index == 2:
        axs[plt_ind_acc].set_ylim([0, 8])

    if index == 1 or index == 0:
        axs[plt_ind_acc].yaxis.set_major_formatter(
            ticker.ScalarFormatter(useMathText=True, useOffset=False)
        )
        axs[plt_ind_acc].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


def plot_long_time_stats(
    network_names,
    region,
    save_str,
    output_dir,
    N_test,
    lag,
    freqs,
    auto_FFT,
    FFTs,
    auto_mean,
    means,
):

    plt.clf()
    plt.style.use("bmh")

    clist = ["#A00B41", "#3300EA", "#00DCDE", "#A6BD00"]

    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc("axes", titlesize=20)  # fontsize of the axes title
    plt.rc("axes", labelsize=18)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=18)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=18)  # fontsize of the tick labels
    plt.rc("legend", fontsize=10)  # legend fontsize
    plt.rc("figure", titlesize=18)

    fig, axs = plt.subplots(
        2,
        2,
        figsize=(11, 6),
        gridspec_kw={
            "width_ratios": [1, 1],
            "height_ratios": [1, 1],
            "wspace": 0.3,
            "hspace": 0.5,
        },
    )
    plot_time_spec(
        network_names,
        axs,
        (0, 0),
        0,
        N_test,
        freqs,
        auto_FFT,
        FFTs,
        clist,
        False,
    )
    plot_mean(
        network_names,
        axs,
        (0, 1),
        0,
        N_test,
        lag,
        auto_mean,
        means,
        clist,
    )
    plot_time_spec(
        network_names,
        axs,
        (1, 0),
        1,
        N_test,
        freqs,
        auto_FFT,
        FFTs,
        clist,
    )
    plot_mean(
        network_names,
        axs,
        (1, 1),
        2,
        N_test,
        lag,
        auto_mean,
        means,
        clist,
    )

    region_title = ""

    for i in region:
        if region == "Quiescent_Ext":
            region_title = "South Pacific"
        elif region == "Africa_Ext":
            region_title = "African Cape"
        elif i == "_":
            region_title += " "
        elif i == "E":
            break
        else:
            region_title += i
    region_title = str(region_title)

    # fig.suptitle("Long-Time Statistics " + region_title, fontsize=16)

    plt.savefig(
        Path(output_dir)
        / ("Long_Time_Comp_Boundary_" + region + "_" + save_str + ".png"),
        bbox_inches="tight",
    )
    plt.clf()


def plot_short_time_stats(
    network_names,
    region,
    save_str,
    output_dir,
    N_test,
    lag,
    auto_ACC,
    ACCs,
    auto_rmse,
    rmses,
    auto_KE,
    KEs,
    auto_corrs,
    corrs,
):

    clist = ["#A00B41", "#3300EA", "#00DCDE", "#A6BD00"]

    def init_plt():
        plt.clf()
        plt.style.use("bmh")

        # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc("axes", titlesize=20)  # fontsize of the axes title
        plt.rc("axes", labelsize=18)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=18)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=18)  # fontsize of the tick labels
        plt.rc("legend", fontsize=10)  # legend fontsize
        plt.rc("figure", titlesize=18)

        fig, axs = plt.subplots(
            2,
            2,
            figsize=(11, 6),
            gridspec_kw={
                "width_ratios": [1, 1],
                "height_ratios": [1, 1],
                "wspace": 0.4,
                "hspace": 0.5,
            },
        )
        return fig, axs

    # fig, axs = init_plt()
    # plot_acc(
    #     network_names,
    #     axs,
    #     (0, 0),
    #     2,
    #     N_test,
    #     lag,
    #     auto_ACC,
    #     ACCs,
    #     clist,
    # )
    # plot_corr(
    #     network_names,
    #     axs,
    #     (0, 1),
    #     1,
    #     N_test,
    #     lag,
    #     auto_corrs,
    #     corrs,
    #     clist,
    # )
    # plot_rmse(network_names, axs, (1, 0), 2, N_test, lag, auto_rmse, rmses, clist, True)
    # plot_KE(network_names, axs, (1, 1), N_test, lag, auto_KE, KEs, clist)

    # fig.suptitle("Short-Time Statistics 1" + region, fontsize=16)

    # plt.savefig(
    #     Path(output_dir)
    #     / ("Short_Time_Comp_Boundary1_" + region + "_" + save_str + ".png"),
    #     bbox_inches="tight",
    # )

    # fig, axs = init_plt()
    # plot_acc(
    #     network_names,
    #     axs,
    #     (0, 0),
    #     0,
    #     N_test,
    #     lag,
    #     auto_ACC,
    #     ACCs,
    #     clist,
    # )
    # plot_acc(
    #     network_names,
    #     axs,
    #     (0, 1),
    #     1,
    #     N_test,
    #     lag,
    #     auto_ACC,
    #     ACCs,
    #     clist,
    # )
    # plot_rmse(network_names, axs, (1, 0), 0, N_test, lag, auto_rmse, rmses, clist, True)
    # plot_rmse(
    #     network_names,
    #     axs,
    #     (1, 1),
    #     1,
    #     N_test,
    #     lag,
    #     auto_rmse,
    #     rmses,
    #     clist,
    # )

    # # fig.suptitle("Short-Time Statistics 2" + region, fontsize=16)

    # plt.savefig(
    #     Path(output_dir)
    #     / ("Short_Time_Comp_Boundary2_" + region + "_" + save_str + ".png"),
    #     bbox_inches="tight",
    # )

    fig, axs = init_plt()
    plot_acc(
        network_names,
        axs,
        (0, 0),
        0,
        N_test,
        lag,
        auto_ACC,
        ACCs,
        clist,
    )
    plot_acc(
        network_names,
        axs,
        (0, 1),
        2,
        N_test,
        lag,
        auto_ACC,
        ACCs,
        clist,
    )
    plot_rmse(network_names, axs, (1, 0), 0, N_test, lag, auto_rmse, rmses, clist, True)
    plot_rmse(
        network_names,
        axs,
        (1, 1),
        2,
        N_test,
        lag,
        auto_rmse,
        rmses,
        clist,
    )

    # fig.suptitle("Short-Time Statistics 2" + region, fontsize=16)

    plt.savefig(
        Path(output_dir)
        / ("Short_Time_Comp_Boundary_" + region + "_" + save_str + ".png"),
        bbox_inches="tight",
    )

    plt.clf()


def plot_metrics_KE_spectrum(
    network_names, region, save_str, output_dir, KE_spec_true, KE_specs
):

    plt.style.use("bmh")

    clist = ["#A00B41", "#3300EA", "#00DCDE", "#A6BD00"]

    # KE Spectrum
    for i, KE_spec_i in enumerate(KE_specs):
        if KE_spec_i is not None:
            plt.loglog(
                KE_spec_i.freq_r,
                KE_spec_i,
                c=clist[i],
                label=f"{network_names[i]}",
            )

    plt.loglog(KE_spec_true.freq_r, KE_spec_true, "--k", label="CM2.6")

    plt.xlabel(r"Wave number $( 1/km )$")
    plt.ylabel(r"KE $( J/m^2 )$")

    plt.legend(loc="lower left")
    plt.savefig(
        Path(output_dir) / ("KE_spectrum" + region + "_" + save_str + ".png"),
        bbox_inches="tight",
    )
    plt.clf()


def plot_metrics_KE(
    network_names,
    region,
    save_str,
    output_dir,
    KE_true,
    KEs,
):
    plt.style.use("bmh")

    clist = ["#A00B41", "#3300EA", "#00DCDE", "#A6BD00"]

    N_plot = len(KE_true)

    # KE
    rho = 1020
    for i, KE_i in enumerate(KEs):
        if KE_i is not None:
            plt.plot(
                np.arange(1, N_plot + 1),
                KE_i * rho,
                c=clist[i],
                label=f"{network_names[i]}",
            )

    plt.plot(np.arange(1, N_plot + 1), KE_true * rho, "--k", label="CM2.6")
    plt.xlabel(r"time $( days )$")
    plt.ylabel(r"KE $( J/m^2 )$")
    plt.legend(loc="lower left")
    plt.savefig(
        Path(output_dir) / ("KE" + region + "_" + save_str + ".png"),
        bbox_inches="tight",
    )
    plt.clf()


def plot_metrics_enstrophy_spectrum(
    network_names, region, save_str, output_dir, enst_spec_true, enst_specs
):
    plt.style.use("bmh")

    clist = ["#A00B41", "#3300EA", "#00DCDE", "#A6BD00"]

    # Enstrophy Spectrum
    for i, enst_spec_i in enumerate(enst_specs):
        if enst_spec_i is not None:
            plt.loglog(
                enst_spec_i.freq_r,
                enst_spec_i,
                c=clist[i],
                label=f"{network_names[i]}",
            )

    plt.loglog(enst_spec_true.freq_r, enst_spec_true, "--k", label="CM2.6")
    plt.xlabel(r"Wave number $( 1/km )$")
    plt.ylabel(r"Enstrophy $( m^2/s^2 )$")
    plt.legend(loc="lower left")
    plt.savefig(
        Path(output_dir) / ("Enstrophy_Spectrum" + region + "_" + save_str + ".png"),
        bbox_inches="tight",
    )
    plt.clf()


def plot_metrics_entrophy(
    network_names,
    region,
    save_str,
    output_dir,
    enst_true,
    ensts,
):
    plt.style.use("bmh")

    clist = ["#A00B41", "#3300EA", "#00DCDE", "#A6BD00"]

    N_plot = len(enst_true)

    # Enstrophy
    for i, enst_i in enumerate(ensts):
        if enst_i is not None:
            plt.plot(
                np.arange(1, N_plot + 1),
                enst_i,
                c=clist[i],
                label=f"{network_names[i]}",
            )

    plt.plot(np.arange(1, N_plot + 1), enst_true, "--k", label="CM2.6")
    plt.xlabel(r"time $( days )$")
    plt.ylabel(r"Enstrophy $( m^2/s^2 )$")
    plt.legend(loc="lower left")
    plt.savefig(
        Path(output_dir) / ("Enstrophy" + region + "_" + save_str + ".png"),
        bbox_inches="tight",
    )
    plt.clf()


def plot_metrics_corr(
    network_names,
    region,
    save_str,
    output_dir,
    corr_T_true,
    corr_Ts,
):
    plt.style.use("bmh")

    clist = ["#A00B41", "#3300EA", "#00DCDE", "#A6BD00"]

    # Corr
    N_eval = len(corr_T_true)
    for i, corr_Ti in enumerate(corr_Ts):
        if corr_Ti is not None:
            plt.plot(
                np.arange(1, N_eval + 1),
                corr_Ti,
                c=clist[i],
                label=f"{network_names[i]}",
            )

    plt.plot(np.arange(1, N_eval + 1), corr_T_true, "--k", label="CM2.6")
    plt.xlabel(r"time $( days )$")
    plt.ylabel(r"Correlation $\overline{T}$")
    plt.ylim([0, 1])
    plt.xlim([0, N_eval])

    plt.legend(loc="lower left")
    plt.savefig(
        Path(output_dir) / ("Corr" + region + "_" + save_str + ".png"),
        bbox_inches="tight",
    )
    plt.clf()


def plot_metrics_rmse(
    network_names, region, save_str, output_dir, RMSE_T_true, RMSE_Ts
):
    plt.style.use("bmh")

    clist = ["#A00B41", "#3300EA", "#00DCDE", "#A6BD00"]

    # RMSE
    N_eval = len(RMSE_T_true)
    for i, RMSE_Ti in enumerate(RMSE_Ts):
        if RMSE_Ti is not None:
            plt.plot(
                np.arange(1, N_eval + 1),
                RMSE_Ti,
                c=clist[i],
                label=f"{network_names[i]}",
            )

    plt.plot(np.arange(1, N_eval + 1), RMSE_T_true, "--k", label="CM2.6")
    plt.xlabel(r"time $( days )$")
    plt.ylabel(r"RMSE $\overline{T}$")
    plt.xlim([0, N_eval])

    plt.legend()
    plt.savefig(
        Path(output_dir) / ("RMSE" + region + "_" + save_str + ".png"),
        bbox_inches="tight",
    )
    plt.clf()


def plot_metrics_acc(network_names, region, save_str, output_dir, ACC_T_true, ACC_Ts):
    plt.style.use("bmh")

    clist = ["#A00B41", "#3300EA", "#00DCDE", "#A6BD00"]

    # ACC
    N_eval = len(ACC_T_true)
    for i, ACC_Ti in enumerate(ACC_Ts):
        if ACC_Ti is not None:
            plt.plot(
                np.arange(1, N_eval + 1),
                ACC_Ti,
                c=clist[i],
                label=f"{network_names[i]}",
            )

    plt.plot(np.arange(1, N_eval + 1), ACC_T_true, "--k", label="CM2.6")
    plt.xlabel(r"time $( days )$")
    plt.ylabel(r"ACC $\overline{T}$")
    plt.ylim([0, 1])
    plt.xlim([0, N_eval])

    plt.legend(loc="lower left")
    plt.savefig(
        Path(output_dir) / ("ACC" + region + "_" + save_str + ".png"),
        bbox_inches="tight",
    )
    plt.clf()


def plot_metrics_pdf(
    network_names,
    region,
    output_dir,
    pdf,
):
    plt.style.use("bmh")

    clist = ["#A00B41", "#3300EA", "#00DCDE", "#A6BD00"]

    # PDF
    var_list = {
        "1": r"$\overline{v}$ $( m/s )$",
        "0": r"$\overline{u}$ $( m/s )$",
        "2": r"$\overline{T}$ $( ^\circ C )$",
    }

    for ind_plot in range(3):
        plt.semilogy(*pdf[ind_plot]["true"], lw=2, c="k", label="CM2.6")
        for i, network_name in enumerate(network_names):
            plt.semilogy(
                *pdf[ind_plot][network_name],
                lw=2,
                color=clist[i],
                label=f"{network_name}",
            )

        plt.ylim(
            [
                pdf[ind_plot]["true_pdf"].min() * 2,
                pdf[ind_plot]["true_pdf"].max() * 2.5,
            ]
        )

        plt.legend()

        plt.xlabel(var_list[str(ind_plot)])
        plt.ylabel(r"${p(}$" + var_list[str(ind_plot)][:14] + "${)}$")

        plt.savefig(
            Path(output_dir) / ("PDF" + region + "_" + str(ind_plot) + ".png"),
            bbox_inches="tight",
        )
        plt.clf()


def plot_long_KE(
    network_names,
    region,
    save_str,
    output_dir,
    grids,
    Nb,
    wet_nan,
    long_KE_true,
    long_KEs,
):

    plt.style.use("bmh")

    # Long KE
    plt.rcParams.update({"font.size": 12})
    if len(long_KEs) > 1:
        fig, axs = plt.subplots(
            2,
            3,
            figsize=(12, 5),
            gridspec_kw={
                "width_ratios": [1, 1, 1],
                "height_ratios": [1, 1],
                "wspace": 0.25,
                "hspace": 0.5,
            },
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
    elif len(long_KEs) == 1:
        fig, axs = plt.subplots(
            2,
            2,
            figsize=(12, 5),
            gridspec_kw={
                "width_ratios": [1, 1],
                "height_ratios": [1, 1],
                "wspace": 0.25,
                "hspace": 0.5,
            },
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
    else:
        print("0 entries in long_KE")
        return

    vmin = 0
    vmax = 45

    if "global" in region:
        x_plot = grids["x_C"]
        y_plot = grids["y_C"]
    else:
        x_plot = grids["x_C"][Nb:-Nb, Nb:-Nb]
        y_plot = grids["y_C"][Nb:-Nb, Nb:-Nb]

    cmap = cmocean.cm.thermal  # cmocean.cm.diff

    # Ground Truth
    if "global" in region:
        plt0 = axs[0, 0].pcolormesh(
            x_plot,
            y_plot,
            long_KE_true * wet_nan,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
    else:
        plt0 = axs[0, 0].pcolormesh(
            x_plot,
            y_plot,
            long_KE_true[Nb:-Nb, Nb:-Nb] * wet_nan[Nb:-Nb, Nb:-Nb],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )

    axs[0, 0].add_feature(cart.feature.LAND, zorder=100, edgecolor="k")
    gl = axs[0, 0].gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=2,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.yrotation = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    axs[0, 0].set_title(r"CM2.6", size=15)

    pos = axs[1, 0].get_position()

    # Set the new anchor point to be in the middle
    new_pos = [
        pos.x0 - 0.075,
        pos.y0 + 0.15,
        pos.width * 1.75,
        pos.height * 1.5,
    ]  # Adjust 0.2 as needed

    # Create a new axes with the adjusted position
    cax = fig.add_axes(new_pos)

    cbar = plt.colorbar(plt0, ax=cax, orientation="horizontal", aspect=10)
    cbar.ax.tick_params(labelsize=16)  # Set the font size for tick labels

    cbar.set_ticks(
        [np.ceil(vmin), np.round((vmin + vmax) / 2), np.floor(vmax)]
    )  # cbar.set_ticks([vmin, 0, vmax])

    cbar.set_label(r"KE $( J/m^2 )$", fontsize=20)

    fig.delaxes(axs[1, 1])
    fig.delaxes(cax)

    for i, long_KE_i in enumerate(long_KEs):
        if long_KE_i is not None:
            if i == 0:
                idy, idx = 0, 1
            elif i == 1:
                idy, idx = 0, 2
            elif i == 2:
                idy, idx = 1, 2

            if "global" in region:
                axs[idy, idx].pcolormesh(
                    x_plot,
                    y_plot,
                    long_KE_i * wet_nan,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    shading="auto",
                )
            else:
                axs[idy, idx].pcolormesh(
                    x_plot,
                    y_plot,
                    long_KE_i[Nb:-Nb, Nb:-Nb] * wet_nan[Nb:-Nb, Nb:-Nb],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    shading="auto",
                )

            axs[idy, idx].add_feature(cart.feature.LAND, zorder=100, edgecolor="k")
            gl = axs[idy, idx].gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=True,
                linewidth=2,
                color="gray",
                alpha=0.5,
                linestyle="--",
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.yrotation = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            axs[idy, idx].set_title(network_names[i], size=15)

    axs[1, 0].set_axis_off()
    axs[1, 2].set_axis_off()
    if len(long_KEs) < 3:
        axs[1, 1].set_axis_off()

    region_title = ""

    for i in region:
        if region == "Quiescent_Ext":
            region_title = "South Pacific"
        elif region == "Africa_Ext":
            region_title = "African Cape"
        elif i == "_":
            region_title += " "
        elif i == "E":
            break
        else:
            region_title += i
    region_title = str(region_title)

    # a = fig.suptitle(
    #     r"Mean KE " + region_title + ": $t = " + str(1000) + "$ days ",
    #     fontsize=16,
    # )

    plt.savefig(
        Path(output_dir) / ("Mean_KE" + region + "_" + save_str + ".png"),
        bbox_inches="tight",
    )
    plt.clf()


def get_initial_snapshot_fig(
    network_names,
    N_plot,
    region,
    grids,
    test_data,
    wet_nan,
    model_preds,
    mean_out,
    std_out,
    ind_plot,
    Nb,
):

    plt.rcParams.update({"font.size": 12})
    var_list = {
        "1": r"$\overline{v}$ $( m/s )$",
        "0": r"$\overline{u}$ $(m/s)$",
        "2": r"$\overline{T}$ $( ^\circ C )$",
    }
    if len(model_preds) > 1:
        fig, axs = plt.subplots(
            2,
            3,
            figsize=(12, 5),
            gridspec_kw={
                "width_ratios": [1, 1, 1],
                "height_ratios": [1, 1],
                "wspace": 0.25,
                "hspace": 0.5,
            },
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
    elif len(model_preds) == 1:
        fig, axs = plt.subplots(
            2,
            2,
            figsize=(12, 5),
            gridspec_kw={
                "width_ratios": [1, 1],
                "height_ratios": [1, 1],
                "wspace": 0.25,
                "hspace": 0.5,
            },
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
    else:
        print("0 entries in model_preds")
        return

    T_plot = 1000

    vmin = mean_out[ind_plot] - std_out[ind_plot]
    vmax = mean_out[ind_plot] + std_out[ind_plot]

    if region == "Tropics_Ext" and ind_plot == 2:
        vmin = mean_out[ind_plot] - (0.5 * std_out[ind_plot])
        vmax = mean_out[ind_plot] + (std_out[ind_plot])
    elif region == "Africa_Ext" and ind_plot == 2:
        vmin = mean_out[ind_plot] - (1.25 * std_out[ind_plot])
        vmax = mean_out[ind_plot] + (2 * std_out[ind_plot])
    elif region == "Gulf_Stream_Ext" and ind_plot == 2:
        vmin = mean_out[ind_plot] - (1.75 * std_out[ind_plot])
        vmax = mean_out[ind_plot] + (1.75 * std_out[ind_plot])
    elif ind_plot == 2:
        vmax = 40

    if ind_plot in [0, 1]:
        vmin -= std_out[ind_plot]
        vmax += std_out[ind_plot]
        limit = np.round(np.max([abs(vmin), abs(vmax)]), 1)
        vmin = -limit
        vmax = limit

    if "global" in region:
        x_plot = grids["x_C"]
        y_plot = grids["y_C"]
    else:
        x_plot = grids["x_C"][Nb:-Nb, Nb:-Nb]
        y_plot = grids["y_C"][Nb:-Nb, Nb:-Nb]

    if ind_plot == 2:
        cmap = cmocean.cm.thermal
    else:
        cmap = cmocean.cm.diff

    # Ground Truth
    if "global" in region:
        plt0 = axs[0, 0].pcolormesh(
            x_plot,
            y_plot,
            test_data[N_plot - 1][1][ind_plot].cpu() * wet_nan * std_out[ind_plot]
            + mean_out[ind_plot],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
    else:
        plt0 = axs[0, 0].pcolormesh(
            x_plot,
            y_plot,
            test_data[N_plot - 1][1][ind_plot, Nb:-Nb, Nb:-Nb].cpu()
            * wet_nan[Nb:-Nb, Nb:-Nb]
            * std_out[ind_plot]
            + mean_out[ind_plot],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )

    axs[0, 0].add_feature(cart.feature.LAND, zorder=100, edgecolor="k")
    gl = axs[0, 0].gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=2,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.yrotation = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    axs[0, 0].set_title(r"CM2.6", size=15)

    pos = axs[1, 0].get_position()

    # Set the new anchor point to be in the middle
    new_pos = [
        pos.x0 - 0.075,
        pos.y0 + 0.15,
        pos.width * 1.75,
        pos.height * 1.5,
    ]  # Adjust 0.2 as needed

    # Create a new axes with the adjusted position
    cax = fig.add_axes(new_pos)

    cbar = plt.colorbar(plt0, ax=cax, orientation="horizontal", aspect=10)
    cbar.ax.tick_params(labelsize=16)  # Set the font size for tick labels
    if ind_plot == 2:
        cbar.set_ticks([np.ceil(vmin), np.round((vmin + vmax) / 2), np.floor(vmax)])
    else:
        cbar.set_ticks([vmin, 0, vmax])

    cbar.set_label(var_list[str(ind_plot)], fontsize=20)

    fig.delaxes(axs[1, 1])
    fig.delaxes(cax)

    plts = [plt0]
    for i, model_pred in enumerate(model_preds):
        if model_pred is not None:
            if i == 0:
                idy, idx = 0, 1
            elif i == 1:
                idy, idx = 0, 2
            elif i == 2:
                idy, idx = 1, 2

            if "global" in region:
                plt_temp = axs[idy, idx].pcolormesh(
                    x_plot,
                    y_plot,
                    model_pred[T_plot - 1, :, :, ind_plot] * wet_nan,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    shading="auto",
                )
            else:
                plt_temp = axs[idy, idx].pcolormesh(
                    x_plot,
                    y_plot,
                    model_pred[T_plot - 1, Nb:-Nb, Nb:-Nb, ind_plot]
                    * wet_nan[Nb:-Nb, Nb:-Nb],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    shading="auto",
                )

            axs[idy, idx].add_feature(cart.feature.LAND, zorder=100, edgecolor="k")
            gl = axs[idy, idx].gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=True,
                linewidth=2,
                color="gray",
                alpha=0.5,
                linestyle="--",
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.yrotation = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            axs[idy, idx].set_title(network_names[i], size=15)
            plts.append(plt_temp)

    axs[1, 0].set_axis_off()
    axs[1, 2].set_axis_off()
    if len(model_preds) < 3:
        axs[1, 1].set_axis_off()

    region_title = ""

    for i in region:
        if region == "Quiescent_Ext":
            region_title = "South Pacific"
        elif region == "Africa_Ext":
            region_title = "African Cape"
        elif i == "_":
            region_title += " "
        elif i == "E":
            break
        else:
            region_title += i
    region_title = str(region_title)

    a = fig.suptitle(
        r"$t = " + str(N_plot) + "$ days ",
        fontsize=16,
    )
    return fig, plts, a
