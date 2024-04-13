import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cmocean
from pathlib import Path
import numpy as np
import cartopy.crs as ccrs
import cartopy as cart
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def plot_time_spec(
    network,
    unet_network,
    axs,
    plt_index,
    index,
    N_test,
    freqs,
    auto_FFT,
    FFTs_unet,
    FFTs_net,
    clist,
    legend=True,
):
    T_plot = 200

    N_int = int(T_plot)
    N_true = min(N_test, N_int)

    var_list = {
        "1": r"$\widehat{\bar{v}} ~~\mathrm{(m/s)}$",
        "0": r"$\widehat{\bar{u}} ~~\mathrm{(m/s)}$",
        "2": r"$\widehat{\bar{T}} ~ (^\circ C)$",
    }

    axs[plt_index].semilogx(
        freqs[:N_int], auto_FFT[:N_int, index], "--k", label="CM2.6", zorder=5
    )

    if FFTs_unet is not None:
        axs[plt_index].plot(
            freqs[:N_int],
            FFTs_unet.mean(axis=0)[:N_int, index],
            color=clist[2],
            label=unet_network + r"($\mathbf{u},\tau_u,\tau_v,T_{\mathrm{atm}}$)",
        )
        axs[plt_index].fill_between(
            freqs[:N_int],
            FFTs_unet.mean(axis=0)[:N_int, index]
            - FFTs_unet.std(axis=0)[:N_int, index],
            FFTs_unet.mean(axis=0)[:N_int, index]
            + FFTs_unet.std(axis=0)[:N_int, index],
            ls="--",
            color=clist[2],
            alpha=0.25,
        )

    if FFTs_net is not None:
        axs[plt_index].plot(
            freqs[:N_int],
            FFTs_net.mean(axis=0)[:N_int, index],
            color=clist[3],
            label=network + r"($\mathbf{u},\tau_u,\tau_v,T_{\mathrm{atm}}$)",
        )
        axs[plt_index].fill_between(
            freqs[:N_int],
            FFTs_net.mean(axis=0)[:N_int, index] - FFTs_net.std(axis=0)[:N_int, index],
            FFTs_net.mean(axis=0)[:N_int, index] + FFTs_net.std(axis=0)[:N_int, index],
            ls="--",
            color=clist[3],
            alpha=0.25,
        )

    axs[plt_index].set_ylabel(r"" + var_list[str(index)])
    axs[plt_index].set_xlabel("Frequency (1/day)")

    axs[plt_index].set_xlim([0, freqs[T_plot]])
    axs[plt_index].set_ylim([0, auto_FFT[1:N_int, index].max() * 2])

    if legend:
        axs[plt_index].legend(ncol=1, loc="upper right")

    # plt.tight_layout()


def plot_var(
    network,
    unet_network,
    axs,
    plt_index,
    index,
    N_test,
    lag,
    auto_var,
    var_unet,
    var_net,
    clist,
):
    T_plot = 1098

    N_int = int(T_plot / lag)
    N_true = min(N_test, N_int)

    var_list = {
        "1": r"$\mathrm{Var}(\bar{v})$",
        "0": r"$\mathrm{Var}(\bar{u})$",
        "2": r"$\mathrm{Var}(\bar{T})$",
    }

    axs[plt_index].plot(
        (np.arange(N_int) * lag) / 366,
        auto_var[:N_int, index],
        "--k",
        label="CM2.6",
        zorder=5,
    )

    if var_unet is not None:
        axs[plt_index].plot(
            (np.arange(N_int) * lag) / 366,
            var_unet.mean(axis=0)[:N_int, index],
            color=clist[2],
            label=unet_network + r"($\mathbf{u},\tau_u,\tau_v,T_{ref}$)",
        )
        axs[plt_index].fill_between(
            (np.arange(N_int) * lag) / 366,
            var_unet.mean(axis=0)[:N_int, index] - var_unet.std(axis=0)[:N_int, index],
            var_unet.mean(axis=0)[:N_int, index] + var_unet.std(axis=0)[:N_int, index],
            ls="--",
            color=clist[2],
            alpha=0.25,
        )

    if var_net is not None:
        axs[plt_index].plot(
            (np.arange(N_int) * lag) / 366,
            var_net.mean(axis=0)[:N_int, index],
            color=clist[3],
            label=network + r"($\mathbf{u},\tau_u,\tau_v,T_{ref}$)",
        )
        axs[plt_index].fill_between(
            (np.arange(N_int) * lag) / 366,
            var_net.mean(axis=0)[:N_int, index] - var_net.std(axis=0)[:N_int, index],
            var_net.mean(axis=0)[:N_int, index] + var_net.std(axis=0)[:N_int, index],
            ls="--",
            color=clist[3],
            alpha=0.25,
        )

    axs[plt_index].set_ylabel(r"" + var_list[str(index)])
    axs[plt_index].set_xlabel("Time (years)")

    axs[plt_index].set_xlim([0, T_plot / 366])
    axs[plt_index].yaxis.set_major_formatter(
        ticker.ScalarFormatter(useMathText=True, useOffset=False)
    )
    axs[plt_index].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[plt_index].xaxis.set_major_locator(
        ticker.MultipleLocator(base=0.5)
    )  # Adjust base as needed


#     axs[plt_index].legend(ncol=2)


def plot_mean(
    network,
    unet_network,
    axs,
    plt_index,
    index,
    N_test,
    lag,
    auto_mean,
    mean_unet,
    mean_net,
    clist,
):

    T_plot = N_test

    N_int = int(T_plot / lag)
    N_true = min(N_test, N_int)

    var_list = {
        "1": r"$\bar{v}~~\mathrm{(m/s)}$",
        "0": r"$\bar{u}~~\mathrm{(m/s)}$",
        "2": r"$\bar{T} ~ (^\circ C)$",
    }

    axs[plt_index].plot(
        (np.arange(N_int) * lag) / 366,
        auto_mean[:N_int, index],
        "--k",
        label="CM2.6",
        zorder=5,
    )

    if mean_unet is not None:
        axs[plt_index].plot(
            (np.arange(N_int) * lag) / 366,
            mean_unet.mean(axis=0)[:N_int, index],
            color=clist[2],
            label=unet_network + r"($\mathbf{u},\tau_u,\tau_v,T_{ref}$)",
        )
        axs[plt_index].fill_between(
            (np.arange(N_int) * lag) / 366,
            mean_unet.mean(axis=0)[:N_int, index]
            - mean_unet.std(axis=0)[:N_int, index],
            mean_unet.mean(axis=0)[:N_int, index]
            + mean_unet.std(axis=0)[:N_int, index],
            ls="--",
            color=clist[2],
            alpha=0.25,
        )

    if mean_net is not None:
        axs[plt_index].plot(
            (np.arange(N_int) * lag) / 366,
            mean_net.mean(axis=0)[:N_int, index],
            color=clist[3],
            label=network + r"($\mathbf{u},\tau_u,\tau_v,T_{ref}$)",
        )
        axs[plt_index].fill_between(
            (np.arange(N_int) * lag) / 366,
            mean_net.mean(axis=0)[:N_int, index] - mean_net.std(axis=0)[:N_int, index],
            mean_net.mean(axis=0)[:N_int, index] + mean_net.std(axis=0)[:N_int, index],
            ls="--",
            color=clist[3],
            alpha=0.25,
        )

    axs[plt_index].set_ylabel(r"" + var_list[str(index)])
    axs[plt_index].set_xlabel("Time (years)")

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
    network,
    unet_network,
    axs,
    plt_ind_acc,
    index,
    N_test,
    lag,
    auto_ACC,
    ACC_unet,
    ACC_net,
    clist,
):
    T_plot = 100

    N_int = int(T_plot / lag)
    N_true = min(N_test, N_int)

    var_list = {
        "1": r"$\bar{v}$ (m/s)",
        "0": r"$\bar{u}$ (m/s)",
        "2": r"$\bar{T} ~ (^\circ C)$",
    }

    axs[plt_ind_acc].plot(
        (np.arange(N_int) * lag),
        auto_ACC.mean(axis=0)[:N_int, index],
        color="dimgrey",
        label="$\mathbf{\Phi}(t=0)$",
    )
    axs[plt_ind_acc].fill_between(
        (np.arange(N_int) * lag),
        auto_ACC.mean(axis=0)[:N_int, index] - auto_ACC.std(axis=0)[:N_int, index],
        auto_ACC.mean(axis=0)[:N_int, index] + auto_ACC.std(axis=0)[:N_int, index],
        ls="-",
        color="dimgrey",
        alpha=0.2,
    )

    if ACC_unet is not None:
        axs[plt_ind_acc].plot(
            (np.arange(N_int) * lag),
            ACC_unet.mean(axis=0)[:N_int, index],
            color=clist[2],
            label=unet_network + r"($\mathbf{\Phi},\tau_u,\tau_v,T_{ref}$)",
        )
        axs[plt_ind_acc].fill_between(
            (np.arange(N_int) * lag),
            ACC_unet.mean(axis=0)[:N_int, index] - ACC_unet.std(axis=0)[:N_int, index],
            ACC_unet.mean(axis=0)[:N_int, index] + ACC_unet.std(axis=0)[:N_int, index],
            ls="-",
            color=clist[2],
            alpha=0.2,
        )

    if ACC_net is not None:
        axs[plt_ind_acc].plot(
            (np.arange(N_int) * lag),
            ACC_net.mean(axis=0)[:N_int, index],
            color=clist[3],
            label=network + r"($\mathbf{\Phi},\tau_u,\tau_v,T_{ref}$)",
        )
        axs[plt_ind_acc].fill_between(
            (np.arange(N_int) * lag),
            ACC_net.mean(axis=0)[:N_int, index] - ACC_net.std(axis=0)[:N_int, index],
            ACC_net.mean(axis=0)[:N_int, index] + ACC_net.std(axis=0)[:N_int, index],
            ls="-",
            color=clist[3],
            alpha=0.2,
        )

    axs[plt_ind_acc].set_ylabel(r"ACC $" + var_list[str(index)][6] + "$")
    axs[plt_ind_acc].set_xlabel("Time (days)")

    axs[plt_ind_acc].set_ylim([0, 1])
    axs[plt_ind_acc].set_xlim([0, T_plot])


#     axs[plt_ind_acc].legend(ncol=2)

#     axs[plt_ind_acc].set_title("Short Rollout "+ region)


def plot_corr(
    network,
    unet_network,
    axs,
    plt_ind_acc,
    index,
    N_test,
    lag,
    auto_corrs,
    corrs_unet,
    corrs_net,
    clist,
):

    T_plot = 100

    N_int = int(T_plot / lag)
    N_true = min(N_test, N_int)

    var_list = {
        "1": r"$\bar{v}$ (m/s)",
        "0": r"$\bar{u}$ (m/s)",
        "2": r"$\bar{T} ~ (^\circ C)$",
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

    if corrs_unet is not None:
        axs[plt_ind_acc].plot(
            (np.arange(N_int) * lag),
            corrs_unet.mean(axis=0)[:N_int, index],
            color=clist[2],
            label=unet_network + r"($\mathbf{\Phi},\tau_u,\tau_v,T_{ref}$)",
        )
        axs[plt_ind_acc].fill_between(
            (np.arange(N_int) * lag),
            corrs_unet.mean(axis=0)[:N_int, index]
            - corrs_unet.std(axis=0)[:N_int, index],
            corrs_unet.mean(axis=0)[:N_int, index]
            + corrs_unet.std(axis=0)[:N_int, index],
            ls="-",
            color=clist[2],
            alpha=0.2,
        )

    if corrs_net is not None:
        axs[plt_ind_acc].plot(
            (np.arange(N_int) * lag),
            corrs_net.mean(axis=0)[:N_int, index],
            color=clist[3],
            label=network + r"($\mathbf{\Phi},\tau_u,\tau_v,T_{ref}$)",
        )
        axs[plt_ind_acc].fill_between(
            (np.arange(N_int) * lag),
            corrs_net.mean(axis=0)[:N_int, index]
            - corrs_net.std(axis=0)[:N_int, index],
            corrs_net.mean(axis=0)[:N_int, index]
            + corrs_net.std(axis=0)[:N_int, index],
            ls="-",
            color=clist[3],
            alpha=0.2,
        )

    axs[plt_ind_acc].set_ylabel(r"Correlation $" + var_list[str(index)][6] + "$")
    axs[plt_ind_acc].set_xlabel("Time (days)")

    axs[plt_ind_acc].set_ylim([0, 1])
    axs[plt_ind_acc].set_xlim([0, T_plot])


#     axs[plt_ind_acc].legend(ncol=2)

#     axs[plt_ind_acc].set_title("Short Rollout "+ region)


def plot_KE(
    network,
    unet_network,
    axs,
    plt_ind_acc,
    N_test,
    lag,
    auto_KE,
    KE_unet,
    KE_net,
    clist,
):

    T_plot = 200

    N_int = int(T_plot / lag)
    N_true = min(N_test, N_int)

    var_list = {
        "1": r"$\bar{v}$ (m/s)",
        "0": r"$\bar{u}$ (m/s)",
        "2": r"$\bar{T} ~ (^\circ C)$",
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

    if KE_unet is not None:
        axs[plt_ind_acc].plot(
            (np.arange(N_int) * lag) / 366,
            KE_unet.mean(axis=0)[:N_int],
            color=clist[2],
            label=unet_network + r"($\mathbf{\Phi},\tau_u,\tau_v,T_{ref}$)",
        )
        axs[plt_ind_acc].fill_between(
            (np.arange(N_int) * lag) / 366,
            KE_unet.mean(axis=0)[:N_int] - KE_unet.std(axis=0)[:N_int],
            KE_unet.mean(axis=0)[:N_int] + KE_unet.std(axis=0)[:N_int],
            ls="-",
            color=clist[2],
            alpha=0.2,
        )

    if KE_net is not None:
        axs[plt_ind_acc].plot(
            (np.arange(N_int) * lag) / 366,
            KE_net.mean(axis=0)[:N_int],
            color=clist[3],
            label=network + r"($\mathbf{\Phi},\tau_u,\tau_v,T_{ref}$)",
        )
        axs[plt_ind_acc].fill_between(
            (np.arange(N_int) * lag) / 366,
            KE_net.mean(axis=0)[:N_int] - KE_net.std(axis=0)[:N_int],
            KE_net.mean(axis=0)[:N_int] + KE_net.std(axis=0)[:N_int],
            ls="-",
            color=clist[3],
            alpha=0.2,
        )

    axs[plt_ind_acc].set_ylabel(r"KE")
    axs[plt_ind_acc].set_xlabel("Time (days)")

    axs[plt_ind_acc].set_ylim([0, 0.05])
    # axs[plt_ind_acc].set_yticks([-.1,-.05,0,.05,.1])


#     axs[plt_ind_acc].legend(ncol=2)

#     if region == "Quiescent":
#         axs[plt_ind_acc].set_title("Long Rollout South Pacific")
#     else:
#         axs[plt_ind_acc].set_title("Long Rollout "+ region)

# plt.tight_layout()
# plt.savefig("/scratch/as15415/Emulation/Figures/Comp_KE_region"+region+".png",bbox_inches='tight')


def plot_rmse(
    network,
    unet_network,
    axs,
    plt_ind_acc,
    index,
    N_test,
    lag,
    auto_rmse,
    rmse_unet,
    rmse_net,
    clist,
):
    T_plot = 200

    N_int = int(T_plot / lag)
    N_true = min(N_test, N_int)

    var_list = {
        "1": r"$\bar{v}$ (m/s)",
        "0": r"$\bar{u}$ (m/s)",
        "2": r"$\bar{T} ~ (^\circ C)$",
    }

    axs[plt_ind_acc].plot(
        (np.arange(N_int) * lag),
        auto_rmse.mean(axis=0)[:N_int, index],
        color="dimgrey",
        label="$\mathbf{\Phi}(t=0)$",
    )
    axs[plt_ind_acc].fill_between(
        (np.arange(N_int) * lag),
        auto_rmse.mean(axis=0)[:N_int, index] - auto_rmse.std(axis=0)[:N_int, index],
        auto_rmse.mean(axis=0)[:N_int, index] + auto_rmse.std(axis=0)[:N_int, index],
        ls="-",
        color="dimgrey",
        alpha=0.2,
    )

    if rmse_unet is not None:
        axs[plt_ind_acc].plot(
            (np.arange(N_int) * lag),
            rmse_unet.mean(axis=0)[:N_int, index],
            color=clist[2],
            label=unet_network + r"($\mathbf{\Phi},\tau_u,\tau_v,T_{ref}$)",
        )
        axs[plt_ind_acc].fill_between(
            (np.arange(N_int) * lag),
            rmse_unet.mean(axis=0)[:N_int, index]
            - rmse_unet.std(axis=0)[:N_int, index],
            rmse_unet.mean(axis=0)[:N_int, index]
            + rmse_unet.std(axis=0)[:N_int, index],
            ls="-",
            color=clist[2],
            alpha=0.2,
        )

    if rmse_net is not None:
        axs[plt_ind_acc].plot(
            (np.arange(N_int) * lag),
            rmse_net.mean(axis=0)[:N_int, index],
            color=clist[3],
            label=network + r"($\mathbf{\Phi},\tau_u,\tau_v,T_{ref}$)",
        )
        axs[plt_ind_acc].fill_between(
            (np.arange(N_int) * lag),
            rmse_net.mean(axis=0)[:N_int, index] - rmse_net.std(axis=0)[:N_int, index],
            rmse_net.mean(axis=0)[:N_int, index] + rmse_net.std(axis=0)[:N_int, index],
            ls="-",
            color=clist[3],
            alpha=0.2,
        )

    axs[plt_ind_acc].set_ylabel(r"RMSE" + var_list[str(index)])
    axs[plt_ind_acc].set_xlabel("Time (days)")

    # axs[plt_ind_acc].set_ylim([0,.25])
    # axs[plt_ind_acc].set_yticks([0,0.05,.1,.15,.2,.25])
    axs[plt_ind_acc].set_xlim([0, T_plot])
    axs[plt_ind_acc].legend(ncol=2)
    if index == 2:
        axs[plt_ind_acc].set_ylim([0, 8])
    #     axs[plt_ind_acc].set_yticks([0,1,2,3,4,5])
    if index == 1 or index == 0:
        axs[plt_ind_acc].yaxis.set_major_formatter(
            ticker.ScalarFormatter(useMathText=True, useOffset=False)
        )
        axs[plt_ind_acc].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


def plot_long_time_stats(
    network,
    unet_network,
    region,
    save_str,
    output_dir,
    N_test,
    lag,
    freqs,
    auto_FFT,
    FFTs_unet,
    FFTs_net,
    auto_mean,
    mean_unet,
    mean_net,
):

    N = 5
    plt.clf()
    plt.style.use("bmh")

    clist_1 = [cmocean.cm.thermal(i / (N - 0.5)) for i in range(1, N)]
    clist_2 = ["#d7191c", "#abd9e9", "#2c7bb6", "#fdae61"]
    clist_3 = ["#91B59A", "#D6A922", "#1E88E5", "#A00B41"]
    clist = clist_3

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
        network,
        unet_network,
        axs,
        (0, 0),
        0,
        N_test,
        freqs,
        auto_FFT,
        FFTs_unet,
        FFTs_net,
        clist,
        False,
    )
    plot_mean(
        network,
        unet_network,
        axs,
        (0, 1),
        0,
        N_test,
        lag,
        auto_mean,
        mean_unet,
        mean_net,
        clist,
    )
    plot_time_spec(
        network,
        unet_network,
        axs,
        (1, 0),
        1,
        N_test,
        freqs,
        auto_FFT,
        FFTs_unet,
        FFTs_net,
        clist,
    )
    plot_mean(
        network,
        unet_network,
        axs,
        (1, 1),
        2,
        N_test,
        lag,
        auto_mean,
        mean_unet,
        mean_net,
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

    fig.suptitle("Long-Time Statistics " + region_title, fontsize=16)

    plt.savefig(
        Path(output_dir)
        / ("Long_Time_Comp_Boundary_" + region + "_" + save_str + ".png"),
        bbox_inches="tight",
    )
    plt.clf()


def plot_short_time_stats(
    network,
    unet_network,
    region,
    save_str,
    output_dir,
    N_test,
    lag,
    auto_ACC,
    ACC_unet,
    ACC_net,
    auto_rmse,
    rmse_unet,
    rmse_net,
    auto_KE,
    KE_unet,
    KE_net,
    auto_corrs,
    corrs_unet,
    corrs_net,
):
    N = 5

    clist_1 = [cmocean.cm.thermal(i / (N - 0.5)) for i in range(1, N)]
    clist_2 = ["#d7191c", "#abd9e9", "#2c7bb6", "#fdae61"]
    clist_3 = ["#91B59A", "#D6A922", "#1E88E5", "#A00B41"]
    clist = clist_3

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

    fig, axs = init_plt()
    plot_acc(
        network,
        unet_network,
        axs,
        (0, 0),
        2,
        N_test,
        lag,
        auto_ACC,
        ACC_unet,
        ACC_net,
        clist,
    )
    plot_corr(
        network,
        unet_network,
        axs,
        (0, 1),
        1,
        N_test,
        lag,
        auto_corrs,
        corrs_unet,
        corrs_net,
        clist,
    )
    plot_rmse(
        network,
        unet_network,
        axs,
        (1, 0),
        2,
        N_test,
        lag,
        auto_rmse,
        rmse_unet,
        rmse_net,
        clist,
    )
    plot_KE(
        network, unet_network, axs, (1, 1), N_test, lag, auto_KE, KE_unet, KE_net, clist
    )

    fig.suptitle("Short-Time Statistics 1" + region, fontsize=16)

    plt.savefig(
        Path(output_dir)
        / ("Short_Time_Comp_Boundary1_" + region + "_" + save_str + ".png"),
        bbox_inches="tight",
    )

    fig, axs = init_plt()
    plot_acc(
        network,
        unet_network,
        axs,
        (0, 0),
        0,
        N_test,
        lag,
        auto_ACC,
        ACC_unet,
        ACC_net,
        clist,
    )
    plot_acc(
        network,
        unet_network,
        axs,
        (0, 1),
        1,
        N_test,
        lag,
        auto_ACC,
        ACC_unet,
        ACC_net,
        clist,
    )
    plot_rmse(
        network,
        unet_network,
        axs,
        (1, 0),
        0,
        N_test,
        lag,
        auto_rmse,
        rmse_unet,
        rmse_net,
        clist,
    )
    plot_rmse(
        network,
        unet_network,
        axs,
        (1, 1),
        1,
        N_test,
        lag,
        auto_rmse,
        rmse_unet,
        rmse_net,
        clist,
    )

    fig.suptitle("Short-Time Statistics 2" + region, fontsize=16)

    plt.savefig(
        Path(output_dir)
        / ("Short_Time_Comp_Boundary2_" + region + "_" + save_str + ".png"),
        bbox_inches="tight",
    )

    plt.clf()


def plot_all_metrics(
    network,
    unet_network,
    region,
    save_str,
    output_dir,
    lag,
    steps,
    KE_spec_true,
    KE_spec_unet,
    KE_spec_net,
    KE_true,
    KE_unet,
    KE_net,
    enst_spec_true,
    enst_spec_unet,
    enst_spec_net,
    corr_T_true,
    corr_T_unet,
    corr_T_net,
    enst_true,
    enst_unet,
    enst_net,
    RMSE_T_true,
    RMSE_T_unet,
    RMSE_T_net,
    ACC_T_true,
    ACC_T_unet,
    ACC_T_net,
):
    N = 5
    N_plot = 200
    plt.clf()
    plt.style.use("bmh")

    clist_1 = [cmocean.cm.thermal(i / (N - 0.5)) for i in range(1, N)]
    clist_2 = ["#d7191c", "#abd9e9", "#2c7bb6", "#fdae61"]
    clist_3 = ["#91B59A", "#D6A922", "#1E88E5", "#A00B41"]
    clist_5 = ["#A00B41", "#00DCDE", "#A6BD00", "#3300EA"]
    clist_6 = ["#A00B41", "#DE7400", "#00BD8E", "#3300EA"]
    clist = clist_5

    # KE Spectrum
    if KE_spec_net is not None:
        plt.loglog(
            KE_spec_net.freq_r,
            KE_spec_net,
            c=clist[0],
            label=f"{network} ~ $\Delta t = {lag},~ N = {steps}$",
        )
    if KE_spec_unet is not None:
        plt.loglog(
            KE_spec_unet.freq_r,
            KE_spec_unet,
            c=clist[3],
            label=unet_network + f" ~ $\Delta t = {lag},~ N = {steps}$",
        )

    plt.loglog(KE_spec_true.freq_r, KE_spec_true, "--k")

    plt.xlabel("Wave number (1/km)")
    plt.ylabel("Kinetic Energy")

    plt.legend(loc="lower left")
    plt.savefig(
        Path(output_dir) / ("KE_spectrum" + region + "_" + save_str + ".png"),
        bbox_inches="tight",
    )
    plt.clf()

    # KE
    rho = 1020

    if KE_net is not None:
        plt.plot(
            np.arange(1, N_plot + 1),
            KE_net * rho,
            c=clist[0],
            label=f"{network} ~ $\Delta t = {lag},~ N = {steps}$",
        )
    if KE_unet is not None:
        plt.plot(
            np.arange(1, N_plot + 1),
            KE_unet * rho,
            c=clist[3],
            label=unet_network + f" ~ $\Delta t = {lag},~ N = {steps}$",
        )

    plt.plot(np.arange(1, N_plot + 1), KE_true * rho, "--k")
    plt.xlabel("time (days)")
    plt.ylabel("Kinetic Energy")
    plt.legend(loc="lower left")
    plt.savefig(
        Path(output_dir) / ("KE" + region + "_" + save_str + ".png"),
        bbox_inches="tight",
    )
    plt.clf()

    # Enstrophy Spectrum
    if enst_spec_net is not None:
        plt.loglog(
            enst_spec_net.freq_r,
            enst_spec_net,
            c=clist[0],
            label=f"{network} ~ $\Delta t = {lag},~ N = {steps}$",
        )
    if enst_spec_unet is not None:
        plt.loglog(
            enst_spec_unet.freq_r,
            enst_spec_unet,
            c=clist[3],
            label=unet_network + f" ~ $\Delta t = {lag},~ N = {steps}$",
        )

    plt.loglog(enst_spec_true.freq_r, enst_spec_true, "--k")
    plt.xlabel("Wave number (1/km)")
    plt.ylabel("Enstrophy")
    plt.legend(loc="lower left")
    plt.savefig(
        Path(output_dir) / ("Enstrophy_Spectrum" + region + "_" + save_str + ".png"),
        bbox_inches="tight",
    )
    plt.clf()

    # Enstrophy
    if enst_net is not None:
        plt.plot(
            np.arange(1, N_plot + 1),
            enst_net,
            c=clist[0],
            label=f"{network} ~ $\Delta t = {lag},~ N = {steps}$",
        )

    if enst_unet is not None:
        plt.plot(
            np.arange(1, N_plot + 1),
            enst_unet,
            c=clist[3],
            label=unet_network + f" ~ $\Delta t = {lag},~ N = {steps}$",
        )

    plt.plot(np.arange(1, N_plot + 1), enst_true, "--k")
    plt.xlabel("time (days)")
    plt.ylabel("Enstrophy")
    plt.legend(loc="lower left")
    plt.savefig(
        Path(output_dir) / ("Enstrophy" + region + "_" + save_str + ".png"),
        bbox_inches="tight",
    )
    plt.clf()

    # Corr
    N_eval = 200
    if corr_T_net is not None:
        plt.plot(
            np.arange(1, N_eval + 1),
            corr_T_net,
            c=clist[0],
            label=f"{network} ~ $\Delta t = {lag},~ N = {steps}$",
        )

    if corr_T_unet is not None:
        plt.plot(
            np.arange(1, N_eval + 1),
            corr_T_unet,
            c=clist[3],
            label=unet_network + f" ~ $\Delta t = {lag},~ N = {steps}$",
        )

    plt.plot(np.arange(1, N_eval + 1), corr_T_true, "--k")
    plt.xlabel("time (days)")
    plt.ylabel(r"Correlation $T$")
    plt.ylim([0, 1])
    plt.xlim([0, N_eval])

    plt.legend(loc="lower left")
    plt.savefig(
        Path(output_dir) / ("Corr" + region + "_" + save_str + ".png"),
        bbox_inches="tight",
    )
    plt.clf()

    # RMSE
    if RMSE_T_net is not None:
        plt.plot(
            np.arange(1, N_eval + 1),
            RMSE_T_net,
            c=clist[0],
            label=f"{network} ~ $\Delta t = {lag},~ N = {steps}$",
        )

    if RMSE_T_unet is not None:
        plt.plot(
            np.arange(1, N_eval + 1),
            RMSE_T_unet,
            c=clist[3],
            label=unet_network + f" ~ $\Delta t = {lag},~ N = {steps}$",
        )

    plt.plot(np.arange(1, N_eval + 1), RMSE_T_true, "--k")
    plt.xlabel("time (days)")
    plt.ylabel(r"RMSE $T$")
    plt.xlim([0, N_eval])

    plt.legend()
    plt.savefig(
        Path(output_dir) / ("RMSE" + region + "_" + save_str + ".png"),
        bbox_inches="tight",
    )
    plt.clf()

    # ACC
    N_eval = 100
    if ACC_T_net is not None:
        plt.plot(
            np.arange(1, N_eval + 1),
            ACC_T_net,
            c=clist[0],
            label=f"{network} ~ $\Delta t = {lag},~ N = {steps}$",
        )

    if ACC_T_unet is not None:
        plt.plot(
            np.arange(1, N_eval + 1),
            ACC_T_unet,
            c=clist[3],
            label=unet_network + f" ~ $\Delta t = {lag},~ N = {steps}$",
        )

    plt.plot(np.arange(1, N_eval + 1), ACC_T_true, "--k")
    plt.xlabel("time (days)")
    plt.ylabel(r"ACC $T$")
    plt.ylim([0, 1])
    plt.xlim([0, N_eval])

    plt.legend(loc="lower left")
    plt.savefig(
        Path(output_dir) / ("ACC" + region + "_" + save_str + ".png"),
        bbox_inches="tight",
    )
    plt.clf()


def get_initial_snapshot_fig(
    network,
    unet_network,
    N_plot,
    region,
    grids,
    test_data,
    wet_nan,
    model_pred_net,
    model_pred_unet,
    mean_out,
    std_out,
    ind_plot,
    Nb,
    use_unet=True,
    only_unet=False,
):

    plt.rcParams.update({"font.size": 15})
    var_list = {
        "1": r"$\bar{v}~~\mathrm{(m/s)}$",
        "0": r"$\bar{u}~~\mathrm{(m/s)}$",
        "2": r"$\bar{T} ~ (^\circ C)$",
    }
    if use_unet:
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
    else:
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

    if ind_plot in [0, 1]:
        vmin -= std_out[ind_plot]
        vmax += std_out[ind_plot]
        limit = np.round(np.max([abs(vmin), abs(vmax)]), 1)
        vmin = -limit
        vmax = limit

    x_plot = grids["x_C"][Nb:-Nb, Nb:-Nb]
    y_plot = grids["y_C"][Nb:-Nb, Nb:-Nb]

    if ind_plot == 2:
        cmap = cmocean.cm.thermal
    else:
        cmap = cmocean.cm.diff

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

    pos = axs[1, 1].get_position()

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

    plt1 = None
    if not only_unet:
        plt1 = axs[0, 1].pcolormesh(
            x_plot,
            y_plot,
            model_pred_net[T_plot - 1, Nb:-Nb, Nb:-Nb, ind_plot]
            * wet_nan[Nb:-Nb, Nb:-Nb],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )

        axs[0, 1].add_feature(cart.feature.LAND, zorder=100, edgecolor="k")
        gl = axs[0, 1].gridlines(
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
        axs[0, 1].set_title(
            network + r"($\mathbf{u},\tau_u,\tau_v,T_{\mathrm{atm}}$)", size=15
        )

    plt2 = None
    if use_unet:
        plt2 = axs[0, 2].pcolormesh(
            x_plot,
            y_plot,
            model_pred_unet[T_plot - 1, Nb:-Nb, Nb:-Nb, ind_plot]
            * wet_nan[Nb:-Nb, Nb:-Nb],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )

        axs[0, 2].add_feature(cart.feature.LAND, zorder=100, edgecolor="k")
        gl = axs[0, 2].gridlines(
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
        axs[0, 2].set_title(
            unet_network + r"($\mathbf{u},\tau_u,\tau_v,T_{\mathrm{atm}}$)", size=15
        )
        axs[1, 2].set_axis_off()

    axs[1, 0].set_axis_off()

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
        r"Benefit of Atmospheric Boundary Terms "
        + region_title
        + ": $t = "
        + str(N_plot)
        + "$ days ",
        fontsize=16,
    )
    return fig, plt0, plt1, plt2, a
