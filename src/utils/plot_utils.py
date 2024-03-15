import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cmocean
from pathlib import Path
import numpy as np

def plot_time_spec(axs, plt_index, index, N_test, freqs, auto_FFT, FFTs_unet, FFTs_vit, clist, legend=True):
    T_plot = 200

    N_int = int(T_plot)
    N_true = min(N_test,N_int)

    var_list = {"1":r"$\widehat{\bar{v}} ~~\mathrm{(m/s)}$","0":r"$\widehat{\bar{u}} ~~\mathrm{(m/s)}$","2":r"$\widehat{\bar{T}} ~ (^\circ C)$"}



    axs[plt_index].semilogx(freqs[:N_int],auto_FFT[:N_int,index],"--k",label = "CM2.6",zorder=5)


    axs[plt_index].plot(freqs[:N_int],FFTs_unet.mean(axis=0)[:N_int,index],color=clist[2],label=r"Unet($\mathbf{u},\tau_u,\tau_v,T_{\mathrm{atm}}$)")
    axs[plt_index].fill_between(freqs[:N_int],FFTs_unet.mean(axis=0)[:N_int,index]-FFTs_unet.std(axis=0)[:N_int,index],
                     FFTs_unet.mean(axis=0)[:N_int,index]+FFTs_unet.std(axis=0)[:N_int,index],
                     ls="--",color=clist[2],alpha=.25)

    axs[plt_index].plot(freqs[:N_int],FFTs_vit.mean(axis=0)[:N_int,index],color=clist[3],label=r"ViT($\mathbf{u},\tau_u,\tau_v,T_{\mathrm{atm}}$)")
    axs[plt_index].fill_between(freqs[:N_int],FFTs_vit.mean(axis=0)[:N_int,index]-FFTs_vit.std(axis=0)[:N_int,index],
                     FFTs_vit.mean(axis=0)[:N_int,index]+FFTs_vit.std(axis=0)[:N_int,index],
                     ls="--",color=clist[3],alpha=.25)

    axs[plt_index].set_ylabel(r"" +var_list[str(index)])
    axs[plt_index].set_xlabel("Frequency (1/day)")

    axs[plt_index].set_xlim([0,freqs[T_plot]])
    axs[plt_index].set_ylim([0,auto_FFT[1:N_int,index].max()*2])

    if legend:
        axs[plt_index].legend(ncol=1, loc = "upper right")



    # plt.tight_layout()

    
def plot_var(axs, plt_index, index, N_test, lag, auto_var, var_unet, var_vit, clist):
    T_plot = 1098

    N_int = int(T_plot/lag)
    N_true = min(N_test,N_int)

    var_list = {"1":r"$\mathrm{Var}(\bar{v})$","0":r"$\mathrm{Var}(\bar{u})$","2":r"$\mathrm{Var}(\bar{T})$"}


    axs[plt_index].plot((np.arange(N_int)*lag)/366,auto_var[:N_int,index],"--k",label = "CM2.6",zorder=5)

    axs[plt_index].plot((np.arange(N_int)*lag)/366,var_unet.mean(axis=0)[:N_int,index],color=clist[2],label=r"Unet($\mathbf{u},\tau_u,\tau_v,T_{ref}$)")
    axs[plt_index].fill_between((np.arange(N_int)*lag)/366,var_unet.mean(axis=0)[:N_int,index]-var_unet.std(axis=0)[:N_int,index],
                     var_unet.mean(axis=0)[:N_int,index]+var_unet.std(axis=0)[:N_int,index],
                     ls="--",color=clist[2],alpha=.25)

    axs[plt_index].plot((np.arange(N_int)*lag)/366,var_vit.mean(axis=0)[:N_int,index],color=clist[3],label=r"ViT($\mathbf{u},\tau_u,\tau_v,T_{ref}$)")
    axs[plt_index].fill_between((np.arange(N_int)*lag)/366,var_vit.mean(axis=0)[:N_int,index]-var_vit.std(axis=0)[:N_int,index],
                     var_vit.mean(axis=0)[:N_int,index]+var_vit.std(axis=0)[:N_int,index],
                     ls="--",color=clist[3],alpha=.25)


    axs[plt_index].set_ylabel(r"" +var_list[str(index)])
    axs[plt_index].set_xlabel("Time (years)")

    axs[plt_index].set_xlim([0,T_plot/366])
    axs[plt_index].yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True, useOffset=False))
    axs[plt_index].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axs[plt_index].xaxis.set_major_locator(ticker.MultipleLocator(base=0.5))  # Adjust base as needed

#     axs[plt_index].legend(ncol=2)


def plot_mean(axs, plt_index, index, N_test, lag, auto_mean, mean_unet, mean_vit, clist):

    T_plot = 3000

    N_int = int(T_plot/lag)
    N_true = min(N_test,N_int)

    var_list = {"1":r"$\bar{v}~~\mathrm{(m/s)}$","0":r"$\bar{u}~~\mathrm{(m/s)}$","2":r"$\bar{T} ~ (^\circ C)$"}


    axs[plt_index].plot((np.arange(N_int)*lag)/366,auto_mean[:N_int,index],"--k",label = "CM2.6",zorder=5)

    axs[plt_index].plot((np.arange(N_int)*lag)/366,mean_unet.mean(axis=0)[:N_int,index],color=clist[2],label=r"Unet($\mathbf{u},\tau_u,\tau_v,T_{ref}$)")
    axs[plt_index].fill_between((np.arange(N_int)*lag)/366,mean_unet.mean(axis=0)[:N_int,index]-mean_unet.std(axis=0)[:N_int,index],
                     mean_unet.mean(axis=0)[:N_int,index]+mean_unet.std(axis=0)[:N_int,index],
                     ls="--",color=clist[2],alpha=.25)

    axs[plt_index].plot((np.arange(N_int)*lag)/366,mean_vit.mean(axis=0)[:N_int,index],color=clist[3],label=r"ViT($\mathbf{u},\tau_u,\tau_v,T_{ref}$)")
    axs[plt_index].fill_between((np.arange(N_int)*lag)/366,mean_vit.mean(axis=0)[:N_int,index]-mean_vit.std(axis=0)[:N_int,index],
                     mean_vit.mean(axis=0)[:N_int,index]+mean_vit.std(axis=0)[:N_int,index],
                     ls="--",color=clist[3],alpha=.25)


    axs[plt_index].set_ylabel(r"" +var_list[str(index)])
    axs[plt_index].set_xlabel("Time (years)")

    min_val = auto_mean[:N_int,index].min()
    max_val = auto_mean[:N_int,index].max()

    
    if min_val > 0:
        axs[plt_index].set_ylim([min_val*.8,max_val*1.1])
    elif min_val<0 and max_val>0:
        axs[plt_index].set_ylim([min_val*1.1,max_val*1.1])
    else:
        axs[plt_index].set_ylim([min_val*1.1,0])


    if index == 2:
        axs[plt_index].set_xlim([4,8])
        axs[plt_index].xaxis.set_major_locator(ticker.MultipleLocator(base=1))  # Adjust base as needed

    #     axs[plt_index].set_ylim([22,28])
    else:
        axs[plt_index].set_xlim([7,8])
        axs[plt_index].yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True, useOffset=False))
        axs[plt_index].ticklabel_format(axis='y', style='sci', scilimits=(0,0))    
        axs[plt_index].xaxis.set_major_locator(ticker.MultipleLocator(base=0.5))  # Adjust base as needed
   

def plot_acc(axs, plt_ind_acc, index, N_test, lag, auto_ACC, ACC_unet, ACC_vit, clist):
    T_plot = 100


    N_int = int(T_plot/lag)
    N_true = min(N_test,N_int)

    var_list = {"1":r"$\bar{v}$ (m/s)","0":r"$\bar{u}$ (m/s)","2":r"$\bar{T} ~ (^\circ C)$"}


    
    axs[plt_ind_acc].plot((np.arange(N_int)*lag),auto_ACC.mean(axis=0)[:N_int,index],color="dimgrey",label = "$\mathbf{\Phi}(t=0)$")
    axs[plt_ind_acc].fill_between((np.arange(N_int)*lag),auto_ACC.mean(axis=0)[:N_int,index]-auto_ACC.std(axis=0)[:N_int,index],
                     auto_ACC.mean(axis=0)[:N_int,index]+auto_ACC.std(axis=0)[:N_int,index],
                     ls="-",color="dimgrey",alpha=.2)

    axs[plt_ind_acc].plot((np.arange(N_int)*lag),ACC_unet.mean(axis=0)[:N_int,index],color=clist[2],label=r"Unet($\mathbf{\Phi},\tau_u,\tau_v,T_{ref}$)")
    axs[plt_ind_acc].fill_between((np.arange(N_int)*lag),ACC_unet.mean(axis=0)[:N_int,index]-ACC_unet.std(axis=0)[:N_int,index],
                     ACC_unet.mean(axis=0)[:N_int,index]+ACC_unet.std(axis=0)[:N_int,index],
                     ls="-",color=clist[2],alpha=.2)

    axs[plt_ind_acc].plot((np.arange(N_int)*lag),ACC_vit.mean(axis=0)[:N_int,index],color=clist[3],label=r"ViT($\mathbf{\Phi},\tau_u,\tau_v,T_{ref}$)")
    axs[plt_ind_acc].fill_between((np.arange(N_int)*lag),ACC_vit.mean(axis=0)[:N_int,index]-ACC_vit.std(axis=0)[:N_int,index],
                     ACC_vit.mean(axis=0)[:N_int,index]+ACC_vit.std(axis=0)[:N_int,index],
                     ls="-",color=clist[3],alpha=.2)


    axs[plt_ind_acc].set_ylabel(r"ACC $" +var_list[str(index)][6]+"$")
    axs[plt_ind_acc].set_xlabel("Time (days)")


    axs[plt_ind_acc].set_ylim([0,1])
    axs[plt_ind_acc].set_xlim([0,T_plot])
#     axs[plt_ind_acc].legend(ncol=2)

#     axs[plt_ind_acc].set_title("Short Rollout "+ region)

def plot_corr(axs, plt_ind_acc, index, N_test, lag, auto_corrs, corrs_unet, corrs_vit, clist):


    T_plot = 100

    N_int = int(T_plot/lag)
    N_true = min(N_test,N_int)

    var_list = {"1":r"$\bar{v}$ (m/s)","0":r"$\bar{u}$ (m/s)","2":r"$\bar{T} ~ (^\circ C)$"}




    axs[plt_ind_acc].plot((np.arange(N_int)*lag),auto_corrs.mean(axis=0)[:N_int,index],color="dimgrey",label = "$\mathbf{\Phi}(t=0)$")
    axs[plt_ind_acc].fill_between((np.arange(N_int)*lag),auto_corrs.mean(axis=0)[:N_int,index]-auto_corrs.std(axis=0)[:N_int,index],
                     auto_corrs.mean(axis=0)[:N_int,index]+auto_corrs.std(axis=0)[:N_int,index],
                     ls="-",color="dimgrey",alpha=.2)

    axs[plt_ind_acc].plot((np.arange(N_int)*lag),corrs_unet.mean(axis=0)[:N_int,index],color=clist[2],label=r"Unet($\mathbf{\Phi},\tau_u,\tau_v,T_{ref}$)")
    axs[plt_ind_acc].fill_between((np.arange(N_int)*lag),corrs_unet.mean(axis=0)[:N_int,index]-corrs_unet.std(axis=0)[:N_int,index],
                     corrs_unet.mean(axis=0)[:N_int,index]+corrs_unet.std(axis=0)[:N_int,index],
                     ls="-",color=clist[2],alpha=.2)

    axs[plt_ind_acc].plot((np.arange(N_int)*lag),corrs_vit.mean(axis=0)[:N_int,index],color=clist[3],label=r"ViT($\mathbf{\Phi},\tau_u,\tau_v,T_{ref}$)")
    axs[plt_ind_acc].fill_between((np.arange(N_int)*lag),corrs_vit.mean(axis=0)[:N_int,index]-corrs_vit.std(axis=0)[:N_int,index],
                     corrs_vit.mean(axis=0)[:N_int,index]+corrs_vit.std(axis=0)[:N_int,index],
                     ls="-",color=clist[3],alpha=.2)




    axs[plt_ind_acc].set_ylabel(r"Correlation $" +var_list[str(index)][6]+"$")
    axs[plt_ind_acc].set_xlabel("Time (days)")


    axs[plt_ind_acc].set_ylim([0,1])
    axs[plt_ind_acc].set_xlim([0,T_plot])
#     axs[plt_ind_acc].legend(ncol=2)

#     axs[plt_ind_acc].set_title("Short Rollout "+ region)


def plot_KE(axs, plt_ind_acc, N_test, lag, auto_KE, KE_unet, KE_vit, clist):

    T_plot = 200

    N_int = int(T_plot/lag)
    N_true = min(N_test,N_int)

    var_list = {"1":r"$\bar{v}$ (m/s)","0":r"$\bar{u}$ (m/s)","2":r"$\bar{T} ~ (^\circ C)$"}




    axs[plt_ind_acc].plot((np.arange(N_int)*lag)/366,auto_KE[:N_int].mean(axis=0),color="dimgrey",label = "$\mathbf{\Phi}(t=0)$")
    axs[plt_ind_acc].fill_between((np.arange(N_int)*lag)/366,auto_KE.mean(axis=0)[:N_int]-auto_KE.std(axis=0)[:N_int],
                     auto_KE.mean(axis=0)[:N_int]+auto_KE.std(axis=0)[:N_int],
                     ls="-",color="dimgrey",alpha=.2)

    axs[plt_ind_acc].plot((np.arange(N_int)*lag)/366,KE_unet.mean(axis=0)[:N_int],color=clist[2],label=r"Unet($\mathbf{\Phi},\tau_u,\tau_v,T_{ref}$)")
    axs[plt_ind_acc].fill_between((np.arange(N_int)*lag)/366,KE_unet.mean(axis=0)[:N_int]-KE_unet.std(axis=0)[:N_int],
                     KE_unet.mean(axis=0)[:N_int]+KE_unet.std(axis=0)[:N_int],
                     ls="-",color=clist[2],alpha=.2)

    axs[plt_ind_acc].plot((np.arange(N_int)*lag)/366,KE_vit.mean(axis=0)[:N_int],color=clist[3],label=r"ViT($\mathbf{\Phi},\tau_u,\tau_v,T_{ref}$)")
    axs[plt_ind_acc].fill_between((np.arange(N_int)*lag)/366,KE_vit.mean(axis=0)[:N_int]-KE_vit.std(axis=0)[:N_int],
                     KE_vit.mean(axis=0)[:N_int]+KE_vit.std(axis=0)[:N_int],
                     ls="-",color=clist[3],alpha=.2)

    axs[plt_ind_acc].set_ylabel(r"KE")
    axs[plt_ind_acc].set_xlabel("Time (days)")

    axs[plt_ind_acc].set_ylim([0,.05])
    # axs[plt_ind_acc].set_yticks([-.1,-.05,0,.05,.1])
#     axs[plt_ind_acc].legend(ncol=2)

#     if region == "Quiescent":
#         axs[plt_ind_acc].set_title("Long Rollout South Pacific")
#     else:
#         axs[plt_ind_acc].set_title("Long Rollout "+ region)

    # plt.tight_layout()
    # plt.savefig("/scratch/as15415/Emulation/Figures/Comp_KE_region"+region+".png",bbox_inches='tight')
    

def plot_rmse(axs, plt_ind_acc, index, N_test, lag, auto_rmse, rmse_unet, rmse_vit, clist):
    T_plot = 200

    N_int = int(T_plot/lag)
    N_true = min(N_test,N_int)


    var_list = {"1":r"$\bar{v}$ (m/s)","0":r"$\bar{u}$ (m/s)","2":r"$\bar{T} ~ (^\circ C)$"}


    axs[plt_ind_acc].plot((np.arange(N_int)*lag),auto_rmse.mean(axis=0)[:N_int,index],color="dimgrey",label = "$\mathbf{\Phi}(t=0)$")
    axs[plt_ind_acc].fill_between((np.arange(N_int)*lag),auto_rmse.mean(axis=0)[:N_int,index]-auto_rmse.std(axis=0)[:N_int,index]
                     ,auto_rmse.mean(axis=0)[:N_int,index]+auto_rmse.std(axis=0)[:N_int,index],
                     ls="-",color="dimgrey",alpha=.2)

    axs[plt_ind_acc].plot((np.arange(N_int)*lag),rmse_unet.mean(axis=0)[:N_int,index],color=clist[2],label=r"Unet($\mathbf{\Phi},\tau_u,\tau_v,T_{ref}$)")
    axs[plt_ind_acc].fill_between((np.arange(N_int)*lag),rmse_unet.mean(axis=0)[:N_int,index]-rmse_unet.std(axis=0)[:N_int,index],
                     rmse_unet.mean(axis=0)[:N_int,index]+rmse_unet.std(axis=0)[:N_int,index],
                     ls="-",color=clist[2],alpha=.2)

    axs[plt_ind_acc].plot((np.arange(N_int)*lag),rmse_vit.mean(axis=0)[:N_int,index],color=clist[3],label=r"ViT($\mathbf{\Phi},\tau_u,\tau_v,T_{ref}$)")
    axs[plt_ind_acc].fill_between((np.arange(N_int)*lag),rmse_vit.mean(axis=0)[:N_int,index]-rmse_vit.std(axis=0)[:N_int,index],
                     rmse_vit.mean(axis=0)[:N_int,index]+rmse_vit.std(axis=0)[:N_int,index],
                     ls="-",color=clist[3],alpha=.2)


    axs[plt_ind_acc].set_ylabel(r"RMSE" +var_list[str(index)])
    axs[plt_ind_acc].set_xlabel("Time (days)")


    # axs[plt_ind_acc].set_ylim([0,.25])
    # axs[plt_ind_acc].set_yticks([0,0.05,.1,.15,.2,.25])
    axs[plt_ind_acc].set_xlim([0,T_plot])
    axs[plt_ind_acc].legend(ncol=2)
    if index == 2:
        axs[plt_ind_acc].set_ylim([0,8])
    #     axs[plt_ind_acc].set_yticks([0,1,2,3,4,5])
    if index == 1 or index == 0:
        axs[plt_ind_acc].yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True, useOffset=False))
        axs[plt_ind_acc].ticklabel_format(axis='y', style='sci', scilimits=(0,0))   


def plot_long_time_stats(region, save_str, output_dir, N_test, lag, freqs, auto_FFT, FFTs_unet, FFTs_vit,\
                        auto_mean, mean_unet, mean_vit):
    
    N = 5
    plt.clf()
    plt.style.use('bmh')

    clist_1 = [cmocean.cm.thermal(i/(N-.5)) for i in range(1,N)]
    clist_2 = ['#d7191c','#abd9e9','#2c7bb6','#fdae61']
    clist_3 = ["#91B59A","#D6A922","#1E88E5","#A00B41"]
    clist = clist_3

    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=20)     # fontsize of the axes title
    plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    plt.rc('legend', fontsize=10)    # legend fontsize
    plt.rc('figure', titlesize=18)


    fig, axs = plt.subplots(2, 2, figsize=(11,6),
                            gridspec_kw={'width_ratios': [1,1], 'height_ratios': [1,1], 'wspace': 0.3,'hspace':.5})
    plot_time_spec(axs, (0,0), 0, N_test, freqs, auto_FFT, FFTs_unet, FFTs_vit, clist, False)
    plot_mean(axs, (0,1), 0, N_test, lag, auto_mean, mean_unet, mean_vit, clist)
    plot_time_spec(axs, (1,0), 1, N_test, freqs, auto_FFT, FFTs_unet, FFTs_vit, clist)
    plot_mean(axs, (1,1), 2, N_test, lag, auto_mean, mean_unet, mean_vit, clist)

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
            region_title+= i
    region_title = str(region_title)

    fig.suptitle('Long-Time Statistics ' +region_title, fontsize=16)

    plt.savefig(Path(output_dir) / ("Long_Time_Comp_Boundary_"+region+"_"+save_str+".png"),bbox_inches='tight')
    plt.clf()

def plot_short_time_stats(region, save_str, output_dir, N_test, lag, auto_ACC, ACC_unet, ACC_vit,\
                          auto_rmse, rmse_unet, rmse_vit, auto_KE, KE_unet, KE_vit,\
                          auto_corrs, corrs_unet, corrs_vit):
    N = 5
    plt.clf()
    plt.style.use('bmh')

    clist_1 = [cmocean.cm.thermal(i/(N-.5)) for i in range(1,N)]
    clist_2 = ['#d7191c','#abd9e9','#2c7bb6','#fdae61']
    clist_3 = ["#91B59A","#D6A922","#1E88E5","#A00B41"]
    clist = clist_3

    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=20)     # fontsize of the axes title
    plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    plt.rc('legend', fontsize=10)    # legend fontsize
    plt.rc('figure', titlesize=18)


    fig, axs = plt.subplots(2, 2, figsize=(11,6),
                            gridspec_kw={'width_ratios': [1,1], 'height_ratios': [1,1], 'wspace': 0.4,'hspace':.5})
    # plot_acc((0,0), 2, N_test, lag, auto_ACC, ACC_unet, ACC_vit, clist)
    # plot_corr((0,1), 1, N_test, lag, auto_corrs, corrs_unet, corrs_vit, clist)
    # plot_rmse((1,0), 2, N_test, lag, auto_rmse, rmse_unet, rmse_vit, clist)
    # plot_KE((1,1), N_test, lag, auto_KE, KE_unet, KE_vit, clist)

    plot_acc(axs, (0,0), 0, N_test, lag, auto_ACC, ACC_unet, ACC_vit, clist)
    plot_acc(axs, (0,1), 1, N_test, lag, auto_ACC, ACC_unet, ACC_vit, clist)
    plot_rmse(axs, (1,0), 0, N_test, lag, auto_rmse, rmse_unet, rmse_vit, clist)
    plot_rmse(axs, (1,1), 1, N_test, lag, auto_rmse, rmse_unet, rmse_vit, clist)

    fig.suptitle('Short-Time Statistics ' +region, fontsize=16)

    plt.savefig(Path(output_dir) / ("Short_Time_Comp_Boundary_"+region+"_"+save_str+".png"),bbox_inches='tight')
    plt.clf()
    

def plot_all_metrics(region, save_str, output_dir, lag, steps, KE_spec_true, KE_spec_unet, KE_spec_vit,\
                KE_true, KE_unet, KE_vit, enst_spec_true, enst_spec_unet, enst_spec_vit, corr_T_true, corr_T_unet, corr_T_vit,\
                enst_true, enst_unet, enst_vit, RMSE_T_true, RMSE_T_unet, RMSE_T_vit,\
                ACC_T_true, ACC_T_unet, ACC_T_vit):
    N = 5
    N_plot = 200
    plt.clf()
    plt.style.use('bmh')

    clist_1 = [cmocean.cm.thermal(i/(N-.5)) for i in range(1,N)]
    clist_2 = ['#d7191c','#abd9e9','#2c7bb6','#fdae61']
    clist_3 = ["#91B59A","#D6A922","#1E88E5","#A00B41"]
    clist_5 = ["#A00B41","#00DCDE","#A6BD00","#3300EA"]
    clist_6 = ["#A00B41","#DE7400","#00BD8E","#3300EA"]
    clist = clist_5
    
    # KE Spectrum
    plt.loglog(KE_spec_vit.freq_r,KE_spec_vit,c=clist[0],label = f"VIT ~ $\Delta t = {lag},~ N = {steps}$")
    plt.loglog(KE_spec_unet.freq_r,KE_spec_unet,c=clist[3],label = f"UNET ~ $\Delta t = {lag},~ N = {steps}$")

    plt.loglog(KE_spec_true.freq_r,KE_spec_true,"--k")

    plt.xlabel("Wave number (1/km)")
    plt.ylabel("Kinetic Energy")

    plt.legend(loc= "lower left")
    plt.savefig(Path(output_dir) / ("KE_spectrum"+region+"_"+save_str+".png"),bbox_inches='tight')
    plt.clf()

    # KE 
    rho = 1020

    plt.plot(np.arange(1,N_plot+1),KE_vit*rho,c=clist[0],label = f"VIT ~ $\Delta t = {lag},~ N = {steps}$")
    plt.plot(np.arange(1,N_plot+1),KE_unet*rho,c=clist[3],label = f"UNET ~ $\Delta t = {lag},~ N = {steps}$")

    plt.plot(np.arange(1,N_plot+1),KE_true*rho,"--k")
    plt.xlabel("time (days)")
    plt.ylabel("Kinetic Energy")
    plt.legend(loc= "lower left")
    plt.savefig(Path(output_dir) / ("KE"+region+"_"+save_str+".png"),bbox_inches='tight')
    plt.clf()

    # Enstrophy Spectrum
    plt.loglog(enst_spec_vit.freq_r,enst_spec_vit,c=clist[0],label = f"VIT ~ $\Delta t = {lag},~ N = {steps}$")
    plt.loglog(enst_spec_unet.freq_r,enst_spec_unet,c=clist[3],label = f"UNET ~ $\Delta t = {lag},~ N = {steps}$")

    plt.loglog(enst_spec_true.freq_r,enst_spec_true,"--k")
    plt.xlabel("Wave number (1/km)")
    plt.ylabel("Enstrophy")
    plt.legend(loc= "lower left")
    plt.savefig(Path(output_dir) / ("Enstrophy_Spectrum"+region+"_"+save_str+".png"),bbox_inches='tight')
    plt.clf()
    
    # Enstrophy 
    plt.plot(np.arange(1,N_plot+1),enst_vit,c=clist[0],label = f"VIT ~ $\Delta t = {lag},~ N = {steps}$")
    plt.plot(np.arange(1,N_plot+1),enst_unet,c=clist[3],label = f"UNET ~ $\Delta t = {lag},~ N = {steps}$")

    plt.plot(np.arange(1,N_plot+1),enst_true,"--k")
    plt.xlabel("time (days)")
    plt.ylabel("Enstrophy")
    plt.legend(loc= "lower left")
    plt.savefig(Path(output_dir) / ("Enstrophy"+region+"_"+save_str+".png"),bbox_inches='tight')
    plt.clf()

    # Corr
    N_eval = 200
    plt.plot(np.arange(1,N_eval+1),corr_T_vit,c=clist[0],label = f"VIT ~ $\Delta t = {lag},~ N = {steps}$")
    plt.plot(np.arange(1,N_eval+1),corr_T_unet,c=clist[3],label = f"UNET ~ $\Delta t = {lag},~ N = {steps}$")

    plt.plot(np.arange(1,N_eval+1),corr_T_true,"--k")
    plt.xlabel("time (days)")
    plt.ylabel(r"Correlation $T$")
    plt.ylim([0,1])
    plt.xlim([0,N_eval])

    plt.legend(loc= "lower left")
    plt.savefig(Path(output_dir) / ("Corr"+region+"_"+save_str+".png"),bbox_inches='tight')
    plt.clf()

    # RMSE

    plt.plot(np.arange(1,N_eval+1),RMSE_T_vit,c=clist[0],label = f"VIT ~ $\Delta t = {lag},~ N = {steps}$")
    plt.plot(np.arange(1,N_eval+1),RMSE_T_unet,c=clist[3],label = f"UNET ~ $\Delta t = {lag},~ N = {steps}$")

    plt.plot(np.arange(1,N_eval+1),RMSE_T_true,"--k")
    plt.xlabel("time (days)")
    plt.ylabel(r"RMSE $T$")
    plt.xlim([0,N_eval])

    plt.legend()
    plt.savefig(Path(output_dir) / ("RMSE"+region+"_"+save_str+".png"),bbox_inches='tight')
    plt.clf()

    # ACC
    N_eval = 100
    plt.plot(np.arange(1,N_eval+1),ACC_T_vit,c=clist[0],label = f"VIT ~ $\Delta t = {lag},~ N = {steps}$")
    plt.plot(np.arange(1,N_eval+1),ACC_T_unet,c=clist[3],label = f"UNET ~ $\Delta t = {lag},~ N = {steps}$")

    plt.plot(np.arange(1,N_eval+1),ACC_T_true,"--k")
    plt.xlabel("time (days)")
    plt.ylabel(r"ACC $T$")
    plt.ylim([0,1])
    plt.xlim([0,N_eval])

    plt.legend(loc= "lower left")
    plt.savefig(Path(output_dir) / ("ACC"+region+"_"+save_str+".png"),bbox_inches='tight')
    plt.clf()

