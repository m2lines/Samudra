import numpy as np
import torch

# Experiment inputs and outputs
DEPTH_LEVELS = ['2_5',
 '10_0',
 '22_5',
 '40_0',
 '65_0',
 '105_0',
 '165_0',
 '250_0',
 '375_0',
 '550_0',
 '775_0',
 '1050_0',
 '1400_0',
 '1850_0',
 '2400_0',
 '3100_0',
 '4000_0',
 '5000_0',
 '6000_0']

INPT_VARS = {
    "1": ["um", "vm"],
    "2": ["um", "vm", "ur", "vr"],
    "3": ["um", "vm", "Tm"],
    "4": ["um", "vm", "ur", "vr", "Tm", "Tr"],
    "5": ["ur", "vr"],
    "6": ["ur", "vr", "Tr"],
    "7": ["Tm"],
    "8": ["Tm", "Tr"],
    "9": ["u", "v"],
    "10": ["u", "v", "T"],
    "11": ["tau_u", "tau_v"],
    "12": ["tau_u", "tau_v", "t_ref"],
    "3D": ["uo", "vo", "thetao", "so", "zos"],
    "2D": [k + DEPTH_LEVELS[0]
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
    ]
    + ["zos"],
    "3D_5": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS[:5]
    ]
    + ["zos"],
    "3D_all": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ]
    + ["zos"],
    "3D_SST_all": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS if not (k == "thetao_lev_" and j == DEPTH_LEVELS[0]) 
    ] 
    + ["zos"],
}
EXTRA_VARS = {
    "1": ["ur", "vr"],
    "2": ["ur", "vr", "Tm"],
    "3": ["Tm"],
    "4": ["ur", "vr", "Tm", "Tr"],
    "5": [],
    "6": ["um", "vm"],
    "7": ["um", "vm", "Tm"],
    "8": ["um", "vm", "Tm", "Tr"],
    "9": ["ur", "vr", "tau_u", "tau_v"],
    "10": ["tau_u", "tau_v"],
    "11": ["t_ref"],
    "12": ["tau_u", "tau_v", "t_ref"],
    "13": ["ur", "vr", "Tr", "tau_u", "tau_v", "t_ref"],
    "3D": ["tauuo", "tauvo", "hfds"],
    "2D": ["tauuo", "tauvo", "hfds"],
    "3D_5": ["tauuo", "tauvo", "hfds"],
    "3D_all": ["tauuo", "tauvo", "hfds"],
    "3D_SST_all": ["tauuo", "tauvo", "hfds", "thetao_lev_0"],
}
OUT_VARS = {
    "1": ["um", "vm"],
    "2": ["um", "vm", "Tm"],
    "3": ["ur", "vr"],
    "4": ["ur", "vr", "Tr"],
    "5": ["u", "v"],
    "6": ["u", "v", "T"],
    "3D": ["uo", "vo", "thetao", "so", "zos"],
    "2D": [k + DEPTH_LEVELS[0]
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
    ]
    + ["zos"],
    "3D_5": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS[:5]
    ]
    + ["zos"],
    "3D_5_noFast": [
        k + str(j)
        for k in ["thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS[:5]
    ]
    + ["zos"],
    "3D_all": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ]
    + ["zos"],
    "3D_SST_all": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS if not (k == "thetao_lev_" and j == DEPTH_LEVELS[0]) 
    ] 
    + ["zos"],
}


def get_eval_maps(exp_num):
    # CH_3D_IDX maps the input variables to their indices in the input tensor
    # DP_3D_IDX maps the depth levels to their indices in the input tensor
    CH_3D_IDX = {}
    VAR_SET = list(dict.fromkeys(([out.split('_')[0] for out in OUT_VARS[exp_num]])))
    assert VAR_SET[-3] == 'thetao' and VAR_SET[-2] == 'so' and VAR_SET[-1] == 'zos'
    DEPTH_SET = list(dict.fromkeys(([out.split('lev_')[-1] for out in OUT_VARS[exp_num]])))
    assert DEPTH_SET[0] == DEPTH_LEVELS[0]
    for kt in VAR_SET:
        CH_3D_IDX[kt] = torch.tensor([])
        for i, k in enumerate(OUT_VARS[exp_num]):
            if kt in k:
                CH_3D_IDX[kt] = torch.cat([CH_3D_IDX[kt], torch.tensor([i])])
        CH_3D_IDX[kt] = CH_3D_IDX[kt].to(torch.int32)

    DP_3D_IDX = {}
    for d in DEPTH_SET:
        DP_3D_IDX[d] = torch.tensor([])
        for i, k in enumerate(OUT_VARS[exp_num]):
            if k == "zos":
                continue
            elif d == k.split("lev_")[-1]:
                DP_3D_IDX[d] = torch.cat([DP_3D_IDX[d], torch.tensor([i])])
        DP_3D_IDX[d] = DP_3D_IDX[d].to(torch.int32)
    DP_3D_IDX[DEPTH_LEVELS[0]] = torch.cat(
        [DP_3D_IDX[DEPTH_LEVELS[0]], torch.tensor([len(OUT_VARS[exp_num]) - 1])]
    )  # zos
    DP_3D_IDX[DEPTH_LEVELS[0]] = DP_3D_IDX[DEPTH_LEVELS[0]].to(torch.int32)
    return CH_3D_IDX, DP_3D_IDX, VAR_SET, DEPTH_SET


# Region boundaries
REGIONS = {
    "Kuroshio": {"lat": [15, 41], "lon": [-215, -185]},
    "Kuroshio_Ext": {"lat": [5, 50], "lon": [-250, -175]},
    "Gulf_Stream": {"lat": [25, 50], "lon": [-70, -35]},
    "Gulf_Stream_Ext": {"lat": [27, 50], "lon": [-82, -35]},
    "Gulf_Stream_Ext2": {"lat": [26, 50.65], "lon": [-82, -50.25]},
    "Gulf_Stream_Ext3": {"lat": [26, 50.65], "lon": [-82, -34.25]},
    "Tropics": {"lat": [-5, 25], "lon": [-95, -65]},
    "Tropics_Ext": {"lat": [-5, 25], "lon": [-115, -45]},
    "South_America": {"lat": [-60, -30], "lon": [-70, -35]},
    "Africa": {"lat": [-50, -20], "lon": [5, 45]},
    "Africa_Ext": {"lat": [-55, -15], "lon": [-5, 55]},
    "Quiescent": {"lat": [-42.5, -17.5], "lon": [-155, -120]},
    "Quiescent_Ext": {"lat": [-55, -10], "lon": [-170, -110]},
    "Pacific": {"lat": [-35, 35], "lon": [-230, -80]},
    "Indian": {"lat": [-30, 28], "lon": [30, 79]},
}

GLOBAL_COMBINED_STATS = {
    "s_in": np.array(
        [
            1.19912029e-01,
            8.75121945e-02,
            1.11957607e01,
            9.65926101e-05,
            7.35161570e-05,
            2.04991480e01,
        ]
    ),
    "s_out": np.array([0.11991318, 0.08751262, 11.19576553]),
    "m_in": np.array(
        [
            -1.52130831e-03,
            4.28648579e-03,
            8.86227188e00,
            7.05813917e-06,
            2.61937937e-07,
            2.78227831e02,
        ]
    ),
    "m_out": np.array([-1.52113173e-03, 4.28606825e-03, 8.86225711e00]),
}
