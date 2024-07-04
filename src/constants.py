import numpy as np
import torch

# Experiment inputs and outputs
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
    "3D_all": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in range(19)
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
    "3D_all": ["tauuo", "tauvo", "hfds"],
}
OUT_VARS = {
    "1": ["um", "vm"],
    "2": ["um", "vm", "Tm"],
    "3": ["ur", "vr"],
    "4": ["ur", "vr", "Tr"],
    "5": ["u", "v"],
    "6": ["u", "v", "T"],
    "3D": ["uo", "vo", "thetao", "so", "zos"],
    "3D_all": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in range(19)
    ]
    + ["zos"],
}

#### SPECIAL CASE : Boundary condition with thetao_lev_0
INPT_VARS['3D_all'].remove('thetao_lev_0')
EXTRA_VARS['3D_all'].append('thetao_lev_0')
OUT_VARS['3D_all'].remove('thetao_lev_0')

CH_3D_IDX = {}
for kt in ['uo', 'vo', 'thetao', 'so']:
    CH_3D_IDX[kt] = torch.tensor([])
    for i, k in enumerate(INPT_VARS['3D_all']):
        if kt in k:
            CH_3D_IDX[kt] = torch.cat([CH_3D_IDX[kt], torch.tensor([i])])

DP_3D_IDX = {}
for d in range(19):
    DP_3D_IDX[d] = torch.tensor([])
    for i, k in enumerate(INPT_VARS['3D_all']):
        if k == 'zos':
            continue
        elif d == int(k.split('lev_')[-1]):
            DP_3D_IDX[d] = torch.cat([DP_3D_IDX[d], torch.tensor([i])])
DP_3D_IDX[0] = torch.cat([DP_3D_IDX[0], torch.tensor([len(INPT_VARS['3D_all'])-1])]) # zos

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
