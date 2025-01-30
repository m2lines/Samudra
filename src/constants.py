import numpy as np
import torch

# Experiment inputs and outputs
DEPTH_LEVELS = [
    "2_5",
    "10_0",
    "22_5",
    "40_0",
    "65_0",
    "105_0",
    "165_0",
    "250_0",
    "375_0",
    "550_0",
    "775_0",
    "1050_0",
    "1400_0",
    "1850_0",
    "2400_0",
    "3100_0",
    "4000_0",
    "5000_0",
    "6000_0",
]

DEPTH_I_LEVELS = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18"
]

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
    "2D": [
        k + DEPTH_I_LEVELS[0] for k in ["uo_", "vo_", "thetao_", "so_"]
    ]
    + ["zos"],
    "3D_5": [
        k + str(j)
        for k in ["uo_", "vo_", "thetao_", "so_"]
        for j in DEPTH_I_LEVELS[:5]
    ]
    + ["zos"],
    "3D_all": [
        k + str(j)
        for k in ["uo_", "vo_", "thetao_", "so_"]
        for j in DEPTH_I_LEVELS
    ]
    + ["zos"],
    "3D_noFast_all": [
        k + str(j) for k in ["thetao_", "so_"] for j in DEPTH_I_LEVELS
    ]
    + ["zos"],
    "3D_TS_all": [k + str(j) for k in ["thetao_", "so_"] for j in DEPTH_I_LEVELS],
    "3D_onlyTemp_all": [k + str(j) for k in ["thetao_"] for j in DEPTH_I_LEVELS],
    "3D_SST_all": [
        k + str(j)
        for k in ["uo_", "vo_", "thetao_", "so_"]
        for j in DEPTH_I_LEVELS
        if not (k == "thetao_" and j == DEPTH_I_LEVELS[0])
    ]
    + ["zos"],
}
EXTRA_VARS = {
    "1": ["ur", "vr"],
    "2": ["ur", "vr", "Tm"],
    "3": ["Tm"],
    "4": ["ur", "vr", "Tm", "Tr"],
    "NoBoundary": [],
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
    "3D_noFast_all": [k + str(j) for k in ["uo_", "vo_"] for j in DEPTH_I_LEVELS]
    + ["tauuo", "tauvo", "hfds"],
    "3D_5": ["tauuo", "tauvo", "hfds"],
    "3D_all": ["tauuo", "tauvo", "hfds"],
    "3D_all_hfds_anom": ["tauuo", "tauvo", "hfds", "hfds_anomalies"],
    "3D_all_hfds_anom_cuminteg": [
        "tauuo",
        "tauvo",
        "hfds",
        "hfds_anomalies",
        "cum_integrated_heat",
    ],
    "3D_all_onlyhfds_cuminteg": ["tauuo", "tauvo", "cum_integrated_heat"],
    "3D_all_SAT_tos": ["tauuo", "tauvo", "DSWRFtoa", "air_temperature_at_two_meters"],
    "3D_all_SAT": ["tauuo", "tauvo", "air_temperature_at_two_meters"],
}
OUT_VARS = {
    "1": ["um", "vm"],
    "2": ["um", "vm", "Tm"],
    "3": ["ur", "vr"],
    "4": ["ur", "vr", "Tr"],
    "5": ["u", "v"],
    "6": ["u", "v", "T"],
    "3D": ["uo", "vo", "thetao", "so", "zos"],
    "2D": [
        k + DEPTH_I_LEVELS[0] for k in ["uo_", "vo_", "thetao_", "so_"]
    ]
    + ["zos"],
    "3D_5": [
        k + str(j)
        for k in ["uo_", "vo_", "thetao_", "so_"]
        for j in DEPTH_I_LEVELS[:5]
    ]
    + ["zos"],
    "3D_noFast_5": [
        k + str(j) for k in ["thetao_", "so_"] for j in DEPTH_I_LEVELS[:5]
    ]
    + ["zos"],
    "3D_all": [
        k + str(j)
        for k in ["uo_", "vo_", "thetao_", "so_"]
        for j in DEPTH_I_LEVELS
    ]
    + ["zos"],
    "3D_noFast_all": [
        k + str(j) for k in ["thetao_", "so_"] for j in DEPTH_I_LEVELS
    ]
    + ["zos"],
    "3D_onlyTemp_all": [k + str(j) for k in ["thetao_"] for j in DEPTH_I_LEVELS],
    "3D_onlyFast_all": [
        k + str(j) for k in ["uo_", "vo_"] for j in DEPTH_I_LEVELS
    ]
    + ["zos"],
    "3D_TS_all": [k + str(j) for k in ["thetao_", "so_"] for j in DEPTH_I_LEVELS],
    "3D_SST_all": [
        k + str(j)
        for k in ["uo_", "vo_", "thetao_", "so_"]
        for j in DEPTH_I_LEVELS
        if not (k == "thetao_" and j == DEPTH_I_LEVELS[0])
    ]
    + ["zos"],
}


def get_eval_maps(exp_num):
    # CH_3D_IDX maps the input variables to their indices in the input tensor
    # DP_3D_IDX maps the depth levels to their indices in the input tensor
    CH_3D_IDX = {}
    VAR_SET = list(dict.fromkeys(([out.split("_")[0] for out in OUT_VARS[exp_num]])))
    # assert VAR_SET[-3] == 'thetao' and VAR_SET[-2] == 'so' and VAR_SET[-1] == 'zos'
    DEPTH_SET = DEPTH_I_LEVELS
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
            elif d == k.split("_")[-1]:
                DP_3D_IDX[d] = torch.cat([DP_3D_IDX[d], torch.tensor([i])])
        DP_3D_IDX[d] = DP_3D_IDX[d].to(torch.int32)
    if "zos" in VAR_SET:
        DP_3D_IDX[DEPTH_I_LEVELS[0]] = torch.cat(
            [DP_3D_IDX[DEPTH_I_LEVELS[0]], torch.tensor([len(OUT_VARS[exp_num]) - 1])]
        )  # zos
    DP_3D_IDX[DEPTH_I_LEVELS[0]] = DP_3D_IDX[DEPTH_I_LEVELS[0]].to(torch.int32)
    return CH_3D_IDX, DP_3D_IDX, VAR_SET, DEPTH_SET

