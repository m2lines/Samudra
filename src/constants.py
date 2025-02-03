from typing import Dict, Optional

import torch
import xarray as xr

# Experiment inputs and outputs
DEPTH_LEVELS = [
    2.5,
    10.0,
    22.5,
    40.0,
    65.0,
    105.0,
    165.0,
    250.0,
    375.0,
    550.0,
    775.0,
    1050.0,
    1400.0,
    1850.0,
    2400.0,
    3100.0,
    4000.0,
    5000.0,
    6000.0,
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
    "18",
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
    "2D": [k + DEPTH_I_LEVELS[0] for k in ["uo_", "vo_", "thetao_", "so_"]] + ["zos"],
    "3D_5": [
        k + str(j) for k in ["uo_", "vo_", "thetao_", "so_"] for j in DEPTH_I_LEVELS[:5]
    ]
    + ["zos"],
    "3D_all": [
        k + str(j) for k in ["uo_", "vo_", "thetao_", "so_"] for j in DEPTH_I_LEVELS
    ]
    + ["zos"],
    "3D_noFast_all": [k + str(j) for k in ["thetao_", "so_"] for j in DEPTH_I_LEVELS]
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
    "2D": [k + DEPTH_I_LEVELS[0] for k in ["uo_", "vo_", "thetao_", "so_"]] + ["zos"],
    "3D_5": [
        k + str(j) for k in ["uo_", "vo_", "thetao_", "so_"] for j in DEPTH_I_LEVELS[:5]
    ]
    + ["zos"],
    "3D_noFast_5": [k + str(j) for k in ["thetao_", "so_"] for j in DEPTH_I_LEVELS[:5]]
    + ["zos"],
    "3D_all": [
        k + str(j) for k in ["uo_", "vo_", "thetao_", "so_"] for j in DEPTH_I_LEVELS
    ]
    + ["zos"],
    "3D_noFast_all": [k + str(j) for k in ["thetao_", "so_"] for j in DEPTH_I_LEVELS]
    + ["zos"],
    "3D_onlyTemp_all": [k + str(j) for k in ["thetao_"] for j in DEPTH_I_LEVELS],
    "3D_onlyFast_all": [k + str(j) for k in ["uo_", "vo_"] for j in DEPTH_I_LEVELS]
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


def construct_metadata(data: xr.Dataset) -> Dict[str, Dict[str, str]]:
    metadata = {}
    for var in data.variables:
        try:
            metadata[var] = {
                "long_name": data[var].long_name,
                "units": data[var].units,
            }
        except AttributeError:
            print(f"{var} has no long_name or units attribute")
            metadata[var] = {
                "long_name": "Unknown",
                "units": "Unknown",
            }
    return metadata


class TensorMap:
    _instance: Optional["TensorMap"] = None

    def __new__(cls, *args, **kwargs) -> "TensorMap":
        # Prevent direct instantiation
        raise TypeError(
            "TensorMap cannot be instantiated directly. Use init_instance() instead."
        )

    @classmethod
    def get_instance(cls) -> "TensorMap":
        if cls._instance is None:
            raise ValueError("TensorMap not initialized")
        return cls._instance

    @classmethod
    def init_instance(cls, exp_num: str) -> "TensorMap":
        if cls._instance is not None:
            raise ValueError("TensorMap already initialized")

        instance = super().__new__(cls)
        instance._initialize(exp_num)
        cls._instance = instance
        return cls._instance

    def _initialize(self, exp_num: str):
        """
        Maps input variables / depth levels to their indices in the input tensor.

        VAR_3D_IDX maps the input variables to their indices in the input tensor
        DP_3D_IDX maps the depth levels to their indices in the input tensor
        """
        self.exp_num = exp_num
        self.VAR_3D_IDX: Dict[str, torch.Tensor] = {}
        self.DP_3D_IDX: Dict[str, torch.Tensor] = {}
        self.VAR_SET = list(
            dict.fromkeys(([out.split("_")[0] for out in OUT_VARS[exp_num]]))
        )
        self.DEPTH_SET = DEPTH_I_LEVELS
        self.outputs = OUT_VARS[exp_num]

        self._populate_var_3d_idx()
        self._populate_dp_3d_idx()

    def _populate_var_3d_idx(self):
        for kt in self.VAR_SET:
            self.VAR_3D_IDX[kt] = torch.tensor([])
            for i, k in enumerate(self.outputs):
                if kt in k:
                    self.VAR_3D_IDX[kt] = torch.cat(
                        [self.VAR_3D_IDX[kt], torch.tensor([i])]
                    )
            self.VAR_3D_IDX[kt] = self.VAR_3D_IDX[kt].to(torch.int32)

    def _populate_dp_3d_idx(self):
        for d in self.DEPTH_SET:
            self.DP_3D_IDX[d] = torch.tensor([])
            for i, k in enumerate(self.outputs):
                if k == "zos":
                    continue
                elif d == k.split("_")[-1]:
                    self.DP_3D_IDX[d] = torch.cat(
                        [self.DP_3D_IDX[d], torch.tensor([i])]
                    )
            self.DP_3D_IDX[d] = self.DP_3D_IDX[d].to(torch.int32)
        if "zos" in self.VAR_SET:
            self.DP_3D_IDX[self.DEPTH_SET[-1]] = torch.cat(
                [
                    self.DP_3D_IDX[self.DEPTH_SET[-1]],
                    torch.tensor([len(self.outputs) - 1]),
                ]
            )  # zos
        self.DP_3D_IDX[self.DEPTH_SET[0]] = self.DP_3D_IDX[self.DEPTH_SET[0]].to(
            torch.int32
        )
