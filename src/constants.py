import logging
from typing import Dict, Optional

import torch
import xarray as xr

# Experiment inputs and outputs
# Assumption that all 3D variables are appended with depth_i_levels
# and all 2D variables do not have any digits / underscores in their names
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

# Not sure why the data has conflicting depth
# thickness and depth levels
DEPTH_THICKNESS = [
    5.0,
    10.0,
    15.0,
    20.0,
    30.0,
    50.0,
    70.0,
    100.0,
    150.0,
    200.0,
    250.0,
    300.0,
    400.0,
    500.0,
    600.0,
    800.0,
    1000.0,
    1000.0,
    1000.0,
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

MASK_VARS = [
    "mask_0",
    "mask_1",
    "mask_2",
    "mask_3",
    "mask_4",
    "mask_5",
    "mask_6",
    "mask_7",
    "mask_8",
    "mask_9",
    "mask_10",
    "mask_11",
    "mask_12",
    "mask_13",
    "mask_14",
    "mask_15",
    "mask_16",
    "mask_17",
    "mask_18",
]

INPT_VARS: Dict[str, list[str]] = {
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
    "3D_tos_all": ["tos", "zos"]
    + [k + str(j) for k in ["so_", "thetao_", "uo_", "vo_"] for j in DEPTH_I_LEVELS],
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
EXTRA_VARS: Dict[str, list[str]] = {
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
    "3D_mask_all": ["hfds", "tauuo", "tauvo"]
    + [k + str(j) for k in ["mask_"] for j in DEPTH_I_LEVELS],
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
OUT_VARS: Dict[str, list[str]] = {
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
    "3D_tos_all": ["tos", "zos"]
    + [k + str(j) for k in ["so_", "thetao_", "uo_", "vo_"] for j in DEPTH_I_LEVELS],
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
            logging.info(f"{var} has no long_name or units attribute")
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
    def init_instance(cls, exp_num: str, exp_num_extra: str) -> "TensorMap":
        if cls._instance is not None:
            raise ValueError("TensorMap already initialized")

        instance = super().__new__(cls)
        instance._initialize(exp_num, exp_num_extra)
        cls._instance = instance
        return cls._instance

    def _initialize(self, exp_num: str, exp_num_extra: str):
        """
        Maps input variables / depth levels to their indices in the output tensor.
        Also maps the boundary variables to their indices in the input tensor.

        OUT_VAR_3D_IDX maps the output variables to their indices in the output tensor
        OUT_DP_3D_IDX maps the depth levels to their indices in the output tensor
        """
        self.exp_num = exp_num
        self.exp_num_extra = exp_num_extra
        self.OUT_VAR_3D_IDX: Dict[str, torch.Tensor] = {}
        self.OUT_DP_3D_IDX: Dict[str, torch.Tensor] = {}
        self.INPT_BOUNDARY_IDX: Dict[str, torch.Tensor] = {}

        self.OUT_VAR_SET_2D: list[str] = []
        self.OUT_VAR_SET_3D: list[str] = []
        for out in OUT_VARS[exp_num]:
            var_split = out.split("_")
            if len(var_split) == 1:
                self.OUT_VAR_SET_2D.append(var_split[0])
            else:
                self.OUT_VAR_SET_3D.append(var_split[0])

        # Consistent order of variables
        self.VAR_SET = list(
            dict.fromkeys(([out.split("_")[0] for out in OUT_VARS[exp_num]]))
        )
        self.DEPTH_SET: list[str] = DEPTH_I_LEVELS
        self.outputs: list[str] = OUT_VARS[exp_num]
        self.extra: list[str] = EXTRA_VARS[exp_num_extra]

        self.dz = torch.tensor(DEPTH_THICKNESS)

        self._populate_var_3d_idx()
        self._populate_dp_3d_idx()
        self._populate_boundary_idx()

    def _populate_var_3d_idx(self):
        """
        Populates the indices of the output variables in the output tensor.
        """
        for kt in self.VAR_SET:
            self.OUT_VAR_3D_IDX[kt] = torch.tensor([])
            for i, k in enumerate(self.outputs):
                if kt in k:
                    self.OUT_VAR_3D_IDX[kt] = torch.cat(
                        [self.OUT_VAR_3D_IDX[kt], torch.tensor([i])]
                    )
            self.OUT_VAR_3D_IDX[kt] = self.OUT_VAR_3D_IDX[kt].to(torch.int32)

    def _populate_dp_3d_idx(self):
        """
        Populates the indices of the depth levels in the output tensor.
        """
        for d in self.DEPTH_SET:
            self.OUT_DP_3D_IDX[d] = torch.tensor([])
            for i, k in enumerate(self.outputs):
                k_split = k.split("_")
                if len(k_split) == 1:
                    continue
                elif d == k_split[-1]:
                    self.OUT_DP_3D_IDX[d] = torch.cat(
                        [self.OUT_DP_3D_IDX[d], torch.tensor([i])]
                    )
            self.OUT_DP_3D_IDX[d] = self.OUT_DP_3D_IDX[d].to(torch.int32)

        self.OUT_DP_3D_IDX[self.DEPTH_SET[0]] = torch.cat(
            [
                self.OUT_DP_3D_IDX[self.DEPTH_SET[0]],
                torch.tensor(
                    [self.OUT_VAR_3D_IDX[var_2D] for var_2D in self.OUT_VAR_SET_2D]
                ),
            ]
        )

    def _populate_boundary_idx(self):
        """
        Populates the indices of the boundary variables in the input tensor.

        We assume the indices INPT_BOUNDARY_IDX will be used after the boundary
        condition is extracted from the input tensor
        """
        for i, k in enumerate(self.extra):
            self.INPT_BOUNDARY_IDX[k] = torch.tensor([i])
