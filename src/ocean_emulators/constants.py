import enum
import logging
from typing import TypeAlias, TypeVar

logger = logging.getLogger(__name__)

import torch
import xarray as xr
from jaxtyping import Bool, Float

from ocean_emulators.utils.multiton import Multiton

# Common Type Aliases
# See "Existing jaxtyping annotations" section of
#  https://docs.kidger.site/jaxtyping/api/array/#array

# Our Arrays will be either `torch.Tensor`s or `xarray.DataArray`s.
Array = TypeVar("Array", torch.Tensor, xr.DataArray)

Grid = Float[Array, "lat lon"]
Prognostic = Float[
    Grid, "*batch prognostic_vars"
]  # equivalent to "*batch prognostic_vars lat lon"
Boundary = Float[Grid, "*batch boundary_vars"]
# A note from jaxtyping (why we can't do "prognostic_vars+boundary_vars"):
#   In practice you should usually only use symbolic axes in annotations
#   for return types, referring only to axes annotated for arguments.
# So, we'll leave this default and use symbolic axes locally.
Input: TypeAlias = Float[Grid, "*batch total_vars"]

Example = tuple[Input, Prognostic] | tuple[xr.Dataset, xr.Dataset]

GridMask = Bool[Array, "lat lon"]
PrognosticMask = Bool[GridMask, "prognostic_vars"]

SingleChannelVar = Float[torch.Tensor, "batch time lat lon"]
DictSingleChannelVar = dict[str, SingleChannelVar]
SinglePrognosticTimeSeries = Float[Grid, "*batch time"]

SingleTimeSeriesOutput = Float[torch.Tensor, "batch=1 time prognostic_vars lat lon"]
BatchTimeSeriesOutput = Float[
    torch.Tensor, "batch time=(hist+1) prognostic_vars lat lon"
]
HistBatched = Float[torch.Tensor, "batch_hist prognostic_vars lat lon"]
HistChanneled = Float[torch.Tensor, "batch hist_prognostic_vars lat lon"]


MAX_TRAIN_MODEL_STEPS_FORWARD = 200

# Experiment prognostic and boundary variables
# Assumption that all 3D variables are appended with depth_i_levels
# and all 2D variables do not have any digits / underscores in their names

# These represent depth centers
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

# Depth thicknesses
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

RHO_0 = 1035.0  # DENSITY_OF_WATER_CM4 kg/m^3
CP_SW = 3992.0  # SPECIFIC_HEAT_OF_WATER_CM4 J/kg/K
SECONDS_PER_5DAY = 5 * 24 * 60 * 60  # 5 day average
TIME_DELTA = 5  # Time delta in days

PrognosticVarNames = list[str]
PROGNOSTIC_VARS: dict[str, PrognosticVarNames] = {
    "thetao_1": [f"thetao_{DEPTH_I_LEVELS[0]}"],
    "thermo_dynamic_5": [
        k + str(j) for k in ["uo_", "vo_", "thetao_", "so_"] for j in DEPTH_I_LEVELS[:5]
    ]
    + ["zos"],
    "thermo_dynamic_all": [
        k + str(j) for k in ["uo_", "vo_", "thetao_", "so_"] for j in DEPTH_I_LEVELS
    ]
    + ["zos"],
    "thermo_all": [k + str(j) for k in ["thetao_", "so_"] for j in DEPTH_I_LEVELS]
    + ["zos"],
}
BoundaryVarNames = list[str]
BOUNDARY_VARS: dict[str, BoundaryVarNames] = {
    "hfds": ["hfds"],
    "tau_hfds": ["tauuo", "tauvo", "hfds"],
    "tau_hfds_hfds_anom": ["tauuo", "tauvo", "hfds", "hfds_anomalies"],
}

DEFAULT_METADATA = {
    "thetao": {
        "long_name": "Sea Water Potential Temperature",
        "units": r"\degree C",
    },
    "so": {
        "long_name": "Sea Water Salinity",
        "units": "psu",
    },
    "uo": {
        "long_name": "Sea Water X Velocity",
        "units": "m/s",
    },
    "vo": {
        "long_name": "Sea Water Y Velocity",
        "units": "m/s",
    },
    "zos": {
        "long_name": "Sea surface height above geoid",
        "units": "m",
    },
    "tos": {
        "long_name": "Sea surface temperature",
        "units": r"\degree C",
    },
    "tauuo": {
        "long_name": "Surface Downward X Stress",
        "units": "N/m^2",
    },
    "tauvo": {
        "long_name": "Surface Downward Y Stress",
        "units": "N/m^2",
    },
    "hfds": {
        "long_name": "Surface ocean heat flux from "
        "SW+LW+latent+sensible+masstransfer+frazil+seaice_melt_heat",
        "units": "W/m^2",
    },
    "hfds_anomalies": {
        "long_name": "hfds anomalies",
        "units": "W/m^2",
    },
}


def construct_metadata(data: xr.Dataset) -> dict[str, dict[str, str]]:
    metadata = {}
    for var in data.variables:
        try:
            metadata[str(var)] = {
                "long_name": data[var].long_name,
                "units": data[var].units,
            }
        except AttributeError:
            if var in DEFAULT_METADATA.keys():
                metadata[str(var)] = DEFAULT_METADATA[str(var)]
            elif (key := str(var).split("_")[0]) in DEFAULT_METADATA.keys():
                metadata[str(var)] = DEFAULT_METADATA[key]
            else:
                logger.info(f"{var} does not have any default metadata")
                metadata[str(var)] = {
                    "long_name": "Unknown",
                    "units": "Unknown",
                }

    return metadata


class LoaderVersion(enum.Enum):
    OM4_EAGER = "om4-eager"
    OM4_TORCH = "om4-torch"


# TODO(#95): See if this can be removed and replaced.
class TensorMap(Multiton):
    def _initialize(self, prognostic_vars_key: str, boundary_vars_key: str):
        """
        Maps input variables / depth levels to their indices in the input tensor.

        VAR_3D_IDX maps the input variables to their indices in the input tensor
        DP_3D_IDX maps the depth levels to their indices in the input tensor
        """
        self.prognostic_vars_key = prognostic_vars_key
        self.VAR_3D_IDX: dict[str, torch.Tensor] = {}
        self.DP_3D_IDX: dict[str, torch.Tensor] = {}

        self.INPT_BOUNDARY_IDX: dict[str, torch.Tensor] = {}
        self.VAR_SET_2D = []
        self.VAR_SET_3D = []
        for out in PROGNOSTIC_VARS[prognostic_vars_key]:
            var_split = out.split("_")
            if len(var_split) == 1:
                self.VAR_SET_2D.append(var_split[0])
            else:
                self.VAR_SET_3D.append(var_split[0])

        # Consistent order of variables
        self.VAR_SET = list(
            dict.fromkeys(
                [out.split("_")[0] for out in PROGNOSTIC_VARS[prognostic_vars_key]]
            )
        )

        levels_str = prognostic_vars_key.split("_")[-1]
        if "all" in levels_str:
            levels = 19
        else:
            levels = int(levels_str)

        self.DEPTH_SET = DEPTH_I_LEVELS[:levels]
        self.prognostic_var_names = PROGNOSTIC_VARS[prognostic_vars_key]
        self.boundary_var_names = BOUNDARY_VARS[boundary_vars_key]
        self.dz = torch.tensor(DEPTH_THICKNESS[:levels])

        self._populate_var_3d_idx()
        self._populate_dp_3d_idx()
        self._populate_boundary_idx()

    def _populate_var_3d_idx(self):
        for kt in self.VAR_SET:
            self.VAR_3D_IDX[kt] = torch.tensor([])
            for i, k in enumerate(self.prognostic_var_names):
                if kt in k:
                    self.VAR_3D_IDX[kt] = torch.cat(
                        [self.VAR_3D_IDX[kt], torch.tensor([i])]
                    )
            self.VAR_3D_IDX[kt] = self.VAR_3D_IDX[kt].to(torch.int32)

    def _populate_dp_3d_idx(self):
        for d in self.DEPTH_SET:
            self.DP_3D_IDX[d] = torch.tensor([])
            for i, k in enumerate(self.prognostic_var_names):
                k_split = k.split("_")
                if len(k_split) == 1:
                    continue
                elif d == k_split[-1]:
                    self.DP_3D_IDX[d] = torch.cat(
                        [self.DP_3D_IDX[d], torch.tensor([i])]
                    )
            self.DP_3D_IDX[d] = self.DP_3D_IDX[d].to(torch.int32)

        self.DP_3D_IDX[self.DEPTH_SET[0]] = torch.cat(
            [
                self.DP_3D_IDX[self.DEPTH_SET[0]],
                torch.tensor([self.VAR_3D_IDX[var_2D] for var_2D in self.VAR_SET_2D]),
            ]
        )

    def _populate_boundary_idx(self):
        """
        Populates the indices of the boundary variables in the input tensor.

        We assume the indices INPT_BOUNDARY_IDX will be used after the boundary
        condition is extracted from the input tensor
        """
        for i, k in enumerate(self.boundary_var_names):
            self.INPT_BOUNDARY_IDX[k] = torch.tensor([i])
