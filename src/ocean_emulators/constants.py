import enum
import logging

logger = logging.getLogger(__name__)

import torch
import xarray as xr
from jaxtyping import Bool, Float
from torch import Tensor

from ocean_emulators.utils.multiton import Multiton

# Common Type Aliases
# See "Existing jaxtyping annotations" section of
#  https://docs.kidger.site/jaxtyping/api/array/#array

Lat = Float[Tensor, "lat"]
Lon = Float[Tensor, "lon"]
Grid = Float[Tensor, "lat lon"]
GridSize = tuple[int, int]
Prognostic = Float[
    Grid, "*batch prognostic_vars"
]  # equivalent to "*batch prognostic_vars lat lon"
Boundary = Float[Grid, "*batch boundary_vars"]
# A note from jaxtyping (why we can't do "prognostic_vars+boundary_vars"):
#   In practice you should usually only use symbolic axes in annotations
#   for return types, referring only to axes annotated for arguments.
# So, we'll leave this default and use symbolic axes locally.
type Input = Float[Grid, "*batch total_vars"]

Example = tuple[Input, Prognostic]

GridMask = Bool[Tensor, "lat lon"]
PrognosticMask = Bool[GridMask, "prognostic_vars"]

SingleChannelVar = Float[Tensor, "batch time lat lon"]
DictSingleChannelVar = dict[str, SingleChannelVar]
SinglePrognosticTimeSeries = Float[Grid, "*batch time"]

SingleTimeSeriesOutput = Float[Tensor, "batch=1 time prognostic_vars lat lon"]
BatchTimeSeriesOutput = Float[Tensor, "batch time=(hist+1) prognostic_vars lat lon"]
HistBatched = Float[Tensor, "batch_hist prognostic_vars lat lon"]
HistChanneled = Float[Tensor, "batch hist_prognostic_vars lat lon"]


MAX_TRAIN_MODEL_STEPS_FORWARD = 200

# Experiment prognostic and boundary variables
# Assumption that all 3D variables are appended with depth_i_levels
# and all 2D variables do not have any digits / underscores in their names

# These represent depth centers
DEPTH_LEVELS = [
    0.5,
    1.57,
    2.79,
    4.185,
    5.78,
    7.595,
    9.66,
    12.01,
    14.68,
    17.705,
    21.125,
    24.99,
    29.345,
    34.24,
    39.725,
    45.855,
    52.69,
    60.28,
    68.685,
] # should go deeper --------------------------------------------------------

NEXT_DEPTH_LEVEL = 77.965

# Depth thicknesses
DEPTH_THICKNESS = [n - p for p, n in zip(DEPTH_LEVELS, DEPTH_LEVELS[1:] + [NEXT_DEPTH_LEVEL])]

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
    "wetmask_0",
    "wetmask_1",
    "wetmask_2",
    "wetmask_3",
    "wetmask_4",
    "wetmask_5",
    "wetmask_6",
    "wetmask_7",
    "wetmask_8",
    "wetmask_9",
    "wetmask_10",
    "wetmask_11",
    "wetmask_12",
    "wetmask_13",
    "wetmask_14",
    "wetmask_15",
    "wetmask_16",
    "wetmask_17",
    "wetmask_18",
]

MASK_ALL_LEVELS_VAR = "wetmask"
HEAT_VAR_NAME = "Theta"

RHO_0 = 1035.0  # DENSITY_OF_WATER_CM4 kg/m^3
CP_SW = 3992.0  # SPECIFIC_HEAT_OF_WATER_CM4 J/kg/K
SECONDS_PER_TIME_STEP = 60 # hourly

PrognosticVarNames = list[str]
PROGNOSTIC_VARS: dict[str, PrognosticVarNames] = {
    "single_1": [f"Theta_{DEPTH_I_LEVELS[0]}"],
    "all": [
        k + str(j) for k in ["U_", "V_", "Theta_", "Salt_"] for j in DEPTH_I_LEVELS
    ]
    + ["Eta"],
}

BoundaryVarNames = list[str]
BOUNDARY_VARS: dict[str, BoundaryVarNames] = {
    "single": ["oceQnet"],
    "all": ["oceTAUX", "oceTAUY", "oceQnet", "Eta"],
}

DEFAULT_METADATA = {
    "Theta": {
        "long_name": "Sea Water Potential Temperature",
        "units": r"\degree C",
    },
    "Salt": {
        "long_name": "Sea Water Salinity",
        "units": "psu",
    },
    "U": {
        "long_name": "Sea Water X Velocity",
        "units": "m/s",
    },
    "V": {
        "long_name": "Sea Water Y Velocity",
        "units": "m/s",
    },
    "Eta": {
        "long_name": "Sea surface height above geoid",
        "units": "m",
    },
    "oceTAUX": {
        "long_name": "Surface Downward X Stress",
        "units": "N/m^2",
    },
    "oceTAUY": {
        "long_name": "Surface Downward Y Stress",
        "units": "N/m^2",
    },
    "oceQnet": {
        "long_name": "Surface ocean heat flux from "
        "SW+LW+latent+sensible+masstransfer+frazil+seaice_melt_heat",
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

        assert 19 == len(DEPTH_I_LEVELS) == len(DEPTH_THICKNESS) == len(DEPTH_LEVELS) == len(MASK_VARS)

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
                torch.tensor([self.VAR_3D_IDX[var_2D] for var_2D in self.VAR_SET_2D]).to(torch.int32),
            ]
        ).to(torch.int32)

        print("DP_3D_IDX")
        print(self.DP_3D_IDX)
        print("VAR_3D_IDX")
        print(self.VAR_3D_IDX)

    def _populate_boundary_idx(self):
        """
        Populates the indices of the boundary variables in the input tensor.

        We assume the indices INPT_BOUNDARY_IDX will be used after the boundary
        condition is extracted from the input tensor
        """
        for i, k in enumerate(self.boundary_var_names):
            self.INPT_BOUNDARY_IDX[k] = torch.tensor([i])
