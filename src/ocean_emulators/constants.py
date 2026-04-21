import dataclasses
import enum
import logging
from typing import Literal, Self

logger = logging.getLogger(__name__)

import torch
import xarray as xr
from jaxtyping import Bool, Float
from torch import Tensor

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

DatasetType = Literal["om4", "llc"]

PrognosticVarNames = list[str]
BoundaryVarNames = list[str]


@dataclasses.dataclass(frozen=True)
class DatasetSpatialSubset:
    face: int
    i_start: int
    i_end: int
    j_start: int
    j_end: int


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    type: DatasetType
    depth_levels: tuple[float, ...]
    depth_thickness: tuple[float, ...]
    mask_vars: tuple[str, ...]
    mask_all_levels_var: str
    seconds_per_time_step: int
    prognostic_var_names: PrognosticVarNames
    boundary_var_names: BoundaryVarNames
    default_metadata: dict[str, dict[str, str]]
    ocean_heat_temperature_var: str
    surface_heat_flux_var: str
    spatial_subset: DatasetSpatialSubset | None = None

    def __post_init__(self) -> None:
        if len(self.depth_levels) != len(self.depth_thickness):
            raise ValueError(
                "depth_levels and depth_thickness must have the same length"
            )
        if len(self.depth_levels) != len(self.mask_vars):
            raise ValueError("depth_levels and mask_vars must have the same length")

    @property
    def depth_i_levels(self) -> tuple[str, ...]:
        return tuple(str(i) for i in range(len(self.depth_levels)))

    @property
    def num_prognostic_depth_levels(self) -> int:
        depth_indices = []
        for var_name in self.prognostic_var_names:
            _, sep, suffix = var_name.rpartition("_")
            if sep and suffix.isdigit():
                depth_indices.append(int(suffix))
        return max(depth_indices) + 1 if depth_indices else 1


# Experiment prognostic and boundary variables
# Assumption that all 3D variables are appended with depth_i_levels
# and all 2D variables do not have any digits / underscores in their names

RHO_0 = 1035.0  # DENSITY_OF_WATER_CM4 kg/m^3
CP_SW = 3992.0  # SPECIFIC_HEAT_OF_WATER_CM4 J/kg/K


def _select_var_names(
    vars_by_key: dict[str, list[str]],
    key: str,
    dataset_type: DatasetType,
    var_kind: str,
) -> list[str]:
    try:
        return list(vars_by_key[key])
    except KeyError as exc:
        choices = ", ".join(sorted(vars_by_key))
        raise ValueError(
            f"Unsupported {dataset_type} {var_kind} variable key {key!r}. "
            f"Expected one of: {choices}"
        ) from exc


def build_om4_spec(
    prognostic_vars_key: str = "thermo_dynamic_all",
    boundary_vars_key: str = "tau_hfds",
) -> DatasetSpec:
    return DatasetSpec(
        type="om4",
        depth_levels=(
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
        ),
        depth_thickness=(
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
        ),
        mask_vars=tuple(f"mask_{i}" for i in range(19)),
        mask_all_levels_var="wetmask",
        seconds_per_time_step=5 * 24 * 60 * 60,
        prognostic_var_names=_select_var_names(
            {
                "thetao_1": ["thetao_0"],
                "thermo_dynamic_5": [
                    k + str(j)
                    for k in ["uo_", "vo_", "thetao_", "so_"]
                    for j in range(5)
                ]
                + ["zos"],
                "thermo_dynamic_all": [
                    k + str(j)
                    for k in ["uo_", "vo_", "thetao_", "so_"]
                    for j in range(19)
                ]
                + ["zos"],
                "thermo_all": [
                    k + str(j) for k in ["thetao_", "so_"] for j in range(19)
                ]
                + ["zos"],
            },
            prognostic_vars_key,
            "om4",
            "prognostic",
        ),
        boundary_var_names=_select_var_names(
            {
                "hfds": ["hfds"],
                "tau_hfds": ["tauuo", "tauvo", "hfds"],
                "tau_hfds_hfds_anom": ["tauuo", "tauvo", "hfds", "hfds_anomalies"],
            },
            boundary_vars_key,
            "om4",
            "boundary",
        ),
        default_metadata={
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
        },
        ocean_heat_temperature_var="thetao",
        surface_heat_flux_var="hfds",
    )


def build_llc_spec(
    prognostic_vars_key: str = "single_1",
    boundary_vars_key: str = "single_1",
    face: int = 1,
    i_start: int = 0,
    i_end: int = 720,
    j_start: int = 0,
    j_end: int = 720,
) -> DatasetSpec:
    return DatasetSpec(
        type="llc",
        depth_levels=(
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
            77.965,
            88.175,
            99.37,
            111.6,
            124.915,
            139.365,
            154.99,
            171.825,
            189.9,
            209.235,
            229.855,
            251.77,
            274.985,
            299.505,
            325.32,
            352.42,
            380.79,
            410.41,
            441.255,
            473.305,
            506.54,
            540.935,
            576.465,
            613.11,
            650.855,
            689.685,
            729.595,
            770.585,
            812.66,
            855.835,
            900.135,
            945.595,
        ),
        depth_thickness=(
            1.07,
            1.22,
            1.395,
            1.595,
            1.815,
            2.065,
            2.35,
            2.67,
            3.025,
            3.42,
            3.865,
            4.355,
            4.895,
            5.485,
            6.13,
            6.835,
            7.59,
            8.405,
            9.28,
            10.21,
            11.195,
            12.23,
            13.315,
            14.45,
            15.625,
            16.835,
            18.075,
            19.335,
            20.62,
            21.915,
            23.215,
            24.52,
            25.815,
            27.1,
            28.37,
            29.62,
            30.845,
            32.05,
            33.235,
            34.395,
            35.53,
            36.645,
            37.745,
            38.83,
            39.91,
            40.99,
            42.075,
            43.175,
            44.3,
            45.46,
            54.405,
        ),
        mask_vars=tuple(f"wetmask_{i}" for i in range(51)),
        mask_all_levels_var="wetmask",
        seconds_per_time_step=60,
        prognostic_var_names=_select_var_names(
            {
                "single_1": ["Theta_0"],
                "all": [
                    k + str(j)
                    for k in ["U_", "V_", "Theta_", "Salt_"]
                    for j in range(51)
                ]
                + ["Eta"],
            },
            prognostic_vars_key,
            "llc",
            "prognostic",
        ),
        boundary_var_names=_select_var_names(
            {
                "single_1": ["oceQnet"],
                "all": ["oceTAUX", "oceTAUY", "oceQnet", "Eta"],
            },
            boundary_vars_key,
            "llc",
            "boundary",
        ),
        default_metadata={
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
        },
        ocean_heat_temperature_var="Theta",
        surface_heat_flux_var="oceQnet",
        spatial_subset=DatasetSpatialSubset(
            face=face,
            i_start=i_start,
            i_end=i_end,
            j_start=j_start,
            j_end=j_end,
        ),
    )


def construct_metadata(
    data: xr.Dataset,
    dataset_spec: DatasetSpec,
) -> dict[str, dict[str, str]]:
    metadata = {}
    for var in data.variables:
        try:
            metadata[str(var)] = {
                "long_name": data[var].long_name,
                "units": data[var].units,
            }
        except AttributeError:
            if var in dataset_spec.default_metadata.keys():
                metadata[str(var)] = dataset_spec.default_metadata[str(var)]
            elif (
                key := str(var).split("_")[0]
            ) in dataset_spec.default_metadata.keys():
                metadata[str(var)] = dataset_spec.default_metadata[key]
            else:
                logger.info(f"{var} does not have any default metadata")
                metadata[str(var)] = {
                    "long_name": "Unknown",
                    "units": "Unknown",
                }

    return metadata


class LoaderVersion(enum.Enum):
    OM4_TORCH = "om4-torch"


class TensorMap:
    def __init__(
        self,
        dataset_spec: DatasetSpec,
    ):
        """
        Maps input variables / depth levels to their indices in the input tensor.

        VAR_3D_IDX maps the input variables to their indices in the input tensor
        DP_3D_IDX maps the depth levels to their indices in the input tensor
        """
        self.dataset_spec = dataset_spec
        self.VAR_3D_IDX: dict[str, torch.Tensor] = {}
        self.DP_3D_IDX: dict[str, torch.Tensor] = {}

        self.INPT_BOUNDARY_IDX: dict[str, torch.Tensor] = {}
        self.VAR_SET_2D = []
        self.VAR_SET_3D = []
        self.prognostic_var_names = dataset_spec.prognostic_var_names
        self.boundary_var_names = dataset_spec.boundary_var_names
        for out in self.prognostic_var_names:
            var_split = out.split("_")
            if len(var_split) == 1:
                self.VAR_SET_2D.append(var_split[0])
            else:
                self.VAR_SET_3D.append(var_split[0])

        # Consistent order of variables
        self.VAR_SET = list(
            dict.fromkeys([out.split("_")[0] for out in self.prognostic_var_names])
        )

        levels = dataset_spec.num_prognostic_depth_levels

        self.DEPTH_SET = list(dataset_spec.depth_i_levels[:levels])
        self.dz = torch.tensor(dataset_spec.depth_thickness[:levels])

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
        ).to(torch.int32)

    def _populate_boundary_idx(self):
        """
        Populates the indices of the boundary variables in the input tensor.

        We assume the indices INPT_BOUNDARY_IDX will be used after the boundary
        condition is extracted from the input tensor
        """
        for i, k in enumerate(self.boundary_var_names):
            self.INPT_BOUNDARY_IDX[k] = torch.tensor([i])

    def to(self, device: torch.device) -> Self:
        """Move all index tensors to the given device.

        Call this once after model initialization so that indexing a GPU tensor
        with these indices stays on-device and avoids implicit CUDA syncs.
        """
        for k in self.VAR_3D_IDX:
            self.VAR_3D_IDX[k] = self.VAR_3D_IDX[k].to(device)
        for k in self.DP_3D_IDX:
            self.DP_3D_IDX[k] = self.DP_3D_IDX[k].to(device)
        for k in self.INPT_BOUNDARY_IDX:
            self.INPT_BOUNDARY_IDX[k] = self.INPT_BOUNDARY_IDX[k].to(device)
        self.dz = self.dz.to(device)
        return self
