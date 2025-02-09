import dataclasses
from typing import Dict, List, Literal, Optional, Union

import matplotlib.pyplot as plt
import torch
import wandb
import xarray as xr

from aggregator.metrics import area_weighted_mean_bias, area_weighted_rmse
from utils.distributed import all_reduce_mean

from ..plotting import get_cmap_limits, plot_imshow


@dataclasses.dataclass
class _TargetGenPair:
    name: str
    target: torch.Tensor
    gen: torch.Tensor
    area_weights: torch.Tensor

    def bias(self):
        return self.gen - self.target

    def rmse(self) -> float:
        ret = float(
            area_weighted_rmse(
                gen=self.gen,
                target=self.target,
                area_weights=self.area_weights,
            )
            .cpu()
            .numpy()
        )
        return ret

    def weighted_mean_bias(self) -> float:
        return float(
            area_weighted_mean_bias(
                gen=self.gen,
                target=self.target,
                area_weights=self.area_weights,
            )
            .cpu()
            .numpy()
        )


def get_gen_shape(gen_data: Dict[str, torch.Tensor]):
    for name in gen_data:
        return gen_data[name].shape


class TimeMeanAggregator:
    _image_captions = {
        "bias_map": "{name} time-mean bias (generated - reference) [{units}]",
        "gen_map": "{name} time-mean generated [{units}]",
    }

    def __init__(
        self,
        area_weights: torch.Tensor,
        target: Literal["norm", "denorm"] = "denorm",
        metadata: Optional[Dict[str, Dict[str, str]]] = None,
        reference_means: Optional[xr.Dataset] = None,
    ):
        """
        Args:
            area_weights: Tensor of shape [n_lat, n_lon] representing the area weights.
            target: Whether to compute metrics on the normalized or denormalized data,
                defaults to "denorm".
            metadata: Mapping of variable names their metadata that will
                used in generating logged image captions.
            reference_means: Dataset containing reference time-mean values
                for bias computation.
        """
        self._target = target
        if metadata is None:
            self._metadata: Dict[str, Dict[str, str]] = {}
        else:
            self._metadata = metadata
        # Dictionaries of tensors of shape [n_lat, n_lon] represnting time means
        self._data: Optional[Dict[str, torch.Tensor]] = None
        self._n_timesteps = 0
        self._n_samples: Optional[int] = None
        self._reference_means = reference_means
        self._reference_validated = False
        self._area_weights = area_weights

    @staticmethod
    def _add_or_initialize_time_mean(
        maybe_dict: Optional[Dict[str, torch.Tensor]],
        new_data: Dict[str, torch.Tensor],
        ignore_initial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        sample_dim = 0
        time_dim = 1
        if ignore_initial:
            time_slice = slice(1, None)
        else:
            time_slice = slice(0, None)
        if maybe_dict is None:
            d: Dict[str, torch.Tensor] = {
                name: tensor[:, time_slice].sum(dim=time_dim).sum(dim=sample_dim)
                for name, tensor in new_data.items()
            }
        else:
            d = dict(maybe_dict)
            for name, tensor in new_data.items():
                d[name] += tensor[:, time_slice].sum(dim=time_dim).sum(dim=sample_dim)
        return d

    @torch.no_grad()
    def record_batch(
        self,
        data: Dict[str, torch.Tensor],
        i_time_start: int = 0,
    ):
        ignore_initial = i_time_start == 0
        if ignore_initial:
            raise Exception(
                "This is wrong! Make sure this isnt called while recording "
                "initial prognostic"
            )
        self._data = self._add_or_initialize_time_mean(self._data, data, ignore_initial)
        if self._n_samples is None:
            self._n_samples = data[list(data)[0]].size(0)
        if ignore_initial:
            self._n_timesteps = data[list(data)[0]].size(1) - 1
        else:
            self._n_timesteps += data[list(data)[0]].size(1)
        if not self._reference_validated:
            if self._reference_means is not None:
                self.get_logs(label="")
            self._reference_validated = True

    def get_data(self) -> Dict[str, torch.Tensor]:
        if self._n_timesteps == 0 or self._data is None:
            raise ValueError("No data recorded.")

        ret = {}
        names = sorted(list(self._data.keys()))  # sort for rank-consistent order
        for name in names:
            value = self._data[name]
            gen = all_reduce_mean(value / self._n_timesteps / self._n_samples)
            ret[name] = gen
        return ret

    @torch.no_grad()
    def get_logs(self, label: str) -> Dict[str, Union[float, wandb.Image]]:
        logs: Dict[str, Union[float, wandb.Image]] = {}
        data = self.get_data()
        gen_map_key = "gen_map"
        for name, pred in data.items():
            vmin_pred, vmax_pred = get_cmap_limits(pred.cpu().numpy())
            prediction_image = wandb.Image(
                plot_imshow(pred.cpu().numpy()),
                caption=self._get_caption(gen_map_key, name, vmin_pred, vmax_pred),
            )
            plt.close("all")
            logs.update(
                {
                    f"{gen_map_key}/{name}": prediction_image,
                }
            )
            if self._reference_means is not None and name in self._reference_means:
                pair = _TargetGenPair(
                    name=name,
                    target=torch.as_tensor(
                        self._reference_means[name].values, device=pred.device
                    ),
                    gen=pred,
                    area_weights=self._area_weights,
                )
                bias_map = pair.bias().cpu().numpy()
                vmin_bias, vmax_bias = get_cmap_limits(bias_map, diverging=True)
                bias_fig: plt.Figure = plot_imshow(
                    bias_map, vmin=vmin_bias, vmax=vmax_bias, cmap="RdBu_r"
                )
                bias_image = wandb.Image(
                    bias_fig,
                    caption=self._get_caption("bias_map", name, vmin_bias, vmax_bias),
                )
                logs.update({f"ref_bias/{name}": pair.weighted_mean_bias()})
                logs.update({f"ref_rmse/{name}": pair.rmse()})
                logs.update({f"ref_bias_map/{name}": bias_image})

        if len(label) != 0:
            return {f"{label}/{key}": logs[key] for key in logs}
        return logs

    def _get_caption(self, key: str, name: str, vmin: float, vmax: float) -> str:
        if name in self._metadata:
            caption_name = self._metadata[name]["long_name"]
            units = self._metadata[name]["units"]
        else:
            caption_name, units = name, "unknown_units"
        caption = self._image_captions[key].format(name=caption_name, units=units)
        caption += f" vmin={vmin:.4g}, vmax={vmax:.4g}."
        return caption


class TimeMeanEvaluatorAggregator:
    """Statistics and images on the time-mean state.

    This aggregator keeps track of the time-mean state, then computes
    statistics and images on that time-mean state when logs are retrieved.
    """

    _image_captions = {
        "bias_map": "{name} time-mean bias (generated - target) [{units}]",
        "gen_map": "{name} time-mean generated [{units}]",
    }

    def __init__(
        self,
        area_weights: torch.Tensor,
        target: Literal["norm", "denorm"] = "denorm",
        metadata: Optional[Dict[str, Dict[str, str]]] = None,
        reference_means: Optional[xr.Dataset] = None,
        channel_mean_names: Optional[List[str]] = None,
    ):
        """
        Args:
            area_weights: Tensor of shape [n_lat, n_lon] representing the area weights.
            target: Whether to compute metrics on the normalized or denormalized data,
                defaults to "denorm".
            metadata: Mapping of variable names their metadata that will
                used in generating logged image captions.
            reference_means: Dataset containing reference time-mean values
                for bias computation.
            channel_mean_names: Names of variables whose RMSE will be averaged. If
                not provided, all available variables will be used.
        """
        self._target = target
        if metadata is None:
            self._metadata: Dict[str, Dict[str, str]] = {}
        else:
            self._metadata = metadata
        self._area_weights = area_weights
        # Dictionaries of tensors of shape [n_lat, n_lon] represnting time means
        self._target_agg = TimeMeanAggregator(
            target=target, metadata=metadata, area_weights=self._area_weights
        )
        self._gen_agg = TimeMeanAggregator(
            target=target,
            metadata=metadata,
            reference_means=reference_means,
            area_weights=self._area_weights,
        )
        self._channel_mean_names = channel_mean_names

    @torch.no_grad()
    def record_batch(
        self,
        target_data: Dict[str, torch.Tensor],
        gen_data: Dict[str, torch.Tensor],
        target_data_norm: Dict[str, torch.Tensor],
        gen_data_norm: Dict[str, torch.Tensor],
        i_time_start: int = 0,
    ):
        if self._target == "norm":
            target_data = target_data_norm
            gen_data = gen_data_norm
        self._target_agg.record_batch(target_data, i_time_start)
        self._gen_agg.record_batch(gen_data, i_time_start)

    def _get_target_gen_pairs(self) -> List[_TargetGenPair]:
        target_data = self._target_agg.get_data()
        gen_data = self._gen_agg.get_data()

        ret = []
        for name in gen_data.keys():
            ret.append(
                _TargetGenPair(
                    gen=gen_data[name],
                    target=target_data[name],
                    name=name,
                    area_weights=self._area_weights,
                )
            )
        return ret

    @torch.no_grad()
    def get_logs(
        self, label: str
    ) -> Dict[str, Union[float, torch.Tensor, wandb.Image]]:
        logs = self._gen_agg.get_logs("")
        preds = self._get_target_gen_pairs()
        bias_map_key = "bias_map"
        rmse_all_channels = {}
        for pred in preds:
            bias_data = pred.bias().cpu().numpy()
            vmin_bias, vmax_bias = get_cmap_limits(bias_data, diverging=True)
            bias_fig = plot_imshow(
                bias_data, vmin=vmin_bias, vmax=vmax_bias, cmap="RdBu_r"
            )
            bias_image = wandb.Image(
                bias_fig,
                caption=self._get_caption(
                    bias_map_key, pred.name, vmin_bias, vmax_bias
                ),
            )
            plt.close("all")
            rmse_all_channels[pred.name] = pred.rmse()
            logs.update({f"rmse/{pred.name}": rmse_all_channels[pred.name]})
            if self._target == "denorm":
                logs.update(
                    {
                        f"{bias_map_key}/{pred.name}": bias_image,
                        f"bias/{pred.name}": pred.weighted_mean_bias(),
                    }
                )
        if self._target == "norm":
            metric_name = "rmse/channel_mean"
            if self._channel_mean_names is None:
                values_to_average = list(rmse_all_channels.values())
            else:
                values_to_average = [
                    rmse_all_channels[name] for name in self._channel_mean_names
                ]
            logs.update({metric_name: sum(values_to_average) / len(values_to_average)})

        if len(label) != 0:
            return {f"{label}/{key}": logs[key] for key in logs}
        return logs

    def _get_caption(self, key: str, name: str, vmin: float, vmax: float) -> str:
        if name in self._metadata:
            caption_name = self._metadata[name]["long_name"]
            units = self._metadata[name]["units"]
        else:
            caption_name, units = name, "unknown_units"
        caption = self._image_captions[key].format(name=caption_name, units=units)
        caption += f" vmin={vmin:.4g}, vmax={vmax:.4g}."
        return caption
