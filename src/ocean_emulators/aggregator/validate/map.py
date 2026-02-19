from dataclasses import dataclass, field

import torch

from ocean_emulators.aggregator.plotting import plot_paneled_data
from ocean_emulators.utils.distributed import all_reduce_mean
from ocean_emulators.utils.wandb import Metrics

_MAP_CAPTIONS = {
    "full-field": (
        "{name} one step mean full field; (top) generated and (bottom) target [{units}]"
    ),
    "error": "{name} one step mean full field error (generated - target) [{units}]",
}


@dataclass
class MapState:
    metadata: dict[str, dict[str, str]]
    hist: int
    n_batches: int = 0
    target_data: dict[str, torch.Tensor] = field(default_factory=dict)
    gen_data: dict[str, torch.Tensor] = field(default_factory=dict)


def init_map_state(
    metadata: dict[str, dict[str, str]] | None = None,
    hist: int = 0,
) -> MapState:
    return MapState(metadata={} if metadata is None else metadata, hist=hist)


def _get_caption(captions: dict[str, str], metadata, key: str, name: str) -> str:
    if name in metadata:
        caption_name = metadata[name]["long_name"]
        units = metadata[name]["units"]
    else:
        caption_name, units = name, "unknown_units"
    return captions[key].format(name=caption_name, units=units)


@torch.no_grad()
def record_map_batch(
    state: MapState,
    *,
    loss,
    target_data,
    gen_data,
    input_data,
    target_data_norm,
    gen_data_norm,
    input_data_norm,
):
    del loss, input_data, target_data_norm, gen_data_norm, input_data_norm
    for name in target_data:
        meaned = target_data[name].mean(dim=0)
        if name in state.target_data:
            state.target_data[name] += meaned
        else:
            state.target_data[name] = meaned
    for name in gen_data:
        meaned = gen_data[name].mean(dim=0)
        if name in state.gen_data:
            state.gen_data[name] += meaned
        else:
            state.gen_data[name] = meaned
    state.n_batches += 1


@torch.no_grad()
def get_map_logs(state: MapState, label: str) -> Metrics:
    time_dim = 0
    target_time = state.hist  # Use latest time step
    image_logs = {}
    sorted_names = sorted(list(state.gen_data.keys()))
    for name in sorted_names:
        gen = (
            all_reduce_mean(
                state.gen_data[name].select(dim=time_dim, index=target_time)
            )
            / state.n_batches
        )
        target = (
            all_reduce_mean(
                state.target_data[name].select(dim=time_dim, index=target_time)
            )
            / state.n_batches
        )
        image_logs[f"image-error/{name}"] = plot_paneled_data(
            [[(gen - target).cpu().numpy()]],
            diverging=True,
            caption=_get_caption(_MAP_CAPTIONS, state.metadata, "error", name),
        )
        image_logs[f"image-full-field/{name}"] = plot_paneled_data(
            [
                [gen.cpu().numpy()],
                [target.cpu().numpy()],
            ],
            diverging=False,
            caption=_get_caption(_MAP_CAPTIONS, state.metadata, "full-field", name),
        )
    return {f"{label}/{key}": image_logs[key] for key in image_logs}


class MapAggregator:
    """
    An aggregator that records the average over batches as function of lat and lon.
    """

    _captions = _MAP_CAPTIONS

    def __init__(
        self, metadata: dict[str, dict[str, str]] | None = None, hist: int = 0
    ):
        """
        Args:
            metadata: Mapping of variable names their metadata that will
                used in generating logged image captions.
            hist: Number of history steps to include in the snapshot.
        """
        self._state = init_map_state(metadata=metadata, hist=hist)

    @torch.no_grad()
    def record_batch(
        self,
        loss,
        target_data,
        gen_data,
        input_data,
        target_data_norm,
        gen_data_norm,
        input_data_norm,
    ):
        record_map_batch(
            self._state,
            loss=loss,
            target_data=target_data,
            gen_data=gen_data,
            input_data=input_data,
            target_data_norm=target_data_norm,
            gen_data_norm=gen_data_norm,
            input_data_norm=input_data_norm,
        )

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        return get_map_logs(self._state, label=label)

    def _get_caption(self, key: str, name: str) -> str:
        return _get_caption(self._captions, self._state.metadata, key, name)
