from dataclasses import dataclass

import torch

from ocean_emulators.aggregator.plotting import plot_paneled_data
from ocean_emulators.utils.wandb import Metrics

_SNAPSHOT_CAPTIONS = {
    "full-field": (
        "{name} one step full field for last sample; "
        "(top) generated and (bottom) target [{units}]"
    ),
    "residual": (
        "{name} one step residual (prediction - previous time) for last sample; "
        "(top) generated and (bottom) target [{units}]"
    ),
    "error": (
        "{name} one step full field error (generated - target) "
        "for last sample [{units}]"
    ),
}


@dataclass
class SnapshotState:
    metadata: dict[str, dict[str, str]]
    hist: int
    target_data: dict | None = None
    gen_data: dict | None = None
    target_data_norm: dict | None = None
    gen_data_norm: dict | None = None
    input_data: dict | None = None
    input_data_norm: dict | None = None


def init_snapshot_state(
    metadata: dict[str, dict[str, str]] | None = None,
    hist: int = 0,
) -> SnapshotState:
    return SnapshotState(metadata={} if metadata is None else metadata, hist=hist)


def _get_caption(captions: dict[str, str], metadata, key: str, name: str) -> str:
    if name in metadata:
        caption_name = metadata[name]["long_name"]
        units = metadata[name]["units"]
    else:
        caption_name, units = name, "unknown_units"
    return captions[key].format(name=caption_name, units=units)


@torch.no_grad()
def record_snapshot_batch(
    state: SnapshotState,
    *,
    loss: torch.Tensor,
    target_data,
    gen_data,
    input_data,
    target_data_norm,
    gen_data_norm,
    input_data_norm,
):
    del loss
    state.target_data = target_data
    state.gen_data = gen_data
    state.target_data_norm = target_data_norm
    state.gen_data_norm = gen_data_norm
    state.input_data = input_data
    state.input_data_norm = input_data_norm


@torch.no_grad()
def get_snapshot_logs(state: SnapshotState, label: str) -> Metrics:
    if state.target_data is None or state.gen_data is None or state.input_data is None:
        raise ValueError("No batches have been recorded.")

    time_dim = 1
    target_time = 0  # first output time step
    input_time = state.hist  # last input time step
    image_logs = {}
    for name in state.gen_data.keys():
        gen = state.gen_data[name].select(dim=time_dim, index=target_time)[0].cpu()
        target = (
            state.target_data[name].select(dim=time_dim, index=target_time)[0].cpu()
        )
        input = state.input_data[name].select(dim=time_dim, index=input_time)[0].cpu()
        images = {
            "error": [[(gen - target).numpy()]],
            "full-field": [[gen.numpy()], [target.numpy()]],
            "residual": [[(gen - input).numpy()], [(target - input).numpy()]],
        }
        for key, data in images.items():
            diverging = key == "error" or key == "residual"
            caption = _get_caption(_SNAPSHOT_CAPTIONS, state.metadata, key, name)
            wandb_image = plot_paneled_data(data, diverging, caption=caption)
            image_logs[f"image-{key}/{name}"] = wandb_image
    return {f"{label}/{key}": image_logs[key] for key in image_logs}


class SnapshotAggregator:
    """
    An aggregator that records the first sample of the last batch of data.
    """

    _captions = _SNAPSHOT_CAPTIONS

    def __init__(
        self, metadata: dict[str, dict[str, str]] | None = None, hist: int = 0
    ):
        """
        Args:
            metadata: Mapping of variable names their metadata that will
                used in generating logged image captions.
            hist: Number of history steps to include in the snapshot.
        """
        self._state = init_snapshot_state(metadata=metadata, hist=hist)

    @torch.no_grad()
    def record_batch(
        self,
        loss: torch.Tensor,
        target_data,
        gen_data,
        input_data,
        target_data_norm,
        gen_data_norm,
        input_data_norm,
    ):
        record_snapshot_batch(
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
    def get_logs(self, label: str) -> Metrics:
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        return get_snapshot_logs(self._state, label=label)

    def _get_caption(self, key: str, name: str) -> str:
        return _get_caption(self._captions, self._state.metadata, key, name)
