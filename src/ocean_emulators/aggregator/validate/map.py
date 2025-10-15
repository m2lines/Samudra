import torch

from ocean_emulators.aggregator.plotting import plot_paneled_data
from ocean_emulators.aggregator.validate.sub_aggregator import ValidateSubAggregator
from ocean_emulators.utils.distributed import all_reduce_mean


class MapAggregator(ValidateSubAggregator):
    """
    An aggregator that records the average over batches as function of lat and lon.
    """

    _captions = {
        "full-field": (
            "{name} one step mean full field; "
            "(top) generated and (bottom) target [{units}]"
        ),
        "error": (
            "{name} one step mean full field error (generated - target) [{units}]"
        ),
    }

    def __init__(
        self, metadata: dict[str, dict[str, str]] | None = None, hist: int = 0
    ):
        """
        Args:
            metadata: Mapping of variable names their metadata that will
                used in generating logged image captions.
            hist: Number of history steps to include in the snapshot.
        """
        if metadata is None:
            metadata = {}
        else:
            self._metadata = metadata
        self._n_batches = 0
        self._target_data: dict[str, torch.Tensor] = {}
        self._gen_data: dict[str, torch.Tensor] = {}
        self.hist = hist

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
        self._loss = loss
        for name in target_data:
            if name in self._target_data:
                self._target_data[name] += target_data[name].mean(dim=0)
            else:
                self._target_data[name] = target_data[name].mean(dim=0)
        for name in gen_data:
            if name in self._gen_data:
                self._gen_data[name] += gen_data[name].mean(dim=0)
            else:
                self._gen_data[name] = gen_data[name].mean(dim=0)
        self._n_batches += 1

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        time_dim = 0
        target_time = self.hist  # Use latest time step
        image_logs = {}
        sorted_names = sorted(list(self._gen_data.keys()))
        for name in sorted_names:
            # use first sample in batch
            gen = (
                all_reduce_mean(
                    self._gen_data[name].select(dim=time_dim, index=target_time)
                )
                / self._n_batches
            )
            target = (
                all_reduce_mean(
                    self._target_data[name].select(dim=time_dim, index=target_time)
                )
                / self._n_batches
            )
            image_logs[f"image-error/{name}"] = plot_paneled_data(
                [[(gen - target).cpu().numpy()]],
                diverging=True,
                caption=self._get_caption("error", name),
            )
            image_logs[f"image-full-field/{name}"] = plot_paneled_data(
                [
                    [gen.cpu().numpy()],
                    [target.cpu().numpy()],
                ],
                diverging=False,
                caption=self._get_caption("full-field", name),
            )
        image_logs = {f"{label}/{key}": image_logs[key] for key in image_logs}
        return image_logs

    def _get_caption(self, key: str, name: str) -> str:
        if name in self._metadata:
            caption_name = self._metadata[name]["long_name"]
            units = self._metadata[name]["units"]
        else:
            caption_name, units = name, "unknown_units"
        caption = self._captions[key].format(name=caption_name, units=units)
        return caption
