import torch

from ocean_emulators.aggregator.plotting import plot_paneled_data
from ocean_emulators.aggregator.validate.sub_aggregator import ValidateSubAggregator
from ocean_emulators.utils.wandb import Metrics


class SnapshotAggregator(ValidateSubAggregator):
    """
    An aggregator that records the first sample of the last batch of data.
    """

    _captions = {
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

    def __init__(
        self,
        metadata: dict[str, dict[str, str]] | None = None,
        hist: int = 0,
        include_names: tuple[str, ...] | None = None,
    ):
        """
        Args:
            metadata: Mapping of variable names their metadata that will
                used in generating logged image captions.
            hist: Number of history steps to include in the snapshot.
        """
        self._metadata = metadata or {}
        self.hist = hist
        self._include_names = include_names

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
        self._loss = loss
        self._target_data = target_data
        self._gen_data = gen_data
        self._target_data_norm = target_data_norm
        self._gen_data_norm = gen_data_norm
        self._input_data = input_data
        self._input_data_norm = input_data_norm

    @torch.no_grad()
    def get_logs(self, label: str) -> Metrics:
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        time_dim = 1
        target_time = 0  # first output time step
        input_time = self.hist  # last input time step
        image_logs = {}
        if self._include_names is None:
            names = list(self._gen_data.keys())
        else:
            names = [name for name in self._include_names if name in self._gen_data]
        for name in names:
            # use first sample in batch
            gen = self._gen_data[name].select(dim=time_dim, index=target_time)[0].cpu()
            target = (
                self._target_data[name].select(dim=time_dim, index=target_time)[0].cpu()
            )
            input = (
                self._input_data[name].select(dim=time_dim, index=input_time)[0].cpu()
            )
            images = {}
            images["error"] = [[(gen - target).numpy()]]
            images["full-field"] = [[gen.numpy()], [target.numpy()]]
            images["residual"] = [[(gen - input).numpy()], [(target - input).numpy()]]
            for key, data in images.items():
                if key == "error" or key == "residual":
                    diverging = True
                else:
                    diverging = False
                caption = self._get_caption(key, name)
                wandb_image = plot_paneled_data(data, diverging, caption=caption)
                image_logs[f"image-{key}/{name}"] = wandb_image
        image_logs = {f"{label}/{key}": image_logs[key] for key in image_logs}
        return image_logs

    def _get_caption(self, key: str, name: str) -> str:
        metadata_name = name
        if metadata_name not in self._metadata and "_" in metadata_name:
            metadata_name = metadata_name.rsplit("_", 1)[0]
        if metadata_name in self._metadata:
            caption_name = self._metadata[metadata_name]["long_name"]
            units = self._metadata[metadata_name]["units"]
        else:
            caption_name, units = name, "unknown_units"
        caption = self._captions[key].format(name=caption_name, units=units)
        return caption
