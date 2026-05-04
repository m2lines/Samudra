"""Per-scale validation aggregator for the match/mix schedules.

For each batch, route by ``batch.ctx.label_mask.shape[-2:]`` to a per-scale
state. Loss is summed per scale and all-reduced across ranks at log time
(safe because the DDP sampler emits same-scale chunks across all ranks per
step, so every rank sees the same number of batches per scale). The
snapshot retains the *last* batch per scale and emits one ``error =
gen - target`` map per variable.

Subclasses ``TrainAggregator`` so we keep the default per-channel /
per-depth / per-variable loss breakdowns (averaged across scales) on top
of the per-scale loss + snapshot we add here. Without that inheritance,
the multi-scale path would have *fewer* diagnostics than the previous
``ValidateAggregator({}, ...)`` no-op.

Logs (only on the main process for images):
    val/mean/loss                            (global mean across scales)
    val/depth/{i}/loss, val/var/{name}/loss  (per-channel breakdowns)
    val/{H}x{W}/loss                         (per-scale mean)
    val/{H}x{W}/snapshot/image-error/{var}   (per-scale snapshot)
"""

import torch

from ocean_emulators.aggregator.plotting import plot_paneled_data
from ocean_emulators.aggregator.train import TrainAggregator
from ocean_emulators.constants import (
    BoundaryVarNames,
    GridSize,
    PrognosticVarNames,
    TensorMap,
)
from ocean_emulators.utils.data import DataSource, Normalize, get_aggregator_dicts
from ocean_emulators.utils.distributed import all_reduce_mean, is_main_process
from ocean_emulators.utils.output import ValBatchOutput
from ocean_emulators.utils.wandb import Metrics, MetricsDict


class PerScaleSnapshotValidateAggregator(TrainAggregator):
    """Per-scale validation loss + per-variable error snapshots.

    Use this for ``match``/``mix`` training schedules where averaging across
    resolutions would hide per-scale pathologies. Each registered scale gets
    its own ``Normalize`` so unnormalize-for-display uses the correct stats
    for that scale.

    Inherits ``TrainAggregator`` so the standard ``val/mean/loss``,
    ``val/depth/...``, ``val/var/...``, ``val/channel/...`` breakdowns are
    still produced on top of the per-scale routing this class adds.
    """

    def __init__(
        self,
        sources: list[DataSource],
        prognostic_var_names: PrognosticVarNames,
        boundary_var_names: BoundaryVarNames,
        tensor_map: TensorMap,
        hist: int,
        num_prognostic_channels: int,
        *,
        log_images: bool = True,
    ):
        super().__init__(tensor_map)
        if len(sources) == 0:
            raise ValueError(
                "PerScaleSnapshotValidateAggregator requires at least one source."
            )
        self._hist = hist
        self._num_prognostic_channels = num_prognostic_channels
        self._log_images = log_images

        # NB: ``TrainAggregator`` already uses ``self._n_batches`` (int) as a
        # global batch counter. Use a different name for our per-scale dict
        # so the parent's bookkeeping isn't shadowed.
        self._normalize_for: dict[GridSize, Normalize] = {}
        self._metadata_for: dict[GridSize, dict[str, dict[str, str]]] = {}
        self._loss_sum_per_scale: dict[GridSize, torch.Tensor] = {}
        self._batch_count_per_scale: dict[GridSize, int] = {}
        self._last_batch_per_scale: dict[GridSize, ValBatchOutput | None] = {}
        for src in sources:
            key: GridSize = src.grid_size
            if key in self._normalize_for:
                # Two sources with the same grid size would route ambiguously.
                # If this ever needs to be supported (e.g. cross-resolution
                # mix runs with two distinct datasets at the same grid),
                # extend the routing key beyond grid size.
                raise ValueError(
                    f"Duplicate source grid_size {key}; per-scale routing is "
                    f"by grid size and cannot disambiguate."
                )
            self._normalize_for[key] = Normalize(
                src,
                prognostic_var_names=prognostic_var_names,
                boundary_var_names=boundary_var_names,
            )
            self._metadata_for[key] = src.metadata
            self._loss_sum_per_scale[key] = torch.tensor(0.0)
            self._batch_count_per_scale[key] = 0
            self._last_batch_per_scale[key] = None

    @torch.no_grad()
    def record_validation_batch(self, batch: ValBatchOutput) -> None:
        # Parent records global per-channel/depth/var loss across all scales.
        super().record_batch(batch)

        h, w = batch.ctx.label_mask.shape[-2:]
        key: GridSize = (int(h), int(w))
        if key not in self._normalize_for:
            raise KeyError(
                f"No per-scale state for batch grid_size {key}; "
                f"registered: {list(self._normalize_for)}"
            )
        self._loss_sum_per_scale[key] = (
            self._loss_sum_per_scale[key] + batch.loss.detach().cpu()
        )
        self._batch_count_per_scale[key] += 1
        if self._log_images:
            self._last_batch_per_scale[key] = batch

    @torch.no_grad()
    def get_logs(self, label: str = "val") -> Metrics:
        # Start with the parent's TrainAggregator outputs (val/mean/loss,
        # val/depth/..., val/var/..., val/channel/...), then append per-scale.
        logs: MetricsDict = dict(super().get_logs(label))
        for grid_size, n in self._batch_count_per_scale.items():
            scale = f"{grid_size[0]}x{grid_size[1]}"
            if n == 0:
                # No batches at this scale this epoch. Can happen if drop_last
                # trims a scale's tail or the val set lacks samples at a scale.
                continue
            local_mean = self._loss_sum_per_scale[grid_size] / n
            mean_loss = float(all_reduce_mean(local_mean).cpu().numpy())
            logs[f"{label}/{scale}/loss"] = mean_loss

            last = self._last_batch_per_scale[grid_size]
            if self._log_images and last is not None:
                logs.update(
                    self._error_images(
                        last,
                        grid_size,
                        prefix=f"{label}/{scale}/snapshot",
                    )
                )
        return logs

    @torch.no_grad()
    def _error_images(
        self,
        batch: ValBatchOutput,
        grid_size: GridSize,
        prefix: str,
    ) -> MetricsDict:
        # Skip plot rendering on non-main ranks, but still build the wet mask
        # and unnormalize on every rank if we ever need an all-reduce here
        # (we don't currently — error map is just the last-batch first-sample).
        if not is_main_process():
            return {}

        wet = batch.ctx.label_mask
        first_chunk = wet.shape[0] // (self._hist + 1)
        wet = wet[:first_chunk]

        normalize = self._normalize_for[grid_size]
        _, target_unnorm = get_aggregator_dicts(
            batch.target_data,
            normalize=normalize,
            tensor_map=self.tensor_map,
            wet=wet,
            long_rollout=False,
            input_type="prognostic",
            num_prognostic_channels=self._num_prognostic_channels,
            hist=self._hist,
        )
        _, gen_unnorm = get_aggregator_dicts(
            batch.gen_data,
            normalize=normalize,
            tensor_map=self.tensor_map,
            wet=wet,
            long_rollout=False,
            input_type="prognostic",
            num_prognostic_channels=self._num_prognostic_channels,
            hist=self._hist,
        )

        time_dim = 1
        target_time = 0  # first output time step
        metadata = self._metadata_for[grid_size]
        out: MetricsDict = {}
        for name in gen_unnorm:
            gen = gen_unnorm[name].select(dim=time_dim, index=target_time)[0].cpu()
            target = (
                target_unnorm[name].select(dim=time_dim, index=target_time)[0].cpu()
            )
            error = [[(gen - target).numpy()]]
            caption = self._caption_for(name, metadata)
            out[f"{prefix}/image-error/{name}"] = plot_paneled_data(
                error, diverging=True, caption=caption
            )
        return out

    @staticmethod
    def _caption_for(name: str, metadata: dict[str, dict[str, str]]) -> str:
        if name in metadata:
            long_name = metadata[name]["long_name"]
            units = metadata[name]["units"]
        else:
            long_name, units = name, "unknown_units"
        return (
            f"{long_name} one step full field error (generated - target) "
            f"for last sample [{units}]"
        )
