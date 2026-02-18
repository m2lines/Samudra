import torch
import wandb
import xarray as xr

from ocean_emulators.utils.data import Normalize, get_aggregator_dicts
from ocean_emulators.utils.output import ModelInferenceOutput
from ocean_emulators.utils.wandb import Metrics, MetricsDict

from ..validate.reduced import MeanAggregator as OneStepMeanAggregator
from .reduced import MeanAggregator
from .time_mean import TimeMeanEvaluatorAggregator


class InferenceEvaluatorAggregator:
    """
    Aggregates statistics for inference comparing a generated and target series.
    """

    def __init__(
        self,
        n_timesteps: int,
        metadata: dict[str, dict[str, str]],
        hist: int,
        area_weights: torch.Tensor,
        wet: torch.Tensor,
        num_prognostic_channels: int,
        record_step_20: bool = True,
        log_global_mean_time_series: bool = True,
        log_global_mean_norm_time_series: bool = True,
        time_mean_reference_data: xr.Dataset | None = None,
        channel_mean_names: list[str] | None = None,
    ):
        """
        Args:
            n_timesteps: Number of timesteps of inference that will be run.
            metadata: Mapping of variable names their metadata that will
                used in generating logged image captions.
            hist: Number of timesteps of history.
            area_weights: Area weights for the data.
            wet: Wet mask for the data.
            num_prognostic_channels: Number of prognostic channels in the data.
            record_step_20: Whether to record the mean of the 20th steps.
            log_global_mean_time_series: Whether to log global mean time series metrics.
            log_global_mean_norm_time_series: Whether to log the normalized global mean
                time series metrics.
            time_mean_reference_data: Reference time means for computing bias stats.
            channel_mean_names: List of channel names to compute the mean of.
        """
        self._aggregators: dict[
            str, MeanAggregator | OneStepMeanAggregator | TimeMeanEvaluatorAggregator
        ] = {}
        self._time_dependent_aggregators: dict[str, TimeMeanEvaluatorAggregator] = {}
        self._log_time_series = (
            log_global_mean_time_series or log_global_mean_norm_time_series
        )
        if log_global_mean_time_series:
            self._aggregators["mean"] = MeanAggregator(
                target="denorm",
                n_timesteps=n_timesteps,
                metadata=metadata,
                area_weights=area_weights,
            )
        if log_global_mean_norm_time_series:
            self._aggregators["mean_norm"] = MeanAggregator(
                target="norm",
                n_timesteps=n_timesteps,
                metadata=metadata,
                area_weights=area_weights,
            )
        if record_step_20:
            self._aggregators["mean_step_20"] = OneStepMeanAggregator(
                target_time=20,
                area_weights=area_weights,
            )
        self._aggregators["time_mean"] = TimeMeanEvaluatorAggregator(
            metadata=metadata,
            area_weights=area_weights,
            reference_means=time_mean_reference_data,
            channel_mean_names=channel_mean_names,
        )
        self._aggregators["time_mean_norm"] = TimeMeanEvaluatorAggregator(
            metadata=metadata,
            area_weights=area_weights,
            target="norm",
            reference_means=time_mean_reference_data,
            channel_mean_names=channel_mean_names,
        )

        self._summary_aggregators = {
            name: agg
            for name, agg in list(self._aggregators.items())
            + list(self._time_dependent_aggregators.items())
            if name not in ["mean", "mean_norm"]
        }
        self._n_timesteps_seen = 0
        self._normalize = Normalize.get_instance()
        self.num_prognostic_channels = num_prognostic_channels
        self.hist = hist
        self.wet = wet

    @property
    def log_time_series(self) -> bool:
        return self._log_time_series

    @torch.no_grad()
    def record_batch(self, data: ModelInferenceOutput):
        if len(data.prediction) == 0:
            raise ValueError("No prediction values in data")
        if len(data.target) == 0:
            raise ValueError("No target values in data")
        total_len = len(data.time)
        assert data.prediction.shape[0] == total_len // (self.hist + 1)
        target_norm_dict, target_unnorm_dict = get_aggregator_dicts(
            data.target,
            wet=self.wet,
            long_rollout=True,
            input_type="prognostic",
            num_prognostic_channels=self.num_prognostic_channels,
            hist=self.hist,
        )
        gen_norm_dict, gen_unnorm_dict = get_aggregator_dicts(
            data.prediction,
            wet=self.wet,
            long_rollout=True,
            input_type="prognostic",
            num_prognostic_channels=self.num_prognostic_channels,
            hist=self.hist,
        )

        for aggregator in self._aggregators.values():
            aggregator.record_batch(
                target_data=target_unnorm_dict,
                gen_data=gen_unnorm_dict,
                target_data_norm=target_norm_dict,
                gen_data_norm=gen_norm_dict,
                i_time_start=self._n_timesteps_seen,
            )
        # TODO: Add time-dependent aggregators
        # for time_dependent_aggregator in self._time_dependent_aggregators.values():
        #     time_dependent_aggregator.record_batch(
        #         time=data.time,
        #         target_data=target_unnorm_dict,
        #         gen_data=gen_unnorm_dict,
        #     )
        key = list(target_unnorm_dict.keys())[0]
        n_times = target_unnorm_dict[key].shape[1]
        logs = self._get_inference_logs_slice(
            step_slice=slice(self._n_timesteps_seen, self._n_timesteps_seen + n_times),
        )
        self._n_timesteps_seen += n_times
        return logs

    def record_initial_prognostic(
        self,
        initial_prognostic: torch.Tensor,
    ):
        if self._n_timesteps_seen != 0:
            raise RuntimeError(
                "record_initial_condition may only be called once, "
                "before recording any batches"
            )

        data_norm_dict, data_unnorm_dict = get_aggregator_dicts(
            initial_prognostic,
            wet=self.wet,
            long_rollout=True,
            input_type="input",
            num_prognostic_channels=self.num_prognostic_channels,
            hist=self.hist,
        )
        for aggregator_name in ["mean", "mean_norm"]:
            aggregator = self._aggregators.get(aggregator_name)
            if aggregator is not None:
                aggregator.record_batch(
                    target_data=data_unnorm_dict,
                    gen_data=data_unnorm_dict,
                    target_data_norm=data_norm_dict,
                    gen_data_norm=data_norm_dict,
                    i_time_start=0,
                )
        key = list(data_unnorm_dict.keys())[0]
        n_times = data_unnorm_dict[key].shape[1]
        logs = self._get_inference_logs_slice(
            step_slice=slice(0, n_times),
        )
        self._n_timesteps_seen = n_times
        return logs

    def get_summary_logs(self):
        # These aggregators require a full timeseries of
        # data and thus wandb logged at the end
        logs: MetricsDict = {}
        for name, aggregator in self._summary_aggregators.items():
            logs.update(aggregator.get_logs(label=name))
        return logs

    @torch.no_grad()
    def _get_logs(self) -> Metrics:
        """
        Returns logs as can be reported to WandB.
        """
        logs: MetricsDict = {}
        for name, aggregator in self._aggregators.items():
            logs.update(aggregator.get_logs(label=name))
        for name, time_dependent_aggregator in self._time_dependent_aggregators.items():
            logs.update(time_dependent_aggregator.get_logs(label=name))
        return logs

    @torch.no_grad()
    def _get_inference_logs_slice(self, step_slice: slice):
        """
        Returns a subset of the time series for applicable metrics
        for a specific slice of as can be reported to WandB.

        Args:
            step_slice: Timestep slice to determine the time series subset.

        Returns:
            Tuple of start index and list of logs.
        """
        logs = {}
        for name, aggregator in self._aggregators.items():
            if isinstance(aggregator, MeanAggregator):
                logs.update(aggregator.get_logs(label=name, step_slice=step_slice))
        return to_inference_logs(logs)


def to_inference_logs(log):
    # we have a dictionary which contains WandB tables
    # which we will convert to a list of dictionaries, one for each
    # row in the tables. Any scalar values will be reported in the last
    # dictionary.
    n_rows = 0
    for val in log.values():
        if isinstance(val, wandb.Table):
            n_rows = max(n_rows, len(val.data))
    logs: list[dict[str, float | int]] = []
    for i in range(max(1, n_rows)):  # need at least one for non-series values
        logs.append({})
    for key, val in log.items():
        if isinstance(val, wandb.Table):
            for i, row in enumerate(val.data):
                for j, col in enumerate(val.columns):
                    key_without_table_name = key[: key.rfind("/")]
                    logs[i][f"{key_without_table_name}/{col}"] = row[j]
        else:
            logs[-1][key] = val
    return logs
