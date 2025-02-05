# from typing import Dict, Iterable, List, Optional, Union

# import torch
# import wandb
# import xarray as xr

# from .annual import GlobalMeanAnnualAggregator
# from .reduced import SingleTargetMeanAggregator
# from .time_mean import TimeMeanAggregator


# class InferenceAggregator:
#     """Aggregates statistics on a single timeseries of data."""

#     def __init__(
#         self,
#         n_timesteps: int,
#         timestep: int,
#         metadata: Dict[str, Dict[str, str]],
#         time_mean_reference_data: Optional[xr.Dataset] = None,
#         log_global_mean_time_series: bool = True,
#     ):
#         self._log_time_series = log_global_mean_time_series
#         aggregators = {}
#         if log_global_mean_time_series:
#             aggregators["mean"] = SingleTargetMeanAggregator(
#                 n_timesteps=n_timesteps,
#             )
#         aggregators["time_mean"] = TimeMeanAggregator(
#             metadata=metadata,
#             reference_means=time_mean_reference_data,
#         )
#         aggregators["annual"] = GlobalMeanAnnualAggregator(
#             timestep,
#             metadata,
#         )
#         self._aggregators = aggregators
#         self._summary_aggregators = {
#             name: aggregators[name] for name in ["time_mean", "annual"]
#         }
#         self._time_dependent_aggregator_names = ["annual"]
#         self._n_timesteps_seen = 0

#     @property
#     def log_time_series(self) -> bool:
#         return self._log_time_series

#     @torch.no_grad()
#     def record_batch(self, data):
#         """
#         Record a batch of data.

#         Args:
#             data: Batch of data to record.
#         """
#         if len(data.data) == 0:
#             raise ValueError("data is empty")
#         for name in self._aggregators:
#             if name in self._time_dependent_aggregator_names:
#                 self._aggregators[name].record_batch(data.time, data.data)
#             else:
#                 self._aggregators[name].record_batch(
#                     data=data.data,
#                     i_time_start=self._n_timesteps_seen,
#                 )
#         n_times = data.time.shape[1]
#         logs = self._get_inference_logs_slice(
#             step_slice=slice(self._n_timesteps_seen,
#                               self._n_timesteps_seen + n_times),
#         )
#         self._n_timesteps_seen += n_times
#         return logs

#     def record_initial_condition(
#         self,
#         initial_condition,
#     ):
#         if self._n_timesteps_seen != 0:
#             raise RuntimeError(
#                 "record_initial_condition may only be called once, "
#                 "before recording any batches"
#             )
#         batch_data = initial_condition.as_batch_data()
#         if "mean" in self._aggregators:
#             self._aggregators["mean"].record_batch(
#                 data=batch_data.data,
#                 i_time_start=0,
#             )
#         n_times = batch_data.time.shape[1]
#         logs = self._get_inference_logs_slice(
#             step_slice=slice(self._n_timesteps_seen,
#                               self._n_timesteps_seen + n_times),
#         )
#         self._n_timesteps_seen = n_times
#         return logs

#     def get_summary_logs(self):
#         logs = {}
#         for name, aggregator in self._summary_aggregators.items():
#             logs.update(aggregator.get_logs(label=name))
#         return logs

#     @torch.no_grad()
#     def _get_logs(self):
#         """
#         Returns logs as can be reported to WandB.
#         """
#         logs = {}
#         for name, aggregator in self._aggregators.items():
#             logs.update(aggregator.get_logs(label=name))
#         return logs

#     @torch.no_grad()
#     def _get_inference_logs(self) -> List[Dict[str, Union[float, int]]]:
#         """
#         Returns a list of logs to report to WandB.

#         This is done because in inference, we use the wandb step
#         as the time step, meaning we need to re-organize the logged data
#         from tables into a list of dictionaries.
#         """
#         return to_inference_logs(self._get_logs())

#     @torch.no_grad()
#     def _get_inference_logs_slice(self, step_slice: slice):
#         """
#         Returns a subset of the time series for applicable metrics
#         for a specific slice of as can be reported to WandB.

#         Args:
#             step_slice: Timestep slice to determine the time series subset.
#         """
#         logs = {}
#         for name, aggregator in self._aggregators.items():
#             if isinstance(aggregator, SingleTargetMeanAggregator):
#                 logs.update(aggregator.get_logs(label=name, step_slice=step_slice))
#         return to_inference_logs(logs)

#     @torch.no_grad()
#     def get_datasets(
#         self, excluded_aggregators: Optional[Iterable[str]] = None
#     ) -> Dict[str, xr.Dataset]:
#         """
#         Returns datasets from combined aggregators.

#         Args:
#             excluded_aggregators: aggregator names for which `get_dataset`
#                 should not be called and no output should be returned.

#         Returns:
#             Dictionary of datasets from aggregators.
#         """
#         if excluded_aggregators is None:
#             excluded_aggregators = []

#         return {
#             name: agg.get_dataset()
#             for name, agg in self._aggregators.items()
#             if name not in excluded_aggregators
#         }


# def to_inference_logs(log):
#     # we have a dictionary which contains WandB tables
#     # which we will convert to a list of dictionaries, one for each
#     # row in the tables. Any scalar values will be reported in the last
#     # dictionary.
#     n_rows = 0
#     for val in log.values():
#         if isinstance(val, wandb.Table):
#             n_rows = max(n_rows, len(val.data))
#     logs: List[Dict[str, Union[float, int]]] = []
#     for i in range(max(1, n_rows)):  # need at least one for non-series values
#         logs.append({})
#     for key, val in log.items():
#         if isinstance(val, wandb.Table):
#             for i, row in enumerate(val.data):
#                 for j, col in enumerate(val.columns):
#                     key_without_table_name = key[: key.rfind("/")]
#                     logs[i][f"{key_without_table_name}/{col}"] = row[j]
#         else:
#             logs[-1][key] = val
#     return logs


# def table_to_logs(table: wandb.Table) -> List[Dict[str, Union[float, int]]]:
#     """
#     Converts a WandB table into a list of dictionaries.
#     """
#     logs = []
#     for row in table.data:
#         logs.append({table.columns[i]: row[i] for i in range(len(row))})
#     return logs
