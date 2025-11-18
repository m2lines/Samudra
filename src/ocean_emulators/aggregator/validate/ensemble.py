import torch
import torch.distributed as dist

from ocean_emulators.aggregator.validate.sub_aggregator import ValidateSubAggregator
from ocean_emulators.utils.distributed import all_reduce_mean


class EnsembleAggregator(ValidateSubAggregator):
    def __init__(
        self,
        area_weights: torch.Tensor,
        var_to_channel: dict[str, int],
        compute_global_best_worst: bool = True,
    ):
        super().__init__()
        self._n_batches = 0
        self._area_weights = area_weights
        self._var_to_channel = dict(var_to_channel)
        self._ensemble_size: int | None = None
        self._compute_global_best_worst = compute_global_best_worst

        # Accumulators (sums over batches; divided by _n_batches in get_logs)
        self._spread_sum = 0.0  # averaged across variables per batch
        self._rmse_sum = (
            0.0  # averaged across variables per batch (ensemble mean vs truth)
        )
        self._mae_sum = 0.0  # averaged across variables per batch
        self._spread_skill_sum = 0.0  # averaged across variables per batch

        # Per-member accumulators (RMSE averaged across variables per batch)
        self._member_rmse_sum: list[float] = []

    @staticmethod
    def _ensure_bc_hw(t: torch.Tensor) -> torch.Tensor:
        """Ensure tensor has shape (B, H, W). Accepts (B,T,H,W), (B,1,H,W) or (B,H,W).

        For tensors with time dimension (T>1), takes the last time step as the prediction.
        """
        if t.ndim == 4:
            # (B, T, H, W) - take last time step or squeeze if T=1
            if t.shape[1] == 1:
                return t.squeeze(1)
            else:
                # Take last time step (the prediction)
                return t[:, -1, :, :]
        if t.ndim == 3:
            return t
        raise ValueError(
            f"Expected (B,T,H,W), (B,1,H,W) or (B,H,W), got {tuple(t.shape)}"
        )

    @staticmethod
    def _safe_scalar(x: torch.Tensor) -> float:
        """Return a Python float from a 0-D tensor, with NaN/Inf guarded to 0.0."""
        v = x.detach()
        if not torch.isfinite(v):
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"_safe_scalar: non-finite value {v.item()}, returning 0.0")
            return 0.0
        return float(v.item())

    @torch.no_grad()
    def _weighted_spatial_mean(
        self, x_bhw: torch.Tensor, w_hw: torch.Tensor
    ) -> torch.Tensor:
        """
        Area-weighted spatial mean of x over (H,W) per sample (B,).
        x_bhw: (B, H, W), w_hw: (H, W) normalized to sum=1.
        """
        B, H, W = x_bhw.shape
        return (x_bhw * w_hw).view(B, H * W).sum(dim=1)

    @torch.no_grad()
    def record_batch(
        self,
        *,
        loss: torch.Tensor = torch.tensor(float("nan")),
        target_data,
        gen_data,
        target_data_norm,
        gen_data_norm,
        input_data: dict[str, torch.Tensor] | None = None,
        input_data_norm: dict[str, torch.Tensor] | None = None,
        i_time_start: int = 0,
        ensemble_data: torch.Tensor | None = None,
    ):
        # TODO (amogh) methods to be validated here
        if ensemble_data is None:
            return

        if ensemble_data.ndim != 5:
            raise ValueError(
                f"ensemble_data must be (E,B,C,H,W), got {tuple(ensemble_data.shape)}"
            )

        E, B, C, H, W = ensemble_data.shape

        # Debug logging
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(
            f"EnsembleAggregator.record_batch: E={E}, B={B}, C={C}, H={H}, W={W}"
        )
        logger.debug(
            f"EnsembleAggregator.record_batch: ensemble_data min/max = {ensemble_data.min():.4f}/{ensemble_data.max():.4f}"
        )
        logger.debug(
            f"EnsembleAggregator.record_batch: target_data keys = {list(target_data.keys())}"
        )
        logger.debug(
            f"EnsembleAggregator.record_batch: var_to_channel = {self._var_to_channel}"
        )

        # Initialize member accumulator length once
        if self._ensemble_size is None:
            self._ensemble_size = E
            self._member_rmse_sum = [0.0] * E
        elif self._ensemble_size != E:
            raise ValueError(
                f"Inconsistent ensemble sizes across batches: {self._ensemble_size} vs {E}"
            )

        # Device/dtype alignment
        device = ensemble_data.device
        dtype = ensemble_data.dtype
        w_hw = self._area_weights.to(device=device, dtype=dtype)
        if w_hw.shape != (H, W):
            raise ValueError(
                f"area_weights must be (H,W)={H, W}, got {tuple(w_hw.shape)}"
            )

        # Normalize weights once
        w_sum = w_hw.sum().clamp_min(1e-12)
        w_hw = w_hw / w_sum  # (H, W) normalized

        # ---- 1) Per-variable batch metrics ----
        per_var_spread = []  # list of scalars
        per_var_rmse = []
        per_var_mae = []

        logger.debug(
            f"  Starting per-variable metrics, looping over {len(self._var_to_channel)} vars"
        )

        for var, ch in self._var_to_channel.items():
            if var not in target_data or var not in gen_data:
                # Skip if either is missing
                logger.debug(
                    f"    Skipping var '{var}' (ch={ch}): not in target_data={var not in target_data} or gen_data={var not in gen_data}"
                )
                continue
            if ch < 0 or ch >= C:
                logger.debug(
                    f"    Skipping var '{var}': channel {ch} out of range [0, {C})"
                )
                continue

            # Targets and predictions for this variable
            tgt_bhw = self._ensure_bc_hw(
                target_data[var].to(device=device, dtype=dtype)
            )  # (B, H, W)
            pred_mean_bhw = self._ensure_bc_hw(
                gen_data[var].to(device=device, dtype=dtype)
            )  # (B, H, W)

            # --- Spread: std over members for this channel ---
            # (E, B, H, W) -> std over E -> (B, H, W)
            ch_members = ensemble_data[:, :, ch, :, :]  # (E, B, H, W)
            ch_std_bhw = ch_members.std(dim=0, correction=0)  # (B, H, W)
            ch_spread_per_sample = self._weighted_spatial_mean(ch_std_bhw, w_hw)  # (B,)
            ch_spread_batch = ch_spread_per_sample.mean()  # scalar
            per_var_spread.append(ch_spread_batch)

            # --- RMSE (ensemble mean vs truth) ---
            diff_bhw = pred_mean_bhw - tgt_bhw
            mse_per_sample = self._weighted_spatial_mean(diff_bhw.pow(2), w_hw)  # (B,)
            rmse_batch = torch.sqrt(mse_per_sample.mean())  # scalar
            per_var_rmse.append(rmse_batch)

            # --- MAE (ensemble mean vs truth) ---
            mae_per_sample = self._weighted_spatial_mean(diff_bhw.abs(), w_hw)  # (B,)
            mae_batch = mae_per_sample.mean()  # scalar
            per_var_mae.append(mae_batch)

        # Aggregate across variables (only those successfully processed)
        if per_var_spread:
            batch_spread = torch.stack(per_var_spread).mean()
            batch_rmse = torch.stack(per_var_rmse).mean()
            batch_mae = torch.stack(per_var_mae).mean()
        else:
            # Nothing to aggregate for this batch
            batch_spread = torch.tensor(0.0, device=device, dtype=dtype)
            batch_rmse = torch.tensor(0.0, device=device, dtype=dtype)
            batch_mae = torch.tensor(0.0, device=device, dtype=dtype)

        # Spread-skill ratio averaged across variables (avoid div-by-zero)
        if per_var_spread and per_var_rmse:
            ss_vals = []
            for s, r in zip(per_var_spread, per_var_rmse):
                r_val = r.clamp_min(1e-12)
                ss_vals.append(s / r_val)
            batch_spread_skill = torch.stack(ss_vals).mean()
        else:
            batch_spread_skill = torch.tensor(0.0, device=device, dtype=dtype)

        # Accumulate scalars as Python floats
        self._spread_sum += self._safe_scalar(batch_spread)
        self._rmse_sum += self._safe_scalar(batch_rmse)
        self._mae_sum += self._safe_scalar(batch_mae)
        self._spread_skill_sum += self._safe_scalar(batch_spread_skill)

        # ---- 2) Per-member RMSE (averaged across variables) ----
        # If you want per-var per-member, extend this section similarly.
        if per_var_rmse:
            # Build per-member per-variable RMSE, then average over variables.
            # member_ch_bhw: (B, H, W) for each var -> RMSE scalar, then average over vars.
            for e in range(E):
                member_rmse_vals = []
                for var, ch in self._var_to_channel.items():
                    if var not in target_data or ch < 0 or ch >= C:
                        continue
                    tgt_bhw = self._ensure_bc_hw(
                        target_data[var].to(device=device, dtype=dtype)
                    )  # (B, H, W)
                    member_ch_bhw = ensemble_data[e, :, ch, :, :]  # (B, H, W)
                    diff2 = (member_ch_bhw - tgt_bhw).pow(2)
                    mse_per_sample = self._weighted_spatial_mean(diff2, w_hw)  # (B,)
                    rmse_member = torch.sqrt(mse_per_sample.mean())
                    member_rmse_vals.append(rmse_member)

                    # Debug first iteration - ONLY ON RANK 0 to avoid multiple outputs
                    if e == 0 and len(member_rmse_vals) == 1:
                        from ocean_emulators.utils.distributed import is_main_process

                        if is_main_process():
                            import pdb

                            pdb.set_trace()
                            logger.info(
                                f"=== DETAILED DEBUG FOR FIRST VAR '{var}' (ch={ch}) ==="
                            )
                            logger.info(f"  tgt_bhw.shape: {tgt_bhw.shape}")
                            logger.info(
                                f"  tgt_bhw all finite? {torch.isfinite(tgt_bhw).all().item()}"
                            )
                            logger.info(
                                f"  tgt_bhw finite count: {torch.isfinite(tgt_bhw).sum().item()}/{tgt_bhw.numel()}"
                            )
                            if torch.isfinite(tgt_bhw).any():
                                logger.info(
                                    f"  tgt_bhw[finite] min/max: {tgt_bhw[torch.isfinite(tgt_bhw)].min().item():.6f} / {tgt_bhw[torch.isfinite(tgt_bhw)].max().item():.6f}"
                                )

                            logger.info(f"  member_ch_bhw.shape: {member_ch_bhw.shape}")
                            logger.info(
                                f"  member_ch_bhw all finite? {torch.isfinite(member_ch_bhw).all().item()}"
                            )
                            logger.info(
                                f"  member_ch_bhw finite count: {torch.isfinite(member_ch_bhw).sum().item()}/{member_ch_bhw.numel()}"
                            )
                            if torch.isfinite(member_ch_bhw).any():
                                logger.info(
                                    f"  member_ch_bhw[finite] min/max: {member_ch_bhw[torch.isfinite(member_ch_bhw)].min().item():.6f} / {member_ch_bhw[torch.isfinite(member_ch_bhw)].max().item():.6f}"
                                )

                            logger.info(
                                f"  diff2 finite count: {torch.isfinite(diff2).sum().item()}/{diff2.numel()}"
                            )
                            if torch.isfinite(diff2).any():
                                logger.info(
                                    f"  diff2[finite] min/max: {diff2[torch.isfinite(diff2)].min().item():.6f} / {diff2[torch.isfinite(diff2)].max().item():.6f}"
                                )

                            logger.info(f"  w_hw.sum(): {w_hw.sum().item()}")
                            logger.info(
                                f"  w_hw nonzero count: {(w_hw > 0).sum().item()}/{w_hw.numel()}"
                            )
                            logger.info(f"  mse_per_sample: {mse_per_sample}")
                            logger.info(f"  rmse_member: {rmse_member.item():.6f}")
                            logger.info(f"=== END DEBUG ===")

                if member_rmse_vals:
                    member_rmse_mean = torch.stack(member_rmse_vals).mean()
                    self._member_rmse_sum[e] += self._safe_scalar(member_rmse_mean)
                    logger.debug(
                        f"  Member {e}: computed {len(member_rmse_vals)} vars, mean RMSE = {member_rmse_mean.item():.6f}"
                    )
                else:
                    # No variables, keep as 0 increment
                    logger.warning(
                        f"  Member {e}: NO variables matched for per-member RMSE!"
                    )

        self._n_batches += 1

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Return ensemble statistics as a dict of WandB-friendly scalars.
        Values are averaged over batches and (where applicable) over variables,
        with cross-rank means computed via all_reduce_mean.
        """
        if self._n_batches == 0:
            return {}

        nb = float(self._n_batches)
        logs: dict[str, float] = {}

        # Use the same device as area_weights for all_reduce operations
        device = self._area_weights.device

        # Reduce means across ranks (scalar all-reduce)
        mean_spread = self._spread_sum / nb
        mean_spread = (
            all_reduce_mean(torch.tensor(mean_spread, device=device)).cpu().item()
        )
        logs[f"{label}/ensemble_spread"] = mean_spread

        mean_rmse = self._rmse_sum / nb
        mean_rmse = all_reduce_mean(torch.tensor(mean_rmse, device=device)).cpu().item()
        logs[f"{label}/ensemble_mean_rmse"] = mean_rmse

        mean_mae = self._mae_sum / nb
        mean_mae = all_reduce_mean(torch.tensor(mean_mae, device=device)).cpu().item()
        logs[f"{label}/ensemble_mean_mae"] = mean_mae

        mean_ss = self._spread_skill_sum / nb
        mean_ss = all_reduce_mean(torch.tensor(mean_ss, device=device)).cpu().item()
        logs[f"{label}/spread_skill_ratio"] = mean_ss

        # Per-member RMSE (averaged over variables and batches)
        if self._member_rmse_sum:
            # Per-member mean over batches (local)
            local_member_means = torch.tensor(
                [rmse_sum / nb for rmse_sum in self._member_rmse_sum],
                dtype=torch.float32,
                device=self._area_weights.device,
            )

            # Cross-rank averaging of each member’s mean (same E across ranks)
            # Note: This assumes the same E and member alignment across ranks.
            # If members differ per rank, switch to all_gather and concat.
            global_member_means = []
            for idx in range(len(self._member_rmse_sum)):
                m = all_reduce_mean(local_member_means[idx])
                logs[f"{label}/member_{idx}_rmse"] = float(m.cpu().item())
                global_member_means.append(float(m.cpu().item()))

            # Global best/worst (optionally gather across ranks for strict global)
            if (
                self._compute_global_best_worst
                and dist.is_available()
                and dist.is_initialized()
            ):
                # Gather per-rank lists to compute true global min/max
                tensor = torch.tensor(
                    global_member_means,
                    device=self._area_weights.device,
                    dtype=torch.float32,
                )
                # all_gather expects equal shapes; concatenating along a new dim
                gather_list = [
                    torch.zeros_like(tensor) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gather_list, tensor)
                stacked = torch.stack(gather_list, dim=0)  # (world_size, E)
                global_vals = (
                    stacked.mean(dim=0).cpu().tolist()
                )  # average of ranks (already means), safe
                logs[f"{label}/best_member_rmse"] = min(global_vals)
                logs[f"{label}/worst_member_rmse"] = max(global_vals)
            else:
                # Fallback to local min/max over already reduced-per-member means
                logs[f"{label}/best_member_rmse"] = min(global_member_means)
                logs[f"{label}/worst_member_rmse"] = max(global_member_means)

        return logs
