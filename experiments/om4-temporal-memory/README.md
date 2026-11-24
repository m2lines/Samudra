# OM4 temporal-memory experiments

Experiment/results record, not intended to merge.

- **[Four-arm training comparison and final report](results/comparison/REPORT.md)**: full gradients, detached physical feedback, persistent latents, and temporal gradients only through latents.
- **[Global maps, quiet-region cutouts, and epoch-70 EMA comparison](results/checkpoint-comparison/REPORT.md)**.
- [Official checkpoint intervention and gradient diagnostics](results/diagnostics/REPORT.md).
- [Single-example overfitting report](results/overfit-one/REPORT.md).
- [Experiment protocol](PROTOCOL.md) and [execution history](RUNS.md).

Reports include figures, metrics, and provenance. Scripts and archived model code accompany the results. Large raw arrays and model checkpoints are not committed. Cluster submission scripts and paths document the historical executions; see the execution history for subsequent storage migrations. Run outputs belong in personal scratch or pool storage, while shared data remains under `/orcd/data/abodner/002/`.

The four-arm result is a single paired training seed and six held-out origins, not a replicated estimate across training seeds. See the final report for checkpoint selection, metric definitions, and scientific limitations.
