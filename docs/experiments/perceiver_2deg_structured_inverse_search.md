<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Structured Perceiver inverse search at 2°

## Questions and hypotheses

This search asks which single-scale encoder/decoder best combines Samudra's
native-SDPA Perceiver routing with the strongest findings from Jesse's coarse
latent/high-resolution dynamics experiments.

1. Does an explicit area-weighted mean plus phase-sensitive moment state retain
   more forecast-relevant detail than pooled or spatial-query Perceiver states?
2. Was adding absolute geometry directly to moment content harmful, compared
   with keeping routing geometry separate?
3. Does a smooth coordinate-resampled base plus zero-initialized,
   position-anchored local SDPA residual improve loss, seams, spatial gradients,
   and rollout RMSE relative to direct output cross-attention?
4. How much moment capacity is needed, and does a denser spatial query grid
   provide a better capacity tradeoff?

The main hypothesis is that `moment16-local` will provide the strongest
combination: the encoder guarantees a resolved route and retains subpatch phase,
while the decoder begins from a smooth coarse field and learns only a local
continuous correction. The search also includes causal encoder-only and
decoder-only comparisons, the strongest prior spatial-grid decoder, a paired DCT
route, and the historical pooled Perceiver control.

## Interventions

All ten candidates share the same 2° OM4 split, seed, 60×72-or-finer processor
token budget, ConvNeXt processor, residual target, optimizer, and 1/3/6/12 epoch
successive-halving schedule. Promotion uses validation loss. Each rung endpoint
also runs the configured autoregressive validation rollout; rollout RMSE,
velocity RMSE, gradient-magnitude fidelity, and seam diagnostics are retained for
scientific review and Pareto analysis rather than collapsed prematurely into one
score.

Encoder group extents are chosen to make the processor grid comparable rather
than naively held equal: 2×2 spatial-query groups cover 12°×20°, 4×4 groups
cover 24°×40°, and single-token moment, DCT, and pooled routes cover 6°×10°.
At 2° all three choices produce a 30×36 token grid.

Each Alpha worker uses two GPUs with DDP. This resource shape had an immediate
`test`-QoS scheduler estimate on launch day, while Alpha's one-GPU RTX pool was
backlogged until the following day; it also keeps both allocated GPUs doing
model work rather than reserving idle capacity.
Gradient accumulation is 16, preserving the prior effective batch of 32
(`batch_size=1 × accumulation=16 × world_size=2`).

Live scheduler probes showed that CPU shape, rather than GPU or memory demand,
controlled placement on the partially occupied H100 nodes: four CPUs with 28
GiB could backfill immediately, whereas 6--12 CPUs moved the same two-GPU job
behind a later reservation. The final search therefore uses two loader workers,
four CPUs, and 28 GiB per two-GPU candidate.

The candidate matrix is defined in
[`search.yaml`](../../src/samudra/configs/perceiver_structured_inverse_2deg/search.yaml).

### Torch relaunch revision after prior-evidence review

Before the Torch relaunch, the Perceiver lab notebooks and Jesse's decoder and
coarse-latent reports were reviewed together. Two original Alpha candidates are
not repeated in the Torch matrix:

- `dct14-paired` is excluded because its low short-budget MSE coincided with
  4.15 times the baseline patch-periodic power and visibly basis-aligned,
  unphysical error texture. It remains mechanistic evidence that spatial phase
  matters, not a promotion-eligible architecture.
- `moment16-geometry-direct` is excluded because adding absolute geometry to
  reconstructive content conflicts with the stronger learned-inverse result:
  keep amplitude-bearing state separate and inject geometry as a
  zero-initialized processor sidecar.

The revised Torch search therefore contains eight candidates: the pooled
Perceiver baseline; spatial-grid 2x2 direct and local-decoder arms; a denser
spatial-grid direct arm; moment-4 and moment-16 local arms; and moment-16 direct
width-128 and width-256 arms. On a four-GPU allocation this produces an exact
8 -> 4 -> 2 -> 2 occupancy schedule: four concurrent one-GPU candidates in the
first two rungs and two concurrent two-GPU candidates in the last two rungs.
The effective global batch remains 32 throughout.

The utilization smoke is a systems gate rather than a scientific comparison.
It crosses the two leading phase-preserving encoders (`spatial_grid` and
`patch_moment`) with direct and physically anchored local decoders. A full
search will not be released unless all four arms make finite optimizer progress
and every allocated GPU sustains more than 50% utilization during steady-state
training. Validation loss from this shortened data window will not select an
architecture.

The first Torch utilization smoke,
`perceiver-2deg-batch16-utilization-smoke--20260904T180756.742480Z`, ran as
Slurm job `16962858` on four RTX6000s with 32 CPUs and 200 GiB of host memory.
Every worker used local batch 16, accumulation 2, and effective global batch
32; all four workers reached two finite batches and one optimizer update before
the allocation was deliberately cancelled. W&B runs and the
[public OSN record](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/experiments/searches/perceiver-2deg-batch16-utilization-smoke--20260904T180756.742480Z/)
were created successfully.

This first systems gate failed. Initial data wait was approximately 106 seconds
per candidate. After one prefetched batch, the next pair arrived only after
34--38 seconds; per-batch load time remained 49--56 seconds while model compute
took approximately 0.5--3.1 seconds. Both one-second and 100-millisecond device
sampling therefore observed nearly all GPUs idle between short compute bursts.
The two-worker loader, not GPU memory or arithmetic, was the immediate
bottleneck. Peak allocated GPU memory was approximately 18.3 GiB for
`moment16-direct`, 30.3 GiB for `moment16-local`, 37.8 GiB for
`spatial-grid2-direct`, and 49.8 GiB for `spatial-grid2-local`; batch 16 fits,
but the last arm has materially less memory headroom.

The follow-up smoke increased persistent CPU loader workers from two to eight
per candidate while retaining the same model batch and optimizer batch. Slurm
job `16962966` completed successfully in 13 minutes 23 seconds. All four workers
completed 30 batches and 15 optimizer updates, wrote checkpoints, finished
validation, logged to W&B, and published the
[public OSN record](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/experiments/searches/perceiver-2deg-batch16-utilization-smoke--20260904T223204.477529Z/).

The additional workers improved throughput but did not pass the utilization
gate. Training took 416--433 seconds per candidate. CUDA-synchronized model
batch time (`progress/gpu_seconds`) divided by training time gives an upper
bound on compute duty cycle of 35.7% for `spatial-grid2-direct`, 27.2% for
`spatial-grid2-local`, 7.1% for `moment16-local`, and 29.5% for
`moment16-direct`. W&B's lower-frequency NVML samples likewise show long idle
periods between occasional utilization spikes. Data-load measurements remained
approximately 42--50 seconds per batch; eight workers converted the earlier
two-batch bursts into longer runs of ready batches, but still produced
12--41-second stalls when the prefetch queue emptied.

Two controlled follow-ups tested alternatives to the shared-filesystem layout.
Direct anonymous streaming from the public OSN Zarr store in Torch job
`16975829` did not produce a first batch in more than three minutes and sampled
zero GPU work, so it was deliberately cancelled. Individual object reads were
responsive, but one training sample requires many time-one, variable-per-object
requests; Xarray/Zarr support for remote stores does not by itself make that
access pattern efficient. The cancellation also reproduced a search-harness
correctness defect: the terminated worker left the durable search state marked
`running` with no results. This is recorded as a merge-blocking issue on the
resource-aware local-executor PR.

A compact representation that stacked depth levels into fewer logical variables
was tested in job `16977817`. It likewise failed to reach a first batch promptly
and was cancelled. The current canonical reader's indexing and conversion path
does not benefit from that representation, so compactness at the object level is
not a sufficient optimization criterion.

The final probe staged an eight-year slice of the *unchanged flat-channel Zarr
layout* once onto the allocation node's local SSD before starting the candidate.
Torch job `16978214`, search run
`perceiver-2deg-node-local-utilization-probe--20260904T235931.143524Z`, used the
light `moment16-local` path on one RTX PRO 6000 Blackwell GPU with batch size 16,
accumulation 2, eight persistent loader workers, eight CPUs, and 64 GiB. It
completed 30 training batches, 15 optimizer updates, validation, W&B run
[`bd8emx91`](https://wandb.ai/ocean_emulators/default/runs/bd8emx91), and the
[public OSN record](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/experiments/searches/perceiver-2deg-node-local-utilization-probe--20260904T235931.143524Z/).

This probe passed the steady-state utilization gate. After the first batch,
data-wait time was normally 0.03--0.22 seconds and batch time was normally
0.53--0.77 seconds. One-second NVML samples across the 30-batch training interval
averaged 78% SM utilization, with most samples between 75% and 99%. The model
therefore has enough arithmetic intensity at batch 16 when data are local, and
eight workers per GPU are sufficient for this store and reader.

The distinction between steady-state and allocation-average utilization remains
important. Train and validation each spent about 200 seconds waiting for their
first batch because eight spawned loader processes imported Python and packages
from the shared scratch environment. The entire job took 11 minutes 50 seconds,
while steady training took only about 18 seconds after startup. The full search
should therefore (1) stage the unchanged store once per allocation, (2) use a
published container/code layer or otherwise avoid spawning workers from a
metadata-congested shared environment, (3) retain persistent workers, and (4)
make rung zero long enough to amortize unavoidable startup. The search runner is
useful for a fixed one-rung matrix of systems hypotheses, but promotion should
remain based on scientific loss or rollout metrics rather than utilization.

## Launch record

The accepted immutable run is
`perceiver-structured-inverse-2deg--20260831T191355.834091Z` at commit
`5a7fc7716df710c162f2b6cdfab0b4413ea4f772`. The W&B group is the same run ID.

- Alpha launch controller: `56021` (`COMPLETED`)
- two-GPU DDP correctness probe: `56023_0` (`COMPLETED`)
- probe-release controller: `56024` (`COMPLETED`)
- rung-zero candidate array: `56030` (tasks 0--9, concurrency 4)
- rung-zero advancement controller: `56031` (`afterany:56030`)

The probe loaded the public 2° data, built the complete model, ran both DDP
ranks, consumed 16 microbatches, and recorded one real optimizer update with a
finite loss of 2.3714. Its steady-state microbatch time was approximately
0.6--1.5 seconds and peak memory approximately 3.4 GiB per GPU. Rung zero was
released only after this evidence was durable.

At the first status check, array tasks 0--2 were concurrently training
`spatial-grid2-direct`, `moment16-geometry-direct`, and `moment16-direct`.
All three recorded a finite first batch; the geometry-moment candidate had
already recorded two optimizer updates. Remaining tasks backfill as one of the
four concurrent slots becomes available.

### Torch adaptive-search relaunch

The resource-aware local executor was relaunched on Torch as Slurm job
`16987589`. The immutable search run is
`perceiver-structured-inverse-2deg-torch-adaptive--20260905T052520.357945Z` at
commit `e378223756705372ed23d330af1c56575fc81ebe`. Its durable
[OSN record](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/experiments/searches/perceiver-structured-inverse-2deg-torch-adaptive--20260905T052520.357945Z/)
was readable while rung zero was running.

The allocation has two RTX 6000 GPUs, 16 CPUs, and 128 GiB of host memory. The
launcher stages the unchanged flat-channel 2-degree Zarr store once to local
SSD, then runs two one-GPU candidates concurrently. Each candidate uses local
batch 16, accumulation 2, effective global batch 32, and six persistent loader
workers. The first two candidates reached real optimizer updates with finite
losses:

- [`spatial-grid2-direct`](https://wandb.ai/ocean_emulators/default/runs/4b7nkuox):
  first recorded optimizer-step loss 2.4492;
- [`moment16-direct`](https://wandb.ai/ocean_emulators/default/runs/qhp876rh):
  first recorded optimizer-step loss 2.1223.

Peak allocation RSS after loader startup was approximately 103 GiB, below the
128 GiB cgroup limit; reducing from eight to six workers per candidate avoided
the loader-start OOM seen in the preceding infrastructure attempt. Warm
data-wait time was approximately 0.03--0.16 seconds, confirming that node-local
staging removed the storage bottleneck. Nevertheless, the direct decoder arms
showed bursty device work: the first 80 one-second steady-state samples averaged
13.3% and 16.5% SM utilization. This is substantially lower than the 78% seen
for the lighter `moment16-local` utilization probe despite negligible data
wait. It therefore points to host-side/model execution gaps in these direct
decoder paths rather than insufficient loader throughput. The search remains
useful scientifically, but this distinction should inform both interpretation
and the next performance intervention.

Several failed launch attempts before `16987589` produced no scientific data.
They exposed two deployment constraints: Torch effectively limits this
partition to roughly 64 GiB of host memory per requested GPU, and a full home
quota prevents Slurm from opening its output file, which presents as immediate
signal 53. Superseded smoke artifacts were preserved under
`/archive/am16581/search-smokes`, restoring home space before the accepted
launch.

## Results

The Torch allocation ended after 34 minutes 23 seconds before rung zero could
finish. Four candidates nevertheless completed all 176 training batches, 88
optimizer updates, validation, and the configured 36-step autoregressive
rollout. Three were recorded as completed workers. `moment4-local` also wrote a
complete `training_summary.json`, but failed afterward while W&B was writing a
validation image, so it is reported as a recoverable result rather than an
accepted search result.

| Candidate | Validation loss | AR inference loss | Train loss | Window-periodic power ratio | Window-jump ratio | Time (min) | Worker outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| [`moment4-local`](https://wandb.ai/ocean_emulators/default/runs/es3yksca) | **0.073229** | 0.414186 | 0.809223 | **0.002358** | 0.945746 | **7.5** | Metrics complete; artifact write failed |
| [`moment16-local`](https://wandb.ai/ocean_emulators/default/runs/8levaldh) | 0.073652 | 0.412987 | 0.787923 | 0.002384 | **0.947076** | 8.1 | Completed |
| [`moment16-direct`](https://wandb.ai/ocean_emulators/default/runs/qhp876rh) | 0.074977 | **0.383711** | **0.649819** | 0.014533 | 1.069773 | 11.6 | Completed |
| [`spatial-grid2-direct`](https://wandb.ai/ocean_emulators/default/runs/4b7nkuox) | 0.074997 | 0.383923 | 0.655559 | 0.016021 | 0.867492 | 14.1 | Completed |

Lower is better for the three loss columns and window-periodic power. A
window-jump ratio of one is ideal, so the table should be read by distance from
one rather than by its raw minimum. Reported times include training, validation,
rollout, metrics, and artifact work for epoch one.

`spatial-grid2-local` initialized and reached one optimizer update before the
allocation failed. `spatial-grid3x4-direct`, `moment16-wide-direct`, and
`pooled-perceiver-direct` never started. Consequently, the rung-zero comparison
is incomplete and no candidate was promoted.

The proximate failure was home-directory quota exhaustion, not a non-finite
model result or GPU OOM. Each completed candidate retained approximately 3.2
GiB locally: `ckpt_1.pt`, `ckpt.pt`, `best_validation_ckpt.pt`, and
`best_inference_ckpt.pt` were each about 700 MiB, while `ema_ckpt.pt` was about
525 MiB. After four candidates had written these largely overlapping states,
`moment4-local` raised `Disk quota exceeded` while creating a W&B image.
Slurm job `16987589` then exited with code 1. The local executor did not commit
the already completed worker results to top-level `results.csv`, and
`state.json` incorrectly remained `running` with zero results and no
promotions. Thus the public OSN directory records provenance and initial state,
while the W&B runs and local worker summaries are the authoritative sources for
the preliminary numbers above.

## Analysis

The completed subset exposes a real decoder tradeoff, but it does not yet name a
winner. The two local-decoder arms improved one-epoch validation loss by
approximately 1.8--2.4% relative to the direct arms. They also reduced the
channel-mean window-periodic power ratio by approximately 84--85%, and their
window-jump ratios were closer to one than either spatial direct result and
slightly closer than moment direct. This supports the hypothesis that a smooth
base plus locally anchored correction suppresses patch-periodic structure.

The same local arms were approximately 7.0--7.4% worse on the 36-step
autoregressive inference loss. Their training losses were also higher. The
direct decoder therefore appears easier to optimize and retains a meaningful
early rollout advantage, even though it leaves more periodic spatial error.
This is precisely why validation loss alone should not drive the architecture
decision: at epoch one it would favor `moment4-local`, whereas rollout loss
would favor `moment16-direct`.

Within the direct-decoder pair, `moment16-direct` and
`spatial-grid2-direct` are effectively tied: their validation losses differ by
0.03% and their inference losses by 0.06%. One epoch provides no evidence that
either encoder is superior. Within the local-decoder pair, reducing from 16 to
four moments improved validation loss by 0.6% and degraded inference loss by
only 0.3%. That weak early signal suggests the extra moment capacity may not be
earning its cost, but it is much too small to rule out divergence at later
rungs.

These conclusions are conditional on only four of eight candidates and one
epoch of optimization. The absent pooled control, wider direct decoder, and
denser spatial grid prevent several primary hypotheses from being answered.
The values should guide the clean rerun and Pareto analysis, not be treated as a
completed successive-halving result.

## Conclusion and future work

The strongest current inference is architectural rather than a winning model:
local structured decoding substantially improves seam behavior and slightly
improves one-step validation, while direct decoding gives better early
autoregressive fidelity. A high-value next intervention should try to combine
the local decoder's smooth spatial assembly with the direct decoder's easier
optimization and rollout behavior, rather than selecting either endpoint from
this incomplete rung.

Before resuming the scientific search:

1. Retain only the checkpoint required to promote or resume an early-rung
   candidate. Publish it with a checksum before pruning local redundant copies.
2. Reduce early-rung image frequency or generate the full visualization suite
   only for promoted candidates. Scalar rollout and seam metrics must remain
   enabled because they revealed the main tradeoff.
3. Write search outputs to a capacity-appropriate filesystem, or stream durable
   artifacts to OSN and reclaim local space after verified publication.
4. Make the controller atomically collect completed worker results even when a
   sibling fails, and transition top-level state to `failed` rather than leaving
   a stale `running` record.
5. Rerun the complete eight-candidate rung at the same immutable code and model
   settings before promotion. Reusing the four summaries is appropriate for
   diagnosis, but a clean, controller-recorded rung is preferable for a
   reproducible successive-halving decision.

After the corrected rung, promotion should consider validation loss, AR
inference loss, and the seam diagnostics as a small Pareto set. If the local
versus direct tradeoff persists through epochs three and six, the next model
search should test hybrid output assembly explicitly.
