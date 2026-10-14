<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Torch to OSN large-chunk loading test, 2026-10-14

The ten valid one-minute measurements reached 349.5 MB/s of compressed object payload and 893.1 MB/s of decoded arrays at ten concurrent reads. Eight reads reached 323.0 MB/s (92.4% of the highest observed result). These are single short measurements on shared CPU nodes, not replicated saturation estimates or an end-to-end training loader test.

## Workload

- Source: https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/raw/om4_5daily.zarr
- Random, distinct thetao/so/uo/vo objects. One chunk covers one timestep and all 19 depths on the native 1080 x 1440 staggered grid: approximately 118.2 MB decoded; compressed size varies by variable and timestamp. Full HTTPS GET into RAM and Blosc decode; no payload written to disk. No application data cache, but OSN/server/network caches were not controlled.
- Stored (on-disk) chunk sizes in the 2,851 valid workload reads: **median 45.2 MB**, interquartile range **33.1-53.4 MB**, and full range **31.4-54.9 MB** (decimal MB). These are compressed object sizes for the measured sample, not a census of the whole store.
- One HTTPS connection per reader and one Blosc thread per reader. Each phase had one excluded warm-up per reader, then 60 seconds admitting new loads and a short final drain. Reported bandwidth is payload bytes divided by the full phase wall time, including decoding and final drain. MB is decimal.
- Load latency includes GET and decode. First-byte latency is measured through the first body byte. The ten-CPU, 8-GB allocation stayed fixed as read concurrency changed.

## Results

| In-flight loads | Wire MB/s | Decoded MB/s | Load p50 (s) | Load p95 (s) | First byte p50 (ms) | Requests | Node |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 48.7 | 128.7 | 0.905 | 1.453 | 70.7 | 66 | cs604 |
| 2 | 96.2 | 240.2 | 0.951 | 1.558 | 67.1 | 124 | cs604 |
| 3 | 141.1 | 370.1 | 0.957 | 1.551 | 68.5 | 191 | cs604 |
| 4 | 177.9 | 465.1 | 1.024 | 1.598 | 67.7 | 239 | cs604 |
| 5 | 202.1 | 520.1 | 1.103 | 1.861 | 70.0 | 269 | cs604 |
| 6 | 220.9 | 556.7 | 1.254 | 1.799 | 69.9 | 289 | cs604 |
| 7 | 267.4 | 686.2 | 1.143 | 1.950 | 70.6 | 357 | cs604 |
| 8 | 323.0 | 821.4 | 1.077 | 1.898 | 74.1 | 427 | cs630 |
| 9 | 319.8 | 807.1 | 1.276 | 2.002 | 71.2 | 423 | cs630 |
| 10 | 349.5 | 893.1 | 1.254 | 2.228 | 73.4 | 466 | cs630 |

The valid rows contain 2,851 distinct successful object reads, 131.843 GB of payload and 337.105 GB of decoded arrays, with zero request/decode errors and zero timed reconnects. First-byte medians were 67-74 ms; median decode cost was approximately 64-72 ms.

## Recovery and limits

Job 19803626 ran on cs604. Its levels 1-7 passed. At level 8, six of eight readers failed immediately after warm-up with RemoteDisconnected; that entire phase is excluded. This does not demonstrate a concurrency saturation limit. The failure timing is consistent with connections closing during the warm-up barrier, but the server-side cause was not established.

Job 19803911 completed levels 8-10 on cs630 after adding at most one timed reconnect for closed connections. No measured requests actually needed a reconnect. It skipped the 1,691 shuffled keys already consumed by the first run, including warm-up and failed requests. The two nodes have the same reported 128-CPU, 513,000-MB resource shape; exact CPU model, co-tenant traffic and NIC performance were not measured. Changing node and time prevents treating the combined series as a strictly controlled same-node scaling curve.

Raw all-depth chunks are much larger than the prepared quarter-degree 4.15-MB objects. The 350-MB/s observation should not be assumed for that training store, or for normalization, tensor assembly, GPU transfer, or multiple GPU ranks sharing bandwidth. Eight in-flight object reads is a reasonable starting point for a future loader check, not a claim that eight DataLoader workers is optimal.

## Provenance and verification

Initial job: 19803626, FAILED after 8m23s; valid rows retained individually. Recovery: 19803911, COMPLETED exit 0 after 3m12s. Both requested cpu_short / torch_pr_347_lzanna, 10 CPUs, 8 GB, no GPUs. Completion callback jobs 19803637 and 19803929 both completed.

The metadata SHA-256 matches across runs. Local summary, raw request and recovery provenance checksums match Torch. Request counts, byte sums, elapsed times, bandwidth arithmetic, distinct keys, requested worker counts, zero errors and recovery COMPLETE.json were independently checked. Scripts, per-run provenance, metadata, raw request records and combined CSV/JSON are retained in this directory. See [run-metadata.json](run-metadata.json) for job identifiers and remote paths. Exact Python source snapshots are retained under [as-run/](as-run/); their SHA-256 values match each run's provenance file.

## Verify or reproduce

This is an experiment/results record and is not intended to merge. The source snapshots are kept byte-for-byte as `.py.txt` artifacts, including their original formatting, so their fingerprints remain verifiable. No training was run.

The original per-run CSV files are compressed losslessly as `*.csv.gz`, preserving their original bytes and line endings. The combined CSV is an uncompressed, derived table. `SHA256SUMS` covers all packaged files.

Verify the archived results locally with standard-library Python:

```bash
python3 experiments/2026-10-14-torch-osn-loading/verify.py
```

The corrected benchmark requires `numcodecs` and reads the public store without credentials. Running the following command starts a new ten-minute data-loading test:

```bash
python3 experiments/2026-10-14-torch-osn-loading/as-run/benchmark_recovery.py.txt \
  --out /tmp/torch-osn-new-results
```

The archived [Slurm harness](as-run/run.sbatch) records the exact container SIF path and resource request used for recovery. On Torch, copy `benchmark_recovery.py.txt` to its configured `$ROOT/benchmark.py` path before submitting the harness. Its default is a full 1-10 sweep; the recovery job used `START_CONCURRENCY=8 SKIP_KEYS=1691`. Use a fresh output location for a new run. The initial failed eight-reader result is retained in its original run directory but excluded from both combined summaries.
