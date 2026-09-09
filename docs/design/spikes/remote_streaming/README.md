<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# LLC4320 remote-streaming spikes

These scripts test a bounded network path from a scheduled Empire AI CPU node to a
real compressed LLC shard served by a scheduled MIT CPU node. They do not
modify the LLC source. The temporary server accepts authenticated byte ranges
for one immutable scratch object and exits after eleven minutes.

This direct compute-node test is a connectivity and transport spike, not the
proposed production service. A production origin requires an ORCD-approved
HTTPS or S3-compatible endpoint, TLS, monitoring, rate limits, and durable
storage.

The direct MIT compute-node server is useful only as a reachability diagnostic.
Its private address may be unroutable from Empire AI, and a successful result
would still not validate a production origin. Use `ssh mit` for the source and
`ssh eai` for the destination; Torch is intentionally out of scope.

## Focused layout tournament

`layout_tournament.py` builds candidates A--E from the same retained, real LLC
`Theta` fixture. It validates every decoded value and measures aligned cores,
16- and 32-cell halos, the four-patch union, four independent tiles, a
misaligned core, a shallow-depth crop, a point time series, and aligned versus
offset adjacent-time pairs. Its tracking store records physical objects,
range calls, and returned bytes below Zarr.

Zarr 3.3 has synchronous and asynchronous partial-read paths. The tracker
intercepts both. `retrace_layout_tournament.py` exists so instrumentation can
be corrected or extended without re-encoding multi-GiB candidates.

`layout_tournament.slurm` limits the five-element array to two concurrent jobs
to avoid imposing an unbounded read/write burst on ORCD scratch.
`summarize_layout_tournament.slurm` runs only after all five candidates finish
successfully. The summary preserves A as the decision prior; it does not choose
a winner from one scalar timing.

`native_reader_tournament.py` repeats the identical correctness and workload
suite with two independent native implementations: the Rust-backed
`zarrs-python` codec pipeline under the ordinary Zarr API, and TensorStore's
C++ Zarr v3 driver. Both consume the already encoded A--E stores, preventing
reader comparisons from being confounded by different source values or
encodings.

The completed CSV tables, raw JSON, exact candidate metadata, limitations, and
ORCD job provenance are in [`results/2026-09-08`](results/2026-09-08/README.md).
