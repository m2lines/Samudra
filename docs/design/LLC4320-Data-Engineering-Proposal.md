<!--
SPDX-FileCopyrightText: 2026 Alexander Merose

SPDX-License-Identifier: CC-BY-4.0
-->

# LLC4320 Data Engineering Proposal

[Alexander Merose](mailto:alex@openathena.ai) Created: Aug 25, 2026 Updated: Sep 2, 2026

Claude Q\&A [here](https://claude.ai/share/17518ac3-cde4-4bec-8f10-20622ddfe02b).

# Problem

Loading the LLC data as it currently is stored on the MIT compute cluster is very slow. The root cause is that the Zarr store for the LLC dataset is chunked sub-optimally for data loading. There are two types of suboptional chunking and storage:

1. Boundary chunking mismatch. While the prognostic data is spatially chunked for each face at 720x720 tiles, the *boundary* dataset is chunked *per face*, i.e in 4320x4320 chunks.
2. Halo exchange. Each 720x720 prognostic tile (and eventually, the boundary tiles) requires a halo of additional data needed for blending forecasts during training and inference. This means that in the worst case, we will need to load nine 720x720 tiles to get the full spatial receptive field.

The cause for data loading slowness for both these chunking schemes is sub-optimal data movement. The bytes on disk are arranged such that we have to load way more than we actually need to use into memory just to throw most of the data away.

# Constraints

Here are a few constraints for the solution space. These are constraints on how we might perform the data engineering of the LLC dataset on MIT’s cluster.

* Minimize copying bytes. If we can get away with not modifying the \~90% of the data that is already in the right form (i.e. the 720x720 prognostic data), we’ll be better for it.
* Modify the data in place, if possible. At the end, we want to consume about the same disk storage space in bytes as we do today, just with a better arrangement. If we can minimize intermediary stores, we’ll be better for it.
* Fast-enough data loading. We may want to trade off tunability of our data stores with perfect loading optimality for our current modeling arrangement. It’s possible our modeling approach will change in predictable ways, but data loading should still be fast enough to keep GPU utilization saturated.

# Plan

1. Ingest the existing LLC dataset into [IceChunk](https://icechunk.io/) via a Virtual Zarr store.
   1. This gives the LLC dataset version control capability without copying bytes. It also provides a way to efficiently store new chunking schemes into the Zarr “repo”.
2. Rechunk the boundary data in place on the IceChunk store.
   1. In practice, with IceChunk, this would involve three phases of operations: First, we’d write the metadata of the new chunks to the IceChunk repo, then we’d make a distributed write of all of the new 720x720 boundary chunks, last we’d merge and commit the chunking change per face or group of faces so each snapshot would be clean/consistent.
3. Create a supplementary halo store and represent the LLC dataset with an Xarray DataTree
   1. Here, we’d load each 720x720 tile in an embarrassingly parallel way and capture the internal halo of that tile of size \`h\`.
   2. We’d then store a sidecar Zarr store with LLC that shares the dimensions of the primary LLC store but also has \`(..., n\_tiles, n\_ring)\`. We’d store one tile per chunk. This will make it easy to reconstruct the halos.
   3. (Optional) reorganize/reconstruct the halo per tile from the internal halo shards from step (b).
   4. Choose halo size \`h\` to be towards the largest size we’re ever likely to need.
   5. Update the IceChunk repo of the LLC dataset to have something like the following \`DataTree\` layout:

```
/prognostic          → virtual refs to source          (0 bytes)
/boundary            → native, rechunked 720×720       (see OQ-1)
/halo/prognostic     → native, ring-shaped
/halo/boundary       → native, ring-shaped
```
