// SPDX-FileCopyrightText: 2026 Samudra Authors
//
// SPDX-License-Identifier: Apache-2.0

use std::{env, path::PathBuf, sync::Arc};

use anyhow::{bail, Context};
use zarrs::{
    array::Array,
    array_subset::ArraySubset,
    filesystem::FilesystemStore,
    storage::{ReadableWritableListableStorage, ReadableWritableListableStorageTraits},
};

type OpenArray = Array<dyn ReadableWritableListableStorageTraits>;

fn read_subset(store: ReadableWritableListableStorage, name: &str) -> anyhow::Result<Vec<f32>> {
    let array: OpenArray =
        Array::open(store, &format!("/{name}")).with_context(|| format!("opening {name}"))?;
    let dimension_names = array
        .dimension_names()
        .as_ref()
        .with_context(|| format!("{name} has no Zarr v3 dimension names"))?;
    let dimensions = dimension_names
        .iter()
        .map(|dimension| dimension.as_deref().context("unnamed dimension"))
        .collect::<anyhow::Result<Vec<_>>>()?;
    if dimensions != ["time", "k", "j", "i"] {
        bail!("{name} has unexpected dimensions {dimensions:?}");
    }
    let subset = ArraySubset::new_with_start_shape(vec![0, 0, 68, 68], vec![1, 4, 80, 80])?;
    array
        .retrieve_array_subset_elements::<f32>(&subset)
        .with_context(|| format!("reading {name}"))
}

fn expected(k: usize, y: usize, x: usize) -> f32 {
    ((k * 216 + y) * 216 + x) as f32
}

fn main() -> anyhow::Result<()> {
    let path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: llc-zarr-rust-compatibility-spike STORE_PATH")?;
    let store: ReadableWritableListableStorage = Arc::new(
        FilesystemStore::new(&path).with_context(|| format!("opening {}", path.display()))?,
    );

    let blosc = read_subset(store.clone(), "blosc")?;
    let zstd = read_subset(store, "zstd")?;
    if blosc != zstd {
        bail!("Blosc and core Zstd arrays decoded differently");
    }
    if blosc.len() != 4 * 80 * 80
        || blosc[0] != expected(0, 68, 68)
        || *blosc.last().context("empty result")? != expected(3, 147, 147)
    {
        bail!("decoded subset did not match the fixture");
    }

    println!("read {} values from each sharded array", blosc.len());
    println!("Blosc Zstd + bitshuffle: OK");
    println!("Zarr v3 core Zstd: OK");
    Ok(())
}
