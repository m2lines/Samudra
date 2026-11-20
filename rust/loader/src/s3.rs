// SPDX-FileCopyrightText: 2026 Samudra Authors
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use std::{
    future::Future,
    sync::{Arc, OnceLock},
    time::Duration,
};
use zarrs::storage::{
    ReadableWritableListableStorage,
    storage_adapter::async_to_sync::{AsyncToSyncBlockOn, AsyncToSyncStorageAdapter},
};
use zarrs_object_store::{
    AsyncObjectStore,
    object_store::{RetryConfig, aws::AmazonS3Builder, prefix::PrefixStore},
};

struct BlockOn(Arc<tokio::runtime::Runtime>);
impl AsyncToSyncBlockOn for BlockOn {
    fn block_on<F: Future>(&self, future: F) -> F::Output {
        self.0.block_on(future)
    }
}

pub fn open(
    bucket: &str,
    prefix: &str,
    endpoint: Option<&str>,
    anon: bool,
) -> anyhow::Result<ReadableWritableListableStorage> {
    // Persistent process-wide I/O runtime; Rayon still bounds concurrent reads.
    static RUNTIME: OnceLock<Arc<tokio::runtime::Runtime>> = OnceLock::new();
    let runtime = if let Some(runtime) = RUNTIME.get() {
        runtime.clone()
    } else {
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()?,
        );
        let _ = RUNTIME.set(runtime);
        RUNTIME.get().expect("runtime initialized").clone()
    };
    let mut builder = AmazonS3Builder::from_env()
        .with_bucket_name(bucket)
        .with_skip_signature(anon)
        .with_retry(RetryConfig {
            max_retries: 3,
            retry_timeout: Duration::from_secs(120),
            ..Default::default()
        });
    if std::env::var_os("AWS_REGION").is_none() && std::env::var_os("AWS_DEFAULT_REGION").is_none()
    {
        builder = builder.with_region("us-east-1");
    }
    if let Some(endpoint) = endpoint {
        builder = builder
            .with_endpoint(endpoint)
            .with_virtual_hosted_style_request(false)
            .with_allow_http(endpoint.starts_with("http://"));
    }
    let store = builder.build().context("building S3 object store")?;
    let store = PrefixStore::new(store, prefix.trim_matches('/'));
    Ok(Arc::new(AsyncToSyncStorageAdapter::new(
        Arc::new(AsyncObjectStore::new(store)),
        BlockOn(runtime),
    )))
}
