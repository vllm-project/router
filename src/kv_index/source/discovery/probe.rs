//! Read worker model metadata and KV event source endpoints.

use std::collections::HashSet;

use tracing::{info, warn};

use super::{
    discovered_block_size, metrics_meta, CacheCapabilities, KvEventSourceEntry,
    KvEventSourcesResponse, KvIndexSupervisor, SourcesProbe, SourcesUnavailable, WorkerDiscovery,
    CONTROL_TIMEOUT,
};
use crate::core::worker::{served_base_model_ids, worker_get};
use crate::protocols::worker_spec::ServerInfo;

pub(super) struct RegistrationProbe {
    pub(super) descriptor: WorkerDescriptor,
    pub(super) base_models: HashSet<String>,
    pub(super) worker: WorkerDiscovery,
}

#[derive(PartialEq)]
pub(super) struct WorkerDescriptor {
    server_info: ServerInfo,
    sources: Option<KvEventSourcesResponse>,
    model_id: Result<Option<String>, String>,
    hybrid: Option<CacheCapabilities>,
}

pub(super) fn worker_descriptor(worker: &WorkerDiscovery) -> Result<WorkerDescriptor, String> {
    let server_info = worker.server_info.as_ref().map_err(Clone::clone)?.clone();
    let sources = match &worker.sources {
        Ok(sources) => Some(sources.clone()),
        Err(unavailable) if !unavailable.probe.retryable() => None,
        Err(unavailable) => return Err(unavailable.detail.clone()),
    };
    Ok(WorkerDescriptor {
        server_info,
        sources,
        model_id: worker.model_id.clone(),
        hybrid: worker.hybrid.clone(),
    })
}

impl KvIndexSupervisor {
    pub(super) async fn probe_registration(&self, base: &str) -> Result<RegistrationProbe, String> {
        let worker = self.discover(base).await;
        let descriptor = worker_descriptor(&worker)?;
        let base_models = if crate::backend::is_grpc_url(base) {
            crate::backend::control::get_grpc_model_info(base, CONTROL_TIMEOUT)
                .await
                .ok()
                .map(|info| info.model_id)
                .filter(|model| !model.trim().is_empty())
                .into_iter()
                .collect()
        } else {
            let request = worker_get(
                &self.client,
                &format!("{}/v1/models", base.trim_end_matches('/')),
                self.api_key.as_deref(),
            );
            let models = match request.send().await {
                Ok(response) if response.status().is_success() => {
                    response.json::<serde_json::Value>().await.ok()
                }
                _ => None,
            };
            models
                .as_ref()
                .into_iter()
                .flat_map(served_base_model_ids)
                .map(str::to_owned)
                .collect()
        };
        Ok(RegistrationProbe {
            descriptor,
            base_models,
            worker,
        })
    }
}

pub(super) fn grpc_kv_event_source_entries(
    worker_url: &str,
    sources: Vec<crate::backend::pb::KvEventSource>,
) -> KvEventSourcesResponse {
    sources
        .into_iter()
        .filter_map(|source| {
            let Some(rank) = source.data_parallel_rank else {
                warn!(
                    "kv_index {}: gRPC KV event source has no data_parallel_rank — skipping source",
                    worker_url
                );
                return None;
            };
            Some((
                rank.to_string(),
                KvEventSourceEntry {
                    endpoint: source.endpoint,
                    replay_endpoint: (!source.replay_endpoint.is_empty())
                        .then_some(source.replay_endpoint),
                    topic: source.topic,
                    hwm: i32::try_from(source.hwm).ok(),
                    buffer_steps: Some(u64::from(source.buffer_steps)),
                    publisher: Some(source.transport),
                    encoding: Some(source.encoding),
                    schema_version: Some(source.schema_version),
                    // The Control RPC has no per-source enable flag; it only
                    // returns sources when KV events are enabled.
                    enable_kv_cache_events: Some(true),
                },
            ))
        })
        .collect()
}

pub(super) fn server_model_id(server_info: &ServerInfo) -> Option<String> {
    server_info
        .model_id
        .clone()
        .filter(|id| !id.trim().is_empty())
        .or_else(|| {
            server_info
                .model_path
                .as_ref()
                .and_then(|path| path.split('/').next_back().map(str::to_string))
                .filter(|id| !id.trim().is_empty())
        })
}

pub(super) fn layout_model<'a>(
    info: &'a ServerInfo,
    served_model: &'a str,
    layout: &CacheCapabilities,
) -> &'a str {
    info.model_path
        .as_deref()
        .filter(|path| layout.ranks.iter().all(|rank| rank.model == *path))
        .unwrap_or(served_model)
}

pub(super) fn grpc_model_id(
    model: Result<crate::backend::pb::ModelInfo, String>,
) -> Result<Option<String>, String> {
    model.map(|model| {
        let model_id = model.model_id;
        (!model_id.trim().is_empty()).then_some(model_id)
    })
}

/// Fetch the HTTP KV event source route. A 404/405 reports
/// [`SourcesProbe::Unsupported`], not a transport failure.
pub(super) async fn fetch_kv_event_sources(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
) -> Result<KvEventSourcesResponse, SourcesUnavailable> {
    let url = format!("{}/kv_event_sources", worker_url.trim_end_matches('/'));
    let response = worker_get(client, &url, api_key)
        .send()
        .await
        .map_err(|error| SourcesUnavailable {
            probe: SourcesProbe::Failed,
            detail: format!("kv_event_sources {url}: {error}"),
        })?;
    let status = response.status();
    if matches!(status.as_u16(), 404 | 405) {
        return Err(SourcesUnavailable {
            probe: SourcesProbe::Unsupported,
            detail: format!("kv_event_sources {url}: {status}"),
        });
    }
    if !status.is_success() {
        return Err(SourcesUnavailable {
            probe: SourcesProbe::Failed,
            detail: format!("kv_event_sources {url}: status {status}"),
        });
    }
    response.json().await.map_err(|error| SourcesUnavailable {
        probe: SourcesProbe::Failed,
        detail: format!("kv_event_sources {url} parse: {error}"),
    })
}

/// Read the worker's resolved contract. The automatic path covers plain full
/// attention; connectors and LoRA keep their existing event handling.
pub(super) async fn fetch_native_full_layout(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
) -> Result<Option<CacheCapabilities>, String> {
    let body: serde_json::Value =
        query_worker_json(client, worker_url, "kv_routing_capabilities", true, api_key).await?;
    if !body
        .get("adapters")
        .and_then(serde_json::Value::as_array)
        .is_some_and(Vec::is_empty)
        || !body
            .get("ranks")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|ranks| {
                !ranks.is_empty()
                    && ranks
                        .iter()
                        .all(|rank| rank.get("transfer").is_some_and(serde_json::Value::is_null))
            })
    {
        return Ok(None);
    }
    let mut capability: CacheCapabilities =
        serde_json::from_value(body).map_err(|e| format!("kv_routing_capabilities parse: {e}"))?;
    for rank in &mut capability.ranks {
        let [group] = rank.groups.as_slice() else {
            return Ok(None);
        };
        if rank.declared_groups.is_some_and(|count| count != 1)
            || rank.hash_block_size != Some(group.block_size)
            || rank.hash_lookahead_tokens != 0
            || rank.speculative_tokens != 0
            || rank.partial_hash_hits
            || group.group_idx != 0
            || group.kind != "full_attention"
            || group.spec_type != "FullAttentionSpec"
            || !group.prefix_cacheable
            || group.prefix_replay_tokens != 0
        {
            return Ok(None);
        }
        // The discovery endpoint lists the live engine's complete topology.
        rank.declared_groups = Some(1);
    }
    capability.ranks.sort_unstable_by_key(|rank| rank.rank);
    Ok(Some(capability))
}

impl From<crate::backend::control::KvEventSourcesError> for SourcesUnavailable {
    fn from(error: crate::backend::control::KvEventSourcesError) -> Self {
        let probe = match &error {
            crate::backend::control::KvEventSourcesError::Unimplemented => {
                SourcesProbe::Unsupported
            }
            crate::backend::control::KvEventSourcesError::Other(_) => SourcesProbe::Failed,
        };
        Self {
            probe,
            detail: error.detail().to_owned(),
        }
    }
}

async fn query_worker_text(
    client: &reqwest::Client,
    worker_url: &str,
    path: &str,
    authenticated: bool,
    api_key: Option<&str>,
) -> Result<String, String> {
    let url = format!("{}/{path}", worker_url.trim_end_matches('/'));
    let request = if authenticated {
        worker_get(client, &url, api_key)
    } else {
        client.get(&url)
    };
    let response = request
        .send()
        .await
        .map_err(|e| format!("{path} {url}: {e}"))?;
    if !response.status().is_success() {
        return Err(format!("{path} {url}: status {}", response.status()));
    }
    response
        .text()
        .await
        .map_err(|e| format!("{path} body: {e}"))
}

async fn query_worker_json<T: serde::de::DeserializeOwned>(
    client: &reqwest::Client,
    worker_url: &str,
    path: &str,
    authenticated: bool,
    api_key: Option<&str>,
) -> Result<T, String> {
    let body = query_worker_text(client, worker_url, path, authenticated, api_key).await?;
    serde_json::from_str(&body).map_err(|e| format!("{path} parse: {e}"))
}

/// Worker cache metadata, tried in the order a deployment is likely to answer it:
/// the multi-backend control contract, the engine config endpoint, then the
/// always-on `/metrics` scrape.
///
/// Every source is optional; the error names what each one said, so an operator
/// can tell a missing endpoint from an endpoint that answered without a block
/// size.
pub(crate) async fn query_worker_metadata(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
) -> Result<ServerInfo, String> {
    let mut attempts: Vec<String> = Vec::new();

    // 1. Multi-backend control metadata. Absent from stock vLLM.
    match query_worker_json::<ServerInfo>(client, worker_url, "get_server_info", true, api_key)
        .await
    {
        Ok(info) if discovered_block_size(&info).is_some() => {
            info!(
                "kv_index {}: worker cache metadata from /get_server_info (block_size={})",
                worker_url,
                discovered_block_size(&info).unwrap_or_default()
            );
            return Ok(info);
        }
        Ok(_) => attempts.push("get_server_info: no block_size".to_string()),
        Err(error) => attempts.push(error),
    }

    // 2. Native engine configuration. Requires VLLM_SERVER_DEV_MODE=1.
    match query_server_info_json(client, worker_url, api_key).await {
        Ok(info) if discovered_block_size(&info).is_some() => {
            info!(
                "kv_index {}: worker cache metadata from /server_info (block_size={})",
                worker_url,
                discovered_block_size(&info).unwrap_or_default()
            );
            return Ok(info);
        }
        Ok(_) => attempts.push("server_info: no block_size".to_string()),
        Err(error) => attempts.push(error),
    }

    // 3. Prometheus scrape. Always on, no dev mode required.
    match query_metrics_metadata(client, worker_url, api_key).await {
        Ok(info) if discovered_block_size(&info).is_some() => {
            info!(
                "kv_index {}: worker cache metadata from /metrics (block_size={})",
                worker_url,
                discovered_block_size(&info).unwrap_or_default()
            );
            return Ok(info);
        }
        Ok(_) => attempts.push("metrics: no block_size".to_string()),
        Err(error) => attempts.push(error),
    }

    Err(attempts.join("; "))
}

/// Engine configuration exposed by `vllm serve` behind `VLLM_SERVER_DEV_MODE=1`.
async fn query_server_info_json(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
) -> Result<ServerInfo, String> {
    let body: serde_json::Value = query_worker_json(
        client,
        worker_url,
        "server_info?config_format=json",
        true,
        api_key,
    )
    .await?;
    let config = body
        .get("vllm_config")
        .ok_or("server_info: missing vllm_config")?;
    let model = config
        .pointer("/model_config/model")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let block_size = config
        .pointer("/cache_config/block_size")
        .and_then(serde_json::Value::as_u64)
        .and_then(|size| u32::try_from(size).ok())
        .filter(|size| *size > 0);
    let effective_attention_block_size = config
        .pointer("/cache_config/effective_attention_block_size")
        .and_then(serde_json::Value::as_u64)
        .and_then(|size| u32::try_from(size).ok())
        .filter(|size| *size > 0);
    Ok(ServerInfo {
        model_id: model.clone(),
        model_path: model,
        kv_block_size: block_size,
        effective_attention_block_size,
        ..Default::default()
    })
}

/// Cache metadata from the always-on `/metrics` scrape.
async fn query_metrics_metadata(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
) -> Result<ServerInfo, String> {
    let text = query_worker_text(client, worker_url, "metrics", true, api_key).await?;
    let engines = metrics_meta::parse_cache_config_info(&text);
    let Some((_, first)) = engines.iter().next() else {
        return Err("metrics: no vllm:cache_config_info series".to_string());
    };
    if let Err(units) = metrics_meta::shared_hash_unit(&engines) {
        return Err(format!(
            "metrics: engines disagree on block size {units:?} — refusing to index"
        ));
    }
    if let Some(layout) = first.kv_cache_layout.as_deref() {
        info!(
            "kv_index {}: engine declares kv_cache_layout={} (hybrid layouts are not derived from metrics yet)",
            worker_url, layout
        );
    }
    let (model_id, model_path) = query_served_model(client, worker_url, api_key)
        .await
        .unzip();
    Ok(ServerInfo {
        model_id,
        model_path: model_path.flatten(),
        kv_block_size: first.block_size,
        effective_attention_block_size: first.effective_attention_block_size,
        ..Default::default()
    })
}

/// The served alias and its model path from the same base-model entry.
async fn query_served_model(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
) -> Option<(String, Option<String>)> {
    let models: serde_json::Value =
        query_worker_json(client, worker_url, "v1/models", true, api_key)
            .await
            .ok()?;
    let id = served_base_model_ids(&models).next()?;
    let path = models
        .get("data")?
        .as_array()?
        .iter()
        .find(|entry| entry.get("id").and_then(serde_json::Value::as_str) == Some(id))?
        .get("root")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    Some((id.to_owned(), path))
}

pub(crate) async fn query_pd_server_info(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
) -> Result<ServerInfo, String> {
    query_worker_metadata(client, worker_url, api_key).await
}

pub(crate) async fn query_pd_kv_event_sources(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
) -> Result<KvEventSourcesResponse, SourcesUnavailable> {
    let mut sources = fetch_kv_event_sources(client, worker_url, api_key).await?;
    // Native Python KVEventsConfig omits these fixed ZMQ wire declarations.
    for source in sources.values_mut() {
        if source.publisher.as_deref() == Some("zmq")
            && source.encoding.is_none()
            && source.schema_version.is_none()
        {
            source.encoding = Some("msgpack".into());
            source.schema_version = Some(1);
        }
    }
    Ok(sources)
}
