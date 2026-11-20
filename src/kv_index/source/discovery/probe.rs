//! Read worker model metadata and KV event source endpoints.

use std::collections::HashSet;
use std::time::Duration;

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
    assumed_plain_full: bool,
}

impl WorkerDescriptor {
    pub(super) fn has_nonplain_layout(&self) -> bool {
        self.hybrid
            .as_ref()
            .is_some_and(|layout| !layout.is_plain_full_attention())
    }
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
        assumed_plain_full: worker.assumed_plain_full,
    })
}

impl KvIndexSupervisor {
    pub(super) async fn probe_registration(
        &self,
        base: &str,
        plain_only: bool,
    ) -> Result<RegistrationProbe, String> {
        let mut worker = self.discover(base).await;
        if plain_only
            && worker
                .hybrid
                .as_ref()
                .is_some_and(|layout| !layout.is_plain_full_attention())
        {
            tracing::debug!(
                worker = base,
                "the current P/D configuration cannot use a hybrid KV layout"
            );
            worker.hybrid = None;
        }
        let descriptor = worker_descriptor(&worker)?;
        let base_models = if let Some(models) = worker.base_models.take() {
            models
        } else {
            query_model_list(&self.client, base, self.api_key.as_deref())
                .await
                .as_ref()
                .map(base_model_names)
                .unwrap_or_default()
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

pub(super) fn server_instance_id(info: &ServerInfo, worker_url: &str) -> String {
    info.instance_id
        .as_deref()
        .filter(|id| !id.is_empty())
        .unwrap_or(worker_url)
        .to_owned()
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

/// Fetch event publishers from either worker transport.
pub(super) async fn fetch_kv_event_sources(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
) -> Result<KvEventSourcesResponse, SourcesUnavailable> {
    if crate::backend::is_grpc_url(worker_url) {
        let response =
            crate::backend::control::get_grpc_kv_event_sources(worker_url, CONTROL_TIMEOUT).await?;
        return Ok(grpc_kv_event_source_entries(worker_url, response.sources));
    }
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

/// Read the source list and its automatically published cache contract.
pub(super) async fn fetch_native_sources(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
    timeout: Duration,
) -> (
    Result<KvEventSourcesResponse, SourcesUnavailable>,
    Result<Option<CacheCapabilities>, String>,
) {
    if crate::backend::is_grpc_url(worker_url) {
        let response =
            match crate::backend::control::get_grpc_kv_event_sources(worker_url, timeout).await {
                Ok(response) => response,
                Err(error) => return (Err(error.into()), Ok(None)),
            };
        let rank_count = response.sources.len();
        let sources = grpc_kv_event_source_entries(worker_url, response.sources);
        if sources.len() != rank_count {
            return (Ok(sources), Err("invalid KV event source ranks".into()));
        }
        let layout = match response.kv_routing_capabilities.as_deref() {
            Some(bytes) => serde_json::from_slice(bytes)
                .map_err(|error| format!("kv_routing_capabilities parse: {error}"))
                .and_then(parse_native_layout)
                .map(Some),
            None => Ok(None),
        };
        return (Ok(sources), layout);
    }

    let (sources, layout) = tokio::join!(
        fetch_kv_event_sources(client, worker_url, api_key),
        tokio::time::timeout(timeout, fetch_http_layout(client, worker_url, api_key)),
    );
    (
        sources,
        layout.unwrap_or_else(|_| Err("kv_routing_capabilities timed out".into())),
    )
}

async fn fetch_http_layout(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
) -> Result<Option<CacheCapabilities>, String> {
    let url = format!(
        "{}/kv_routing_capabilities",
        worker_url.trim_end_matches('/')
    );
    let response = worker_get(client, &url, api_key)
        .send()
        .await
        .map_err(|error| format!("kv_routing_capabilities {url}: {error}"))?;
    let status = response.status();
    if matches!(status.as_u16(), 404 | 405) {
        return Ok(None);
    }
    if !status.is_success() {
        return Err(format!("kv_routing_capabilities {url}: {status}"));
    }
    let body = response
        .json()
        .await
        .map_err(|error| format!("kv_routing_capabilities parse: {error}"))?;
    parse_native_layout(body).map(Some)
}

/// Invalid or incompatible declarations are distinct from an absent endpoint.
fn parse_native_layout(body: serde_json::Value) -> Result<CacheCapabilities, String> {
    let ranks = body
        .get("ranks")
        .and_then(serde_json::Value::as_array)
        .filter(|ranks| !ranks.is_empty())
        .ok_or("worker did not report rank layouts")?;
    if ranks.iter().any(|rank| {
        rank.get("transfer")
            .is_some_and(|transfer| !transfer.is_null())
    }) {
        return Err("worker did not report a complete cache lookup contract".into());
    }
    let reports_partial_hits = ranks
        .iter()
        .all(|rank| rank.get("partial_hash_hits").is_some());
    let mut capability: CacheCapabilities =
        serde_json::from_value(body).map_err(|e| format!("kv_routing_capabilities parse: {e}"))?;
    if !capability.is_plain_full_attention() && !reports_partial_hits {
        return Err("worker did not report partial hash hit support".into());
    }
    if capability
        .ranks
        .iter()
        .any(|rank| rank.declared_groups != u32::try_from(rank.groups.len()).ok())
    {
        return Err("worker did not declare the number of KV groups".into());
    }
    capability.ranks.sort_unstable_by_key(|rank| rank.rank);
    Ok(capability)
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
pub(super) async fn query_worker_metadata(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
) -> Result<ServerInfo, String> {
    query_worker_metadata_and_models(client, worker_url, api_key)
        .await
        .map(|(info, _)| info)
}

pub(super) async fn query_worker_metadata_and_models(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
) -> Result<(ServerInfo, Option<HashSet<String>>), String> {
    let control_error = match query_worker_control_metadata(client, worker_url, api_key).await {
        Ok(info) => return Ok((info, None)),
        Err(error) => error,
    };

    query_metrics_metadata(client, worker_url, api_key)
        .await
        .and_then(|(info, models)| {
            validate_worker_metadata(worker_url, "metrics", Ok(info))
                .map(|info| (info, Some(models)))
        })
        .map_err(|error| format!("{control_error}; {error}"))
}

/// The control endpoints can report server aliases. A metrics scrape cannot.
pub(crate) async fn query_worker_control_metadata(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
) -> Result<ServerInfo, String> {
    // 1. Multi-backend control metadata. Absent from stock vLLM.
    let first_error = match validate_worker_metadata(
        worker_url,
        "get_server_info",
        query_worker_json(client, worker_url, "get_server_info", true, api_key).await,
    ) {
        Ok(info) => return Ok(info),
        Err(error) => error,
    };

    // 2. Native engine configuration. Requires VLLM_SERVER_DEV_MODE=1.
    validate_worker_metadata(
        worker_url,
        "server_info",
        query_server_info_json(client, worker_url, api_key).await,
    )
    .map_err(|error| format!("{first_error}; {error}"))
}

fn validate_worker_metadata(
    worker_url: &str,
    path: &str,
    result: Result<ServerInfo, String>,
) -> Result<ServerInfo, String> {
    let info = result?;
    let size = discovered_block_size(&info).ok_or_else(|| format!("{path}: no block_size"))?;
    info!("kv_index {worker_url}: worker cache metadata from /{path} (block_size={size})");
    Ok(info)
}

/// Engine configuration exposed by `vllm serve` behind `VLLM_SERVER_DEV_MODE=1`.
pub(super) async fn query_server_info_json(
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
    let mooncake_truncates_compressed = match config
        .pointer("/kv_transfer_config/kv_connector")
        .and_then(serde_json::Value::as_str)
    {
        Some("MooncakeHybridConnector" | "MooncakeConnectorV1") => Some(true),
        Some("MooncakeConnector") => Some(false),
        _ => None,
    };
    Ok(ServerInfo {
        model_id: model.clone(),
        model_path: model,
        kv_block_size: block_size,
        effective_attention_block_size,
        mooncake_truncates_compressed,
        ..Default::default()
    })
}

/// Cache metadata from the always-on `/metrics` scrape.
async fn query_metrics_metadata(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
) -> Result<(ServerInfo, HashSet<String>), String> {
    let text = query_worker_text(client, worker_url, "metrics", true, api_key).await?;
    let engines = metrics_meta::parse_cache_config_info(&text);
    if engines.is_empty() {
        return Err("metrics: no vllm:cache_config_info series".to_string());
    }
    let hash_unit = metrics_meta::shared_hash_unit(&engines).map_err(|units| {
        format!("metrics: engines disagree on block size {units:?} — refusing to index")
    })?;
    if let Some(layout) = engines
        .values()
        .find_map(|engine| engine.kv_cache_layout.as_deref())
    {
        info!(
            "kv_index {}: engine declares kv_cache_layout={} (hybrid layouts are not derived from metrics yet)",
            worker_url, layout
        );
    }
    let models = query_model_list(client, worker_url, api_key).await;
    let (model_id, model_path) = models.as_ref().and_then(served_model).unzip();
    let info = ServerInfo {
        model_id,
        model_path: model_path.flatten(),
        kv_block_size: hash_unit,
        ..Default::default()
    };
    Ok((
        info,
        models.as_ref().map(base_model_names).unwrap_or_default(),
    ))
}

pub(super) async fn query_model_list(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
) -> Option<serde_json::Value> {
    query_worker_json(client, worker_url, "v1/models", true, api_key)
        .await
        .ok()
}

pub(super) fn base_model_names(models: &serde_json::Value) -> HashSet<String> {
    served_base_model_ids(models).map(str::to_owned).collect()
}

/// The served alias and its model path from the same base-model entry.
pub(super) fn served_model(models: &serde_json::Value) -> Option<(String, Option<String>)> {
    let id = served_base_model_ids(models).next()?;
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

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn grpc_model_id_preserves_errors_and_rejects_empty_ids() {
        let error = "GetModelInfo unavailable".to_string();
        assert_eq!(grpc_model_id(Err(error.clone())), Err(error));
        assert_eq!(
            grpc_model_id(Ok(crate::backend::pb::ModelInfo::default())),
            Ok(None)
        );
        assert_eq!(
            grpc_model_id(Ok(crate::backend::pb::ModelInfo {
                model_id: "   ".into(),
                ..Default::default()
            })),
            Ok(None)
        );
        assert_eq!(
            grpc_model_id(Ok(crate::backend::pb::ModelInfo {
                model_id: "model-a".into(),
                ..Default::default()
            })),
            Ok(Some("model-a".into()))
        );
    }

    #[test]
    fn grpc_sources_without_rank_are_skipped_instead_of_positionally_assigned() {
        let source = |rank| crate::backend::pb::KvEventSource {
            transport: "zmq".to_string(),
            endpoint: "tcp://worker:5557".to_string(),
            topic: String::new(),
            replay_endpoint: String::new(),
            data_parallel_rank: rank,
            encoding: "msgpack".to_string(),
            schema_version: 1,
            buffer_steps: 1,
            hwm: 100,
            max_queue_size: 0,
        };

        let entries =
            grpc_kv_event_source_entries("grpc://worker:8000", vec![source(None), source(Some(3))]);

        assert_eq!(entries.len(), 1);
        assert!(entries.contains_key("3"));
        assert_eq!(entries["3"].publisher.as_deref(), Some("zmq"));
        assert_eq!(entries["3"].enable_kv_cache_events, Some(true));
    }
}
