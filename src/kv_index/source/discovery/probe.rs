//! Read worker model metadata and KV event source endpoints.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use tokio::time::Instant;
use tonic::client::Grpc;
use tonic_prost::ProstCodec;
use tracing::{debug, info, warn};

use super::lifecycle::{WorkerLifecycle, OPTIONAL_METADATA_BUDGET};
use super::metrics_meta::EngineCacheConfigs;
use super::{
    discovered_block_size, events_enabled, metrics_meta, split_worker_rank, CacheCapabilities,
    KvEventSourceEntry, KvEventSourcesResponse, KvIndexSupervisor, ServerInfo, SourcesProbe,
    SourcesUnavailable, WorkerDiscovery, CONTROL_TIMEOUT,
};
use crate::backend::control::control_channel;
use crate::backend::pb::{GetKvEventSourcesRequest, KvEventSource};
use crate::core::worker::{worker_get, UNLABELED_WORKER_MODEL_ID};
use crate::core::Worker;

pub(super) struct RegistrationProbe {
    pub(super) descriptor: WorkerDescriptor,
    pub(super) worker: WorkerDiscovery,
}

#[derive(PartialEq)]
pub(super) struct WorkerDescriptor {
    server_info: ServerInfo,
    sources: Option<KvEventSourcesResponse>,
    model_id: Result<Option<String>, String>,
    hybrid: Option<CacheCapabilities>,
    assumed_plain_full: bool,
    pub(super) connector: Option<crate::config::KvConnector>,
}

impl WorkerDescriptor {
    pub(super) fn layout(&self) -> Option<&CacheCapabilities> {
        self.hybrid.as_ref()
    }

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
        connector: None,
    })
}

impl KvIndexSupervisor {
    pub(super) async fn probe_registration(
        &self,
        base: &str,
        plain_only: bool,
        connector: Option<crate::config::KvConnector>,
        lifecycle: &WorkerLifecycle,
        deadline: Instant,
    ) -> Result<RegistrationProbe, String> {
        let mut worker = self.discover_for_role(base, connector).await;
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
        let mut descriptor = worker_descriptor(&worker)?;
        descriptor.connector = connector;
        if worker.base_models.is_none() {
            let cached = {
                let state = lifecycle.state.lock();
                (state.info.is_some()
                    && state.descriptor.as_ref() == Some(&descriptor)
                    && (descriptor
                        .server_info
                        .instance_id
                        .as_ref()
                        .is_some_and(|id| !id.is_empty())
                        || descriptor.hybrid.is_some())
                    && !state.base_models.is_empty())
                .then(|| state.base_models.clone())
            };
            worker.base_models = Some(match cached {
                Some(models) => models,
                None => query_model_list(&self.client, base, self.api_key.as_deref())
                    .await
                    .as_ref()
                    .map(base_model_names)
                    .unwrap_or_default(),
            });
        }
        if !lifecycle.keeps_descriptor(&lifecycle.state.lock(), &descriptor) {
            self.probe_admission_hashes(base, &mut worker, Some(deadline))
                .await;
        }
        Ok(RegistrationProbe { descriptor, worker })
    }

    pub(super) async fn probe_admission_hashes(
        &self,
        worker_url: &str,
        worker: &mut WorkerDiscovery,
        deadline: Option<Instant>,
    ) {
        let (base, _) = split_worker_rank(worker_url);
        if !needs_admission_hashes(base, worker) {
            return;
        }
        // Leave enough time for the optional scrape inside the refresh deadline.
        if deadline.is_some_and(|deadline| {
            deadline.saturating_duration_since(Instant::now()) <= OPTIONAL_METADATA_BUDGET
        }) {
            debug!(
                worker_url,
                "Skipping KV hash probe because the refresh budget is exhausted"
            );
            return;
        }
        match tokio::time::timeout(
            OPTIONAL_METADATA_BUDGET,
            query_engine_cache_configs(&self.client, base, self.api_key.as_deref()),
        )
        .await
        {
            Ok(Ok(configs)) => worker.engine_cache_configs = Some(configs),
            Ok(Err(error)) => debug!(worker_url, %error, "Optional KV hash probe failed"),
            Err(_) => debug!(worker_url, "Optional KV hash probe timed out"),
        }
    }
}

pub(super) fn needs_admission_hashes(base: &str, worker: &WorkerDiscovery) -> bool {
    worker.engine_cache_configs.is_none()
        && worker.hybrid.is_some()
        && !crate::backend::is_grpc_url(base)
        && worker
            .sources
            .as_ref()
            .is_ok_and(|sources| sources.values().any(events_enabled))
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
                    encoding: (!source.encoding.is_empty()).then_some(source.encoding),
                    schema_version: (source.schema_version != 0).then_some(source.schema_version),
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

/// Event source response with the optional layout field absent from `vllm-proto` 0.2.
#[derive(Clone, PartialEq, prost::Message)]
pub struct GrpcKvEventSourcesResponse {
    #[prost(message, repeated, tag = "1")]
    pub sources: Vec<KvEventSource>,
    #[prost(bytes = "vec", optional, tag = "2")]
    pub kv_routing_capabilities: Option<Vec<u8>>,
}

pub async fn get_grpc_kv_event_sources(
    worker_url: &str,
    timeout: Duration,
) -> Result<GrpcKvEventSourcesResponse, tonic::Status> {
    let channel = control_channel(worker_url, timeout)
        .await
        .map_err(tonic::Status::unavailable)?;
    let mut client = Grpc::new(channel);
    let call = async {
        client
            .ready()
            .await
            .map_err(|error| tonic::Status::unavailable(format!("service not ready: {error}")))?;
        let mut request = tonic::Request::new(GetKvEventSourcesRequest {});
        request
            .extensions_mut()
            .insert(tonic::GrpcMethod::new("vllm.Control", "GetKvEventSources"));
        client
            .unary(
                request,
                http::uri::PathAndQuery::from_static("/vllm.Control/GetKvEventSources"),
                ProstCodec::default(),
            )
            .await
    };
    let response = tokio::time::timeout(timeout, call)
        .await
        .map_err(|_| tonic::Status::deadline_exceeded("GetKvEventSources timed out"))??;
    Ok(response.into_inner())
}

/// Fetch event publishers from either worker transport.
pub(super) async fn fetch_kv_event_sources(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
) -> Result<KvEventSourcesResponse, SourcesUnavailable> {
    if crate::backend::is_grpc_url(worker_url) {
        let response = get_grpc_kv_event_sources(worker_url, CONTROL_TIMEOUT)
            .await
            .map_err(|status| grpc_sources_unavailable(worker_url, status))?;
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
        let response = match get_grpc_kv_event_sources(worker_url, timeout).await {
            Ok(response) => response,
            Err(status) => return (Err(grpc_sources_unavailable(worker_url, status)), Ok(None)),
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
        tokio::time::timeout(
            timeout,
            fetch_http_layout(client, worker_url, api_key, timeout / 3)
        ),
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
    attempt_timeout: Duration,
) -> Result<Option<CacheCapabilities>, String> {
    let url = format!(
        "{}/kv_routing_capabilities",
        worker_url.trim_end_matches('/')
    );
    let fetch = || tokio::time::timeout(attempt_timeout, worker_get(client, &url, api_key).send());
    let mut response = fetch().await;
    if !matches!(&response, Ok(Ok(response)) if !response.status().is_server_error()) {
        // Retry a transient probe failure before deciding the worker's cache shape.
        response = fetch().await;
    }
    let response = response
        .map_err(|_| format!("kv_routing_capabilities {url} timed out"))?
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

fn grpc_sources_unavailable(worker_url: &str, status: tonic::Status) -> SourcesUnavailable {
    let probe = match status.code() {
        tonic::Code::Unimplemented => SourcesProbe::Unsupported,
        _ => SourcesProbe::Failed,
    };
    SourcesUnavailable {
        probe,
        detail: format!(
            "grpc GetKvEventSources {worker_url}: {:?}: {}",
            status.code(),
            status.message()
        ),
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

pub(super) async fn query_vllm_version(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
) -> Option<(u32, u32)> {
    if crate::backend::is_grpc_url(worker_url) {
        let info = crate::backend::control::get_grpc_server_info(worker_url, CONTROL_TIMEOUT)
            .await
            .ok()?;
        return parse_vllm_version(&info.engine_version);
    }
    let info: serde_json::Value = query_worker_json(client, worker_url, "version", true, api_key)
        .await
        .ok()?;
    parse_vllm_version(info.get("version")?.as_str()?)
}

pub(super) fn parse_vllm_version(version: &str) -> Option<(u32, u32)> {
    let mut parts = version.trim_start_matches('v').split('.');
    Some((parts.next()?.parse().ok()?, parts.next()?.parse().ok()?))
}

#[derive(Debug)]
pub(super) struct WorkerMetadata {
    pub(super) server_info: Result<ServerInfo, String>,
    pub(super) base_models: Option<HashSet<String>>,
    /// Parsed reports can remain available when cache geometry is rejected.
    pub(super) engine_cache_configs: Option<EngineCacheConfigs>,
}

/// Worker cache metadata, tried in the order a deployment is likely to answer it:
/// the multi-backend control contract, the engine config endpoint, then the
/// always-on `/metrics` scrape.
///
/// Every source is optional; the error names what each one said, so an operator
/// can tell a missing endpoint from an endpoint that answered without a block
/// size.
pub(super) async fn query_worker_metadata_and_models(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
) -> WorkerMetadata {
    let control_error = match query_worker_control_metadata(client, worker_url, api_key).await {
        Ok(server_info) => {
            return WorkerMetadata {
                server_info: Ok(server_info),
                base_models: None,
                engine_cache_configs: None,
            };
        }
        Err(error) => error,
    };

    let (metadata, engine_cache_configs) =
        match query_engine_cache_configs(client, worker_url, api_key).await {
            Ok(engines) => (
                query_metrics_metadata(client, worker_url, api_key, &engines).await,
                Some(engines),
            ),
            Err(error) => (Err(error), None),
        };
    let (server_info, base_models) = match metadata {
        Ok((info, models)) => (Ok(info), models),
        Err(error) => (Err(format!("{control_error}; {error}")), None),
    };
    WorkerMetadata {
        server_info,
        base_models,
        engine_cache_configs,
    }
}

/// The control endpoints can report server aliases. A metrics scrape cannot.
async fn query_worker_control_metadata(
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
        Some("MooncakeConnectorV2") => config
            .pointer("/model_config/hf_config")
            .and_then(serde_json::Value::as_object)
            .map(|config| {
                config
                    .get("compress_ratios")
                    .is_some_and(|value| value.is_array() || value.is_object())
            }),
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
    engines: &EngineCacheConfigs,
) -> Result<(ServerInfo, Option<HashSet<String>>), String> {
    let hash_unit = metrics_meta::shared_hash_unit(engines).map_err(|units| {
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
        validate_worker_metadata(worker_url, "metrics", Ok(info))?,
        models.as_ref().map(base_model_names),
    ))
}

pub(super) async fn query_engine_cache_configs(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
) -> Result<EngineCacheConfigs, String> {
    let text = query_worker_text(client, worker_url, "metrics", true, api_key).await?;
    let (engines, malformed_ranks) = metrics_meta::parse_cache_config_info(&text);
    if malformed_ranks > 0 {
        tracing::debug!(
            worker_url,
            metric = metrics_meta::CACHE_CONFIG_INFO,
            label = "engine",
            malformed_ranks,
            "ignoring cache config samples with invalid engine ranks"
        );
    }
    if engines.is_empty() {
        return Err("metrics: no vllm:cache_config_info series".to_string());
    }
    Ok(engines)
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

/// Base-model names used to distinguish served aliases from LoRA names.
/// Probe `/v1/models` only when no model-list snapshot was supplied.
pub(crate) async fn served_model_labels(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
    dp_rank: Option<usize>,
    base_models: Option<HashSet<String>>,
) -> HashMap<String, String> {
    let mut labels = HashMap::new();
    if crate::backend::is_grpc_url(worker_url) {
        return labels;
    }
    let timeout = Duration::from_secs(2);
    let mut aliases: Vec<String> = if let Some(models) = base_models {
        models.into_iter().collect()
    } else {
        let url = format!("{}/v1/models", worker_url.trim_end_matches('/'));
        let mut probe = worker_get(client, &url, api_key).timeout(timeout);
        if let Some(rank) = dp_rank {
            probe = probe.header("X-data-parallel-rank", rank.to_string());
        }
        let served = match probe.send().await {
            Ok(response) if response.status().is_success() => {
                response.json::<serde_json::Value>().await.ok()
            }
            _ => None,
        };
        served
            .as_ref()
            .into_iter()
            .flat_map(served_base_model_ids)
            .map(str::to_owned)
            .collect()
    };
    if let Ok(Ok(info)) = tokio::time::timeout(
        timeout,
        query_worker_control_metadata(client, worker_url, api_key),
    )
    .await
    {
        aliases.extend(info.served_model_aliases);
    }
    if !aliases.is_empty() {
        aliases.sort_unstable();
        aliases.dedup();
        labels.insert("served_model_aliases".to_string(), aliases.join(","));
    }
    labels
}

/// Base model IDs in the OpenAI model list. A LoRA entry names its parent.
fn served_base_model_ids(body: &serde_json::Value) -> impl Iterator<Item = &str> {
    body.get("data")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter(|entry| entry.get("parent").is_none_or(serde_json::Value::is_null))
        .filter_map(|entry| entry.get("id").and_then(serde_json::Value::as_str))
        .filter(|id| !id.trim().is_empty())
}

/// Names this worker serves as base models: the model it registered plus
/// the `--served-model-name` aliases the engine reported, when known.
fn worker_base_model_names(worker: &dyn Worker) -> impl Iterator<Item = &str> {
    std::iter::once(worker.model_id()).chain(
        worker
            .metadata()
            .labels
            .get("served_model_aliases")
            .into_iter()
            .flat_map(|aliases| aliases.split(',').filter(|alias| !alias.is_empty())),
    )
}

/// Whether `name` selects this worker's base model rather than one of its
/// adapters. vLLM selects an adapter through the request's `model`, and an
/// adapter name is neither the model id nor a served alias.
pub(crate) fn worker_serves_base_model(worker: &dyn Worker, name: &str) -> bool {
    worker_base_model_names(worker).any(|model| model == name)
}

pub(crate) fn inferred_lora_name<'a>(
    workers: &[Arc<dyn Worker>],
    requested: &'a str,
) -> Option<&'a str> {
    // vLLM selects adapters through model. An incomplete set of base
    // aliases leaves the identity unclassified.
    (!workers.is_empty()
        && workers.iter().all(|worker| {
            let mut known_base = false;
            for name in worker_base_model_names(worker.as_ref()) {
                if name == requested {
                    return false;
                }
                known_base |= !name.is_empty() && name != UNLABELED_WORKER_MODEL_ID;
            }
            known_base
        }))
    .then_some(requested)
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

    #[tokio::test]
    async fn layout_discovery_retries_transient_http_failure() {
        use axum::{http::StatusCode, routing::get, Router};
        use std::sync::atomic::{AtomicUsize, Ordering};
        let requests = Arc::new(AtomicUsize::new(0));
        let app = Router::new().route(
            "/kv_routing_capabilities",
            get({
                let requests = requests.clone();
                move || {
                    let attempt = requests.fetch_add(1, Ordering::Relaxed);
                    async move {
                        if attempt == 0 {
                            StatusCode::SERVICE_UNAVAILABLE
                        } else {
                            StatusCode::NOT_FOUND
                        }
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let report = fetch_http_layout(&reqwest::Client::new(), &url, None, Duration::from_secs(1))
            .await
            .unwrap();
        assert!(report.is_none());
        assert_eq!(requests.load(Ordering::Relaxed), 2);
        server.abort();
    }

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

        let mut legacy_source = source(Some(0));
        legacy_source.encoding.clear();
        legacy_source.schema_version = 0;
        let entries = grpc_kv_event_source_entries("grpc://worker:8000", vec![legacy_source]);
        assert_eq!(entries["0"].encoding, None);
        assert_eq!(entries["0"].schema_version, None);
        assert!(super::super::events_enabled(&entries["0"]));
    }
}
