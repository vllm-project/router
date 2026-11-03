//! Opt-in, one-way PD Program scheduling and response-lifetime settlement.
use super::dp_utils;
use super::pd_warm_remote::{PdResponseContext, PdWarmRemoteRuntime};
use super::program_adapter::ProgramCompletion;
use super::vllm_service_discovery::{ServiceRegistry, ServiceType};
use crate::core::{BasicWorker, DPAwareWorker, Worker, WorkerRegistry, WorkerType};
use crate::kv_index::{HashMode, KvIndexSupervisor};
use crate::program_scheduling::{
    max_prometheus_gauge, sum_prometheus_gauge, BackendObservationProvider, ProgramIdentity,
    ProgramScheduler, ProgramSchedulerConfig, ProgramTarget, ScheduleError,
    VllmMetricsObservationProvider,
};
use crate::server::AppContext;
use crate::token_estimator::{MomentumTokenEstimator, TokenEstimateScope};
use axum::{
    body::Body,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use parking_lot::Mutex;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeSet, HashMap},
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tracing::warn;

#[derive(Debug)]
pub(super) struct PdProgramRuntime {
    pub scheduler: Arc<ProgramScheduler>,
    pub warm_remote: Option<Arc<PdWarmRemoteRuntime>>,
    pub worker_registry: Arc<WorkerRegistry>,
    pub discovery: bool,
    estimator: Arc<MomentumTokenEstimator>,
    prefill_gates: Arc<Mutex<HashMap<String, Arc<Semaphore>>>>,
    prefill_limit: usize,
    queue_timeout: Duration,
    observer: tokio::task::AbortHandle,
    kv_observer: Option<tokio::task::AbortHandle>,
    prefill_observer: Option<tokio::task::AbortHandle>,
}

impl PdProgramRuntime {
    pub fn new(
        ctx: &Arc<AppContext>,
        services: Arc<ServiceRegistry>,
        discovery: bool,
    ) -> Result<Option<Self>, String> {
        let Some(config) = ctx.router_config.program_scheduling.as_ref() else {
            return Ok(None);
        };
        let warm_remote = config
            .pd
            .warm_remote_enabled()
            .then(|| Arc::new(PdWarmRemoteRuntime::new(config.pd.clone())));
        if let Some(warm_remote) = &warm_remote {
            warm_remote.set_event_tracking(config.pd.kv_events);
        }
        let observer_warm_remote = warm_remote.clone();
        let scheduler = Arc::new(ProgramScheduler::new(
            ProgramSchedulerConfig::from_program_config(config, true),
        ));
        let observation_scheduler = scheduler.clone();
        let workers = ctx.worker_registry.clone();
        let dp_size = ctx.router_config.intra_node_data_parallel_size;
        let provider = VllmMetricsObservationProvider::new(
            ctx.client.clone(),
            ctx.router_config.api_key.clone(),
        );
        let kv_config = config.pd.clone();
        let mut default_headers = HeaderMap::new();
        if let Some(key) = &ctx.router_config.api_key {
            if let Ok(value) = format!("Bearer {key}").parse() {
                default_headers.insert(axum::http::header::AUTHORIZATION, value);
            }
        }
        let kv_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .default_headers(default_headers)
            .build()
            .map_err(|error| format!("PD KV client initialization failed: {error}"))?;
        let kv_workers_registry = workers.clone();
        let kv_services = services.clone();
        let kv_interval = scheduler.metrics_interval().max(Duration::from_secs(1));
        let kv_warm_remote = warm_remote.clone();
        let kv_observer = if kv_config.kv_events {
            Some(
                tokio::spawn(async move {
                    let (supervisor, mut signals) = KvIndexSupervisor::new(
                        kv_client,
                        HashMode::parse_config(&kv_config.kv_hash_mode),
                    );
                    let mut previous = BTreeSet::new();
                    let mut subscribed = BTreeSet::new();
                    let mut loss_epoch = crate::kv_index::signal_loss_epoch();
                    let mut interval = tokio::time::interval(kv_interval);
                    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                    loop {
                        interval.tick().await;
                        while let Ok(signal) = signals.try_recv() {
                            use crate::kv_index::IngestionSignal;
                            if let Some(warm_remote) = &kv_warm_remote {
                                let invalidation = match signal {
                                    IngestionSignal::Invalidated {
                                        source, dp_rank, ..
                                    } => Some((source, Some(dp_rank as usize))),
                                    IngestionSignal::Gap { source, .. }
                                    | IngestionSignal::IncarnationReset { source, .. } => {
                                        Some((source, None))
                                    }
                                    _ => None,
                                };
                                if let Some((source, rank)) = invalidation {
                                    for url in supervisor.worker_urls_for_source(&source) {
                                        warm_remote.invalidate_target(&url, rank);
                                    }
                                }
                            }
                        }
                        let current_loss = crate::kv_index::signal_loss_epoch();
                        if current_loss != loss_epoch {
                            if let Some(warm_remote) = &kv_warm_remote {
                                warm_remote.invalidate_all();
                            }
                            loss_epoch = current_loss;
                        }
                        let urls = all_worker_bases(&kv_workers_registry, &kv_services, discovery);
                        for removed in previous.difference(&urls) {
                            if let Some(warm_remote) = &kv_warm_remote {
                                warm_remote.invalidate_target(removed, None);
                            }
                        }
                        refresh_kv_workers(&supervisor, &mut previous, &mut subscribed, urls).await;
                    }
                })
                .abort_handle(),
            )
        } else {
            None
        };
        let pressure_workers = workers.clone();
        let pressure_services = services.clone();
        let pressure_client = ctx.client.clone();
        let pressure_scheduler = scheduler.clone();
        let pressure_interval =
            Duration::from_secs_f64(config.pd.prefill_kv_window_seconds.max(1.0));
        let prefill_observer = if config.pd.prefill_kv_high_watermark > 0.0 || warm_remote.is_some()
        {
            Some(
                tokio::spawn(async move {
                    let mut interval = tokio::time::interval(pressure_interval);
                    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                    loop {
                        interval.tick().await;
                        let mut kv_pressure: Option<f64> = None;
                        for url in
                            prefill_base_urls(&pressure_workers, &pressure_services, discovery)
                        {
                            let endpoint = format!("{}/metrics", url.trim_end_matches('/'));
                            let text = match pressure_client
                                .get(endpoint)
                                .timeout(Duration::from_secs(2))
                                .send()
                                .await
                            {
                                Ok(response) if response.status().is_success() => {
                                    response.text().await.ok()
                                }
                                _ => None,
                            };
                            let Some(text) = text else { continue };
                            let usage = max_prometheus_gauge(&text, "vllm:kv_cache_usage_perc");
                            let running = sum_prometheus_gauge(&text, "vllm:num_requests_running")
                                .and_then(|n| usize::try_from(n).ok());
                            if usage.is_some() || running.is_some() {
                                pressure_scheduler.observe_prefill_target(&url, usage, running);
                            }
                            if let Some(usage) = usage {
                                kv_pressure = Some(
                                    kv_pressure.map_or(usage, |current: f64| current.max(usage)),
                                );
                            }
                        }
                        if let Some(usage) = kv_pressure {
                            pressure_scheduler.observe_prefill_kv_usage(usage);
                        }
                    }
                })
                .abort_handle(),
            )
        } else {
            None
        };
        let prefill_gates = Arc::new(Mutex::new(HashMap::<String, Arc<Semaphore>>::new()));
        let observer_gates = prefill_gates.clone();
        let prefill_limit = config.pd.max_prefill_requests_per_target;
        let observer = tokio::spawn(async move {
            let mut interval = tokio::time::interval(observation_scheduler.metrics_interval());
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                interval.tick().await;
                let targets = decode_targets(&workers, &services, discovery, dp_size);
                if let Some(warm_remote) = &observer_warm_remote {
                    warm_remote.retain_sources(&targets);
                }
                for pool in observation_scheduler.model_pools() {
                    observation_scheduler.sync_targets(&pool, &targets);
                }
                let epoch = observation_scheduler.begin_observation(&targets);
                let (observations, failures) = provider.observe(&targets).await;
                for failure in failures {
                    warn!(error = %failure.error, "PD metrics scrape failed");
                }
                observation_scheduler.apply_observations(epoch, observations);
                observation_scheduler.tick();
                let live_prefill: BTreeSet<String> = if discovery {
                    services
                        .get_prefill_instances()
                        .iter()
                        .map(|(url, _)| discovered_target(url).id)
                        .collect()
                } else {
                    workers
                        .get_prefill_workers()
                        .iter()
                        .map(|w| w.url().to_string())
                        .collect()
                };
                observer_gates.lock().retain(|id, gate| {
                    live_prefill.contains(id)
                        || Arc::strong_count(gate) > 1
                        || gate.available_permits() != prefill_limit
                });
            }
        })
        .abort_handle();
        Ok(Some(Self {
            scheduler,
            warm_remote,
            worker_registry: ctx.worker_registry.clone(),
            discovery,
            estimator: Arc::new(MomentumTokenEstimator::default()),
            prefill_gates,
            prefill_limit: config.pd.max_prefill_requests_per_target,
            queue_timeout: Duration::from_secs_f64(config.queue_timeout_seconds),
            observer,
            kv_observer,
            prefill_observer,
        }))
    }

    pub async fn acquire(
        &self,
        headers: Option<&HeaderMap>,
        request: &Value,
        path: &str,
        targets: &[ProgramTarget],
    ) -> Result<Option<ProgramCompletion>, ScheduleError> {
        let pool = model_pool(headers, request);
        let Some(mut identity) = ProgramIdentity::from_request_with_enable_key(
            headers,
            Some(request),
            Some(&pool),
            self.scheduler.enable_key(),
        )?
        else {
            return Ok(None);
        };
        // The whole request includes templates, tools and multimodal envelopes.
        // HTTP token estimation is advisory; engine physical limits remain authoritative.
        let (tokens, calibration) = if let Some(tokens) = explicit_completion_tokens(request, path)
        {
            // A flat token-ID prompt is already tokenized. Counting its JSON
            // digits over-reserves admission and can strand first-turn waiters.
            // Empty calibration input keeps this exact count from changing the
            // byte estimator used for text/chat requests in the same pool.
            let (_, calibration) = self
                .estimator
                .estimate(TokenEstimateScope::new(&pool, path), "");
            (tokens, calibration)
        } else {
            let input = serde_json::to_string(request).unwrap_or_default();
            self.estimator
                .estimate(TokenEstimateScope::new(&pool, path), &input)
        };
        let default_output = self.scheduler.config.progress_ttl.decode_buffer_tokens;
        reserve_output_budget(&mut identity, request, path, default_output);
        let dispatch = self
            .scheduler
            .acquire(identity, tokens, targets, None)
            .await?;
        Ok(Some(ProgramCompletion::new(
            self.scheduler.clone(),
            dispatch,
            self.estimator.clone(),
            calibration,
        )))
    }

    pub async fn reserve_prefill(
        &self,
        worker_id: &str,
    ) -> Result<OwnedSemaphorePermit, ScheduleError> {
        let gate = self
            .prefill_gates
            .lock()
            .entry(worker_id.to_string())
            .or_insert_with(|| Arc::new(Semaphore::new(self.prefill_limit)))
            .clone();
        tokio::time::timeout(self.queue_timeout, gate.acquire_owned())
            .await
            .map_err(|_| ScheduleError::QueueTimeout)?
            .map_err(|_| ScheduleError::NoTargets)
    }
}

async fn refresh_kv_workers(
    supervisor: &KvIndexSupervisor,
    previous: &mut BTreeSet<String>,
    subscribed: &mut BTreeSet<String>,
    urls: BTreeSet<String>,
) {
    for removed in previous.difference(&urls) {
        supervisor.on_worker_removed(removed);
        subscribed.remove(removed);
    }
    // Discovery can fail or soft-skip while an engine starts. Only successful
    // subscriptions are complete; other registered workers are retried later.
    let pending = urls.difference(subscribed).cloned().collect::<Vec<_>>();
    for worker in pending {
        match supervisor.on_worker_added(&worker).await {
            Ok(info) if info.cache_key.is_some() && !info.ranks.is_empty() => {
                subscribed.insert(worker);
            }
            Ok(_) => {}
            Err(error) => warn!(%error, "PD KV discovery failed"),
        }
    }
    *previous = urls;
}

fn explicit_completion_tokens(request: &Value, path: &str) -> Option<usize> {
    if path != "/v1/completions" {
        return None;
    }
    let tokens = request.get("prompt")?.as_array()?;
    if tokens.is_empty() {
        return None;
    }
    let valid_id = |token: &Value| token.as_u64().is_some_and(|id| id <= u32::MAX as u64);
    if tokens.iter().all(valid_id) {
        return Some(tokens.len());
    }
    let mut total = 0usize;
    for prompt in tokens {
        let prompt = prompt.as_array()?;
        if !prompt.iter().all(valid_id) {
            return None;
        }
        total = total.saturating_add(prompt.len());
    }
    Some(total)
}

fn completion_prompt_count(request: &Value, path: &str) -> usize {
    if path != "/v1/completions" {
        return 1;
    }
    match request.get("prompt").and_then(Value::as_array) {
        Some(prompts)
            if prompts
                .first()
                .is_some_and(|prompt| prompt.is_string() || prompt.is_array()) =>
        {
            prompts.len()
        }
        _ => 1,
    }
}

fn requested_output_budget(request: &Value, path: &str, default_output: usize) -> Option<usize> {
    let wire_limit = request
        .get("max_completion_tokens")
        .or_else(|| request.get("max_tokens"))
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok());
    let choices = request.get("n").and_then(Value::as_u64).unwrap_or(1).max(1);
    let best_of = if path == "/v1/completions" {
        request
            .get("best_of")
            .and_then(Value::as_u64)
            .unwrap_or(1)
            .max(1)
    } else {
        1
    };
    // `n` returns choices and `best_of` samples candidates; both consume an
    // independent output budget. Without a wire limit the configured decode
    // buffer is the per-choice fallback, so multiply it by the choice count.
    // Saturate instead of wrapping: an impossible budget must fail closed at
    // admission, not under-reserve.
    let choices = usize::try_from(choices.max(best_of)).unwrap_or(usize::MAX);
    let prompt_count = completion_prompt_count(request, path);
    let per_choice = wire_limit.unwrap_or(default_output);
    if wire_limit.is_none() && choices == 1 && prompt_count == 1 {
        return None;
    }
    Some(
        per_choice
            .saturating_mul(choices)
            .saturating_mul(prompt_count),
    )
}

fn reserve_output_budget(
    identity: &mut ProgramIdentity,
    request: &Value,
    path: &str,
    default_output: usize,
) {
    if let Some(limit) = requested_output_budget(request, path, default_output) {
        identity.reserve_pd_output(limit);
    }
}

impl Drop for PdProgramRuntime {
    fn drop(&mut self) {
        self.observer.abort();
        if let Some(observer) = &self.kv_observer {
            observer.abort();
        }
        if let Some(observer) = &self.prefill_observer {
            observer.abort();
        }
    }
}

pub(super) fn model_pool(headers: Option<&HeaderMap>, request: &Value) -> String {
    // Isolate credentials without retaining or logging them. With gateway auth,
    // deploy one authoritative Router shard per authenticated tenant/pool.
    let principal = headers
        .and_then(|h| h.get(axum::http::header::AUTHORIZATION))
        .map(|v| v.as_bytes())
        .unwrap_or_default();
    let model = request
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("default");
    let mut digest = Sha256::new();
    digest.update((principal.len() as u64).to_le_bytes());
    digest.update(principal);
    digest.update(model.as_bytes());
    format!("pd:{:x}", digest.finalize())
}

pub(super) fn worker_target(worker: &Arc<dyn Worker>) -> ProgramTarget {
    let (base_url, rank) = dp_utils::parse_worker_url(worker.url());
    ProgramTarget {
        id: worker.url().to_string(),
        base_url,
        dp_rank: worker.dp_rank().or(rank),
    }
}

pub(super) fn discovered_target(address: &str) -> ProgramTarget {
    let url = if address.starts_with("http://") || address.starts_with("https://") {
        address.to_string()
    } else {
        format!("http://{address}")
    };
    let (base_url, dp_rank) = dp_utils::parse_worker_url(&url);
    ProgramTarget {
        id: url,
        base_url,
        dp_rank,
    }
}

/// Explicit rank selection keeps discovered internal DP D affinity independent of P.
pub(super) fn discovered_decode_instances(
    services: &ServiceRegistry,
    configured_dp: usize,
) -> Vec<(String, String)> {
    services
        .get_decode_instances()
        .into_iter()
        .flat_map(|(address, zmq)| {
            let (_, rank) = dp_utils::parse_worker_url(&address);
            let size = decode_instance_size(
                services.get_reported_dp_size(&address, ServiceType::Decode),
                configured_dp,
            );
            if rank.is_some() || size == 1 {
                vec![(address, zmq)]
            } else {
                (0..size)
                    .map(|rank| (format!("{address}@{rank}"), zmq.clone()))
                    .collect()
            }
        })
        .collect()
}

pub(super) fn discovered_decode_instance_by_id(
    services: &ServiceRegistry,
    configured_dp: usize,
    target_id: &str,
) -> Option<(String, String)> {
    decode_instance_entries(services, configured_dp)
        .into_iter()
        .find(|(url, _, _)| discovered_target(url).id == target_id)
        .map(|(url, zmq, _)| (url, zmq))
}

/// Build the decode worker bound by the scheduler, using the same rank
/// expansion and DP-size fallback as the candidate list it was chosen from.
pub(super) fn discovered_decode_worker(
    services: &ServiceRegistry,
    configured_dp: usize,
    target_id: &str,
) -> Option<Arc<dyn Worker>> {
    let (address, _, size) = decode_instance_entries(services, configured_dp)
        .into_iter()
        .find(|(url, _, _)| discovered_target(url).id == target_id)?;
    let (base, rank) = dp_utils::parse_worker_url(&address);
    let url = if base.starts_with("http://") || base.starts_with("https://") {
        base.clone()
    } else {
        format!("http://{base}")
    };
    Some(match rank {
        // DPAwareWorker strips the `@rank` suffix from dispatched URLs, so the
        // bound rank survives endpoint construction.
        Some(rank) => Arc::new(DPAwareWorker::new(url, rank, size, WorkerType::Decode)),
        None => Arc::new(BasicWorker::new(url, WorkerType::Decode).with_dp_size(size)),
    })
}

/// Registered decode endpoints expanded to one entry per schedulable DP rank,
/// carrying the DP size the scheduler will bind against.
fn decode_instance_entries(
    services: &ServiceRegistry,
    configured_dp: usize,
) -> Vec<(String, String, usize)> {
    services
        .get_decode_instances()
        .into_iter()
        .flat_map(|(address, zmq)| {
            let (_, rank) = dp_utils::parse_worker_url(&address);
            let size = decode_instance_size(
                services.get_reported_dp_size(&address, ServiceType::Decode),
                configured_dp,
            );
            if rank.is_some() || size == 1 {
                vec![(address, zmq, size)]
            } else {
                (0..size)
                    .map(|rank| (format!("{address}@{rank}"), zmq.clone(), size))
                    .collect()
            }
        })
        .collect()
}

fn decode_instance_size(reported_dp: Option<usize>, configured_dp: usize) -> usize {
    // An explicit registration wins even when it is smaller than the router
    // configuration; the configured value is only a fallback for engines that
    // did not report a DP size.
    reported_dp.unwrap_or(configured_dp).max(1)
}

fn decode_targets(
    workers: &WorkerRegistry,
    services: &ServiceRegistry,
    discovery: bool,
    dp_size: usize,
) -> Vec<ProgramTarget> {
    if discovery {
        discovered_decode_instances(services, dp_size)
            .iter()
            .map(|(url, _)| discovered_target(url))
            .collect()
    } else {
        workers
            .get_decode_workers()
            .iter()
            .filter(|w| w.is_available())
            .map(worker_target)
            .collect()
    }
}

fn prefill_base_urls(
    workers: &WorkerRegistry,
    services: &ServiceRegistry,
    discovery: bool,
) -> BTreeSet<String> {
    if discovery {
        services
            .get_prefill_instances()
            .into_iter()
            .map(|(url, _)| discovered_target(&url).base_url)
            .collect()
    } else {
        workers
            .get_prefill_workers()
            .into_iter()
            .filter(|worker| worker.is_available())
            .map(|worker| worker_target(&worker).base_url)
            .collect()
    }
}

fn all_worker_bases(
    workers: &WorkerRegistry,
    services: &ServiceRegistry,
    discovery: bool,
) -> BTreeSet<String> {
    if discovery {
        services
            .get_prefill_instances()
            .into_iter()
            .chain(services.get_decode_instances())
            .map(|(url, _)| discovered_target(&url).base_url)
            .collect()
    } else {
        workers
            .get_prefill_workers()
            .into_iter()
            .chain(workers.get_decode_workers())
            .filter(|w| w.is_available())
            .map(|w| worker_target(&w).base_url)
            .collect()
    }
}

pub(super) fn error_response(error: ScheduleError) -> Response {
    let status = match error {
        ScheduleError::InvalidIdentity(_) => StatusCode::BAD_REQUEST,
        ScheduleError::NoTargets => StatusCode::SERVICE_UNAVAILABLE,
        ScheduleError::QueueTimeout => StatusCode::TOO_MANY_REQUESTS,
    };
    (status, error.to_string()).into_response()
}

/// Transparently pass every byte; completion follows the entire D body, not P
/// completion, headers, or the first output. Dropping it settles failure once.
pub(super) fn track_response(
    response: Response,
    completion: ProgramCompletion,
    streaming: bool,
) -> Response {
    track_response_inner(response, completion, streaming, None)
}

pub(super) fn track_warm_remote_response(
    response: Response,
    completion: ProgramCompletion,
    streaming: bool,
    warm_remote: PdResponseContext,
) -> Response {
    track_response_inner(response, completion, streaming, Some(warm_remote))
}

/// Forward an upstream error response untouched. The Program turn is still
/// settled as failed when the body completes; no candidate is published.
pub(super) fn track_forwarded_response(
    response: Response,
    completion: ProgramCompletion,
    streaming: bool,
) -> Response {
    track_response_inner(response, completion, streaming, None)
}

fn track_response_inner(
    response: Response,
    completion: ProgramCompletion,
    streaming: bool,
    warm_remote: Option<PdResponseContext>,
) -> Response {
    let success_status = response.status().is_success();
    let buffer_limit = warm_remote
        .as_ref()
        .map_or(PdTrackedBody::MAX_BUFFER, |context| {
            context.response_buffer_limit()
        });
    let (parts, body) = response.into_parts();
    Response::from_parts(
        parts,
        Body::from_stream(PdTrackedBody {
            inner: Box::pin(body.into_data_stream()),
            warm_remote,
            completion,
            streaming,
            success_status,
            terminal: false,
            failed: false,
            buffer_overflow: false,
            skipping_oversized_line: false,
            buffer: Vec::new(),
            buffer_limit,
        }),
    )
}

struct PdTrackedBody {
    warm_remote: Option<PdResponseContext>,
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, axum::Error>> + Send>>,
    completion: ProgramCompletion,
    streaming: bool,
    success_status: bool,
    terminal: bool,
    failed: bool,
    buffer_overflow: bool,
    skipping_oversized_line: bool,
    buffer: Vec<u8>,
    /// Body-buffer bound: the configured response bound for warm_remote turns,
    /// the conservative fixed bound otherwise.
    buffer_limit: usize,
}

impl PdTrackedBody {
    const MAX_BUFFER: usize = 1024 * 1024;
    fn observe(&mut self, bytes: &[u8]) {
        if self.streaming {
            self.completion.observe_sse_chunk(bytes);
            for segment in bytes.split_inclusive(|b| *b == b'\n') {
                if self.skipping_oversized_line {
                    if segment.last() == Some(&b'\n') {
                        self.skipping_oversized_line = false;
                    }
                    continue;
                }
                if self.buffer.len().saturating_add(segment.len()) > self.buffer_limit {
                    self.buffer.clear();
                    self.skipping_oversized_line = segment.last() != Some(&b'\n');
                    continue;
                }
                self.buffer.extend_from_slice(segment);
                if segment.last() == Some(&b'\n') {
                    let line = std::mem::take(&mut self.buffer);
                    self.observe_line(&line);
                }
            }
        } else if !self.buffer_overflow {
            if self.buffer.len().saturating_add(bytes.len()) > self.buffer_limit {
                self.buffer_overflow = true;
                self.buffer.clear();
            } else {
                self.buffer.extend_from_slice(bytes);
            }
        }
    }
    fn observe_line(&mut self, line: &[u8]) {
        if let Ok(line) = std::str::from_utf8(line) {
            if let Some(data) = line.trim().strip_prefix("data:") {
                let data = data.trim();
                if data == "[DONE]" {
                    self.terminal = true;
                }
                if data
                    .as_bytes()
                    .windows(b"\"error\"".len())
                    .any(|window| window == b"\"error\"")
                    && serde_json::from_str::<Value>(data)
                        .ok()
                        .is_some_and(|v| v.get("error").is_some())
                {
                    self.failed = true;
                }
                if self.warm_remote.is_some() {
                    if let Ok(value) = serde_json::from_str::<Value>(data) {
                        if let Some(warm_remote) = &mut self.warm_remote {
                            warm_remote.observe(&value);
                        }
                    }
                }
            }
        }
    }
}

impl Drop for PdTrackedBody {
    fn drop(&mut self) {
        self.completion.finish(false);
    }
}

impl Stream for PdTrackedBody {
    type Item = Result<Bytes, axum::Error>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(bytes))) => {
                self.observe(&bytes);
                Poll::Ready(Some(Ok(bytes)))
            }
            Poll::Ready(Some(Err(error))) => {
                self.failed = true;
                self.completion.finish(false);
                Poll::Ready(Some(Err(error)))
            }
            Poll::Ready(None) => {
                if self.streaming {
                    let tail = std::mem::take(&mut self.buffer);
                    self.observe_line(&tail);
                } else {
                    self.completion.observe_json(&self.buffer);
                    if let Ok(value) = serde_json::from_slice::<Value>(&self.buffer) {
                        if value.get("error").is_some() {
                            self.failed = true;
                        }
                        if let Some(warm_remote) = &mut self.warm_remote {
                            warm_remote.observe(&value);
                        }
                    }
                }
                let success =
                    self.success_status && !self.failed && (!self.streaming || self.terminal);
                let completion = self.completion.clone();
                if let Some(mut warm_remote) = self.warm_remote.take() {
                    warm_remote.finish(success, &completion);
                }
                self.completion.finish(success);
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::program_scheduling::ProgramSchedulerConfig;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn decode_target_is_resolved_from_live_discovery_after_admission() {
        let services = ServiceRegistry::new();
        services.register_test_instance(
            "old-d:8000".into(),
            "old-zmq".into(),
            ServiceType::Decode,
            1,
            Some(1),
        );
        let before_admission = discovered_decode_instances(&services, 1);
        services.register_test_instance(
            "new-d:8000".into(),
            "new-zmq".into(),
            ServiceType::Decode,
            1,
            Some(1),
        );
        let selected_id = discovered_target("new-d:8000").id;
        assert!(!before_admission
            .iter()
            .any(|(url, _)| discovered_target(url).id == selected_id));
        assert_eq!(
            discovered_decode_instance_by_id(&services, 1, &selected_id),
            Some(("new-d:8000".into(), "new-zmq".into()))
        );
    }

    #[tokio::test]
    async fn kv_observer_retries_soft_skipped_discovery() {
        use axum::{routing::get, Json, Router};

        let attempts = Arc::new(AtomicUsize::new(0));
        let count = attempts.clone();
        let app = Router::new()
            .route(
                "/get_server_info",
                get(|| async { Json(json!({"model_id":"test","kv_block_size":16})) }),
            )
            .route(
                "/kv_event_sources",
                get(move || {
                    let count = count.clone();
                    async move {
                        count.fetch_add(1, Ordering::SeqCst);
                        StatusCode::NOT_FOUND
                    }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let (supervisor, _signals) =
            KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
        let urls = BTreeSet::from([format!("http://{address}")]);
        let mut previous = BTreeSet::new();
        let mut subscribed = BTreeSet::new();

        refresh_kv_workers(&supervisor, &mut previous, &mut subscribed, urls.clone()).await;
        refresh_kv_workers(&supervisor, &mut previous, &mut subscribed, urls.clone()).await;
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert!(subscribed.is_empty());
        assert_eq!(previous, urls);

        refresh_kv_workers(&supervisor, &mut previous, &mut subscribed, BTreeSet::new()).await;
        assert!(previous.is_empty());
        server.abort();
    }

    #[test]
    fn token_id_prompts_count_tokens_without_json_digit_overhead() {
        assert_eq!(
            explicit_completion_tokens(&json!({"prompt":[1,100000,248319]}), "/v1/completions"),
            Some(3)
        );
        assert_eq!(
            explicit_completion_tokens(&json!({"prompt":[[1,100000],[2,3,4]]}), "/v1/completions"),
            Some(5)
        );
        for prompt in [
            json!("text"),
            json!([]),
            json!([1, -2]),
            json!([1, 1.5]),
            json!([4294967296u64]),
            json!([[1, 2], [-3]]),
            json!([[1, 2], [3.5]]),
        ] {
            assert_eq!(
                explicit_completion_tokens(&json!({"prompt":prompt}), "/v1/completions"),
                None
            );
        }
        assert_eq!(
            explicit_completion_tokens(&json!({"prompt":[1,2]}), "/v1/chat/completions"),
            None
        );
    }

    #[test]
    fn generated_choices_are_included_in_pd_output_reservation() {
        const DEFAULT_OUTPUT: usize = 100;
        let identity = || {
            ProgramIdentity::from_request(
                None,
                Some(&json!({"vllm_xargs":{"agentic_context":{
                    "program_id":"budget","expected_resume":true
                }}})),
                Some("model"),
            )
            .unwrap()
            .unwrap()
        };

        let mut request = identity();
        reserve_output_budget(
            &mut request,
            &json!({"max_tokens":256,"n":4}),
            "/v1/chat/completions",
            DEFAULT_OUTPUT,
        );
        assert_eq!(request.request_hints().expected_output_tokens, Some(1024));

        let mut request = identity();
        reserve_output_budget(
            &mut request,
            &json!({"max_completion_tokens":128,"n":2}),
            "/v1/chat/completions",
            DEFAULT_OUTPUT,
        );
        assert_eq!(request.request_hints().expected_output_tokens, Some(256));

        let mut request = identity();
        reserve_output_budget(
            &mut request,
            &json!({"max_tokens":256,"n":2,"best_of":4}),
            "/v1/completions",
            DEFAULT_OUTPUT,
        );
        assert_eq!(request.request_hints().expected_output_tokens, Some(1024));

        let mut request = identity();
        reserve_output_budget(
            &mut request,
            &json!({"prompt":[[1,2],[3,4,5]],"max_tokens":32,"n":2,"best_of":3}),
            "/v1/completions",
            DEFAULT_OUTPUT,
        );
        assert_eq!(request.request_hints().expected_output_tokens, Some(192));

        let mut request = identity();
        reserve_output_budget(
            &mut request,
            &json!({"prompt":["first","second","third"]}),
            "/v1/completions",
            DEFAULT_OUTPUT,
        );
        assert_eq!(request.request_hints().expected_output_tokens, Some(300));

        let mut request = identity();
        reserve_output_budget(
            &mut request,
            &json!({"max_tokens":256,"n":2,"best_of":4}),
            "/v1/chat/completions",
            DEFAULT_OUTPUT,
        );
        assert_eq!(request.request_hints().expected_output_tokens, Some(512));

        let mut request = identity();
        reserve_output_budget(
            &mut request,
            &json!({"n":4}),
            "/v1/chat/completions",
            DEFAULT_OUTPUT,
        );
        assert_eq!(request.request_hints().expected_output_tokens, Some(400));

        let mut request = identity();
        reserve_output_budget(
            &mut request,
            &json!({"n":1}),
            "/v1/chat/completions",
            DEFAULT_OUTPUT,
        );
        assert_eq!(request.request_hints().expected_output_tokens, None);

        let mut request = identity();
        reserve_output_budget(
            &mut request,
            &json!({"max_tokens":256,"n":0}),
            "/v1/chat/completions",
            DEFAULT_OUTPUT,
        );
        assert_eq!(request.request_hints().expected_output_tokens, Some(256));
    }

    #[test]
    fn explicit_dp_size_wins_over_configured_fallback() {
        assert_eq!(decode_instance_size(Some(2), 4), 2);
        assert_eq!(decode_instance_size(Some(1), 4), 1);
        assert_eq!(decode_instance_size(None, 4), 4);
        assert_eq!(decode_instance_size(None, 0), 1);
    }

    async fn completion() -> (Arc<ProgramScheduler>, ProgramCompletion) {
        let scheduler = Arc::new(ProgramScheduler::new(ProgramSchedulerConfig::default()));
        let request = json!({"vllm_xargs":{"agentic_context":{
            "program_id":"body-test","expected_resume":true,"kv_retention_ttl_seconds":2.0}}});
        let identity = ProgramIdentity::from_request(None, Some(&request), Some("test"))
            .unwrap()
            .unwrap();
        let target = ProgramTarget {
            id: "d@1".into(),
            base_url: "http://d".into(),
            dp_rank: Some(1),
        };
        let dispatch = scheduler
            .acquire(identity, 100, &[target], None)
            .await
            .unwrap();
        let estimator = Arc::new(MomentumTokenEstimator::default());
        let (_, calibration) =
            estimator.estimate(TokenEstimateScope::new("test", "/chat"), "hello");
        let completion =
            ProgramCompletion::new(scheduler.clone(), dispatch, estimator, calibration);
        (scheduler, completion)
    }

    #[test]
    fn model_and_credentials_isolate_identical_program_ids() {
        let mut a = HeaderMap::new();
        a.insert("authorization", "Bearer tenant-a".parse().unwrap());
        let mut b = HeaderMap::new();
        b.insert("authorization", "Bearer tenant-b".parse().unwrap());
        assert_ne!(
            model_pool(Some(&a), &json!({"model":"m"})),
            model_pool(Some(&b), &json!({"model":"m"}))
        );
        assert_ne!(
            model_pool(Some(&a), &json!({"model":"m"})),
            model_pool(Some(&a), &json!({"model":"m2"}))
        );
        assert!(!model_pool(Some(&a), &json!({"model":"m"})).contains("tenant-a"));
    }

    #[tokio::test]
    async fn sse_bytes_headers_and_usage_survive_fragmented_terminal() {
        let (scheduler, completion) = completion().await;
        let chunks = [
            "data: {\"usage\":{\"prompt_tokens\":120,\"completion_tokens\":7}}\n\n",
            "data: [DO",
            "NE]\n\n",
        ];
        let input = chunks.concat();
        let response = Response::builder()
            .header("x-backend", "preserved")
            .body(Body::from_stream(futures_util::stream::iter(
                chunks
                    .into_iter()
                    .map(|s| Ok::<_, std::io::Error>(Bytes::from(s))),
            )))
            .unwrap();
        let response = track_response(response, completion, true);
        assert_eq!(
            scheduler.diagnostics().ranks[0].programs[0].in_flight_requests,
            1
        );
        assert_eq!(response.headers()["x-backend"], "preserved");
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(bytes.as_ref(), input.as_bytes());
        let program = &scheduler.diagnostics().ranks[0].programs[0];
        assert_eq!(program.in_flight_requests, 0);
        assert_eq!(program.estimated_context_tokens, 120);
        assert_eq!(program.state, "active");
    }

    #[tokio::test]
    async fn oversized_sse_line_does_not_fail_completed_stream_or_hide_later_error() {
        let large_line = format!(
            "data: {{\"choices\":[{{\"delta\":{{\"content\":\"{}\"}}}}]}}\n\n",
            "x".repeat(PdTrackedBody::MAX_BUFFER)
        );
        let (first, remainder) = large_line.split_at(PdTrackedBody::MAX_BUFFER / 2);
        let (overflowing, rest) = remainder.split_at(PdTrackedBody::MAX_BUFFER + 1 - first.len());
        for (terminal, expected_state) in [
            ("data: [DONE]\n\n", "active"),
            (
                "data: {\"error\":\"backend failed\"}\n\ndata: [DONE]\n\n",
                "paused",
            ),
        ] {
            let (scheduler, completion) = completion().await;
            let chunks = [
                Bytes::copy_from_slice(first.as_bytes()),
                Bytes::copy_from_slice(overflowing.as_bytes()),
                Bytes::copy_from_slice(rest.as_bytes()),
                Bytes::copy_from_slice(terminal.as_bytes()),
            ];
            let response = track_response(
                Response::new(Body::from_stream(futures_util::stream::iter(
                    chunks.clone().into_iter().map(Ok::<_, std::io::Error>),
                ))),
                completion,
                true,
            );
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            assert_eq!(
                bytes.as_ref(),
                [first, overflowing, rest, terminal].concat().as_bytes()
            );
            let program = &scheduler.diagnostics().ranks[0].programs[0];
            assert_eq!(program.state, expected_state);
            assert_eq!(program.in_flight_requests, 0);
        }
    }

    #[tokio::test]
    async fn partial_sse_eof_and_error_events_are_failures() {
        for data in [
            "data: {\"choices\":[]}\n\n",
            "data: {\"error\":\"backend failed\"}\n\ndata: [DONE]\n\n",
        ] {
            let (scheduler, completion) = completion().await;
            let response = track_response(Response::new(Body::from(data)), completion, true);
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            assert_eq!(scheduler.diagnostics().ranks[0].programs[0].state, "paused");
            assert_eq!(
                scheduler.diagnostics().ranks[0].programs[0].in_flight_requests,
                0
            );
        }
    }

    #[tokio::test]
    async fn body_drop_settles_once_and_late_finish_cannot_reverse_failure() {
        let (scheduler, completion) = completion().await;
        let late = completion.clone();
        let response = track_response(
            Response::new(Body::from("data: [DONE]\n\n")),
            completion,
            true,
        );
        drop(response);
        // The final completion owner is a clone held here: release it to trigger Drop.
        late.finish(false);
        late.finish(true);
        assert_eq!(scheduler.diagnostics().ranks[0].programs[0].state, "paused");
        assert_eq!(
            scheduler.diagnostics().ranks[0].programs[0].in_flight_requests,
            0
        );
    }

    #[tokio::test]
    async fn json_body_success_and_error_status_settle_after_eof() {
        for status in [StatusCode::OK, StatusCode::BAD_GATEWAY] {
            let (scheduler, completion) = completion().await;
            let response = track_response(
                Response::builder()
                    .status(status)
                    .body(Body::from(
                        r#"{"usage":{"prompt_tokens":110,"completion_tokens":8}}"#,
                    ))
                    .unwrap(),
                completion,
                false,
            );
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let snapshot = scheduler.diagnostics();
            assert_eq!(snapshot.ranks[0].programs[0].in_flight_requests, 0);
            assert_eq!(
                snapshot.ranks[0].programs[0].state,
                if status.is_success() {
                    "active"
                } else {
                    "paused"
                }
            );
        }
    }
}
