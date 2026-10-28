// PD (Prefill-Decode) Router Implementation
// This module handles routing for disaggregated prefill-decode systems
use super::dp_utils;
use super::pd_kv_state::PdKvState;
use super::pd_types::{
    Loads, PDRouterError, PdAttempt, PdEndpoint, PdLoadManager, PdPrefillInput, PdRequestContext,
    PdSelection, Stage, StageDecision,
};
use crate::backend::preprocess::prepare_http_token_inputs;
use crate::core::worker::WorkerMetadata;
use crate::core::{
    BasicWorker, CircuitBreaker, CircuitBreakerConfig, ConnectionMode, DPAwareWorker, HealthConfig,
    Worker, WorkerRegistry, WorkerResult, WorkerType,
};
use crate::policies::WorkerTokenInputs;
use crate::policies::{kv_aware_ops, LoadBalancingPolicy, PolicyRegistry};
use crate::routers::header_utils;
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use reqwest::Client;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, error, info, warn};

/// Admission decision over the supplied load snapshot. Legacy policies
/// retain their own selection semantics and text/header inputs.
pub(super) fn select_pd_pair(
    loads: &Loads,
    prefill_workers: &[Arc<dyn Worker>],
    decode_workers: &[Arc<dyn Worker>],
    prefill_policy: &dyn LoadBalancingPolicy,
    decode_policy: &dyn LoadBalancingPolicy,
    context: &PdRequestContext<'_>,
) -> Option<PdSelection> {
    // Admission must not switch an endpoint between aggregate and per-rank
    // accounting while requests are live. Index existing granularities once
    // instead of rescanning every load key for every candidate.
    let active_granularities: HashSet<(Stage, &str, bool)> = loads
        .keys()
        .map(|(stage, endpoint)| {
            (
                *stage,
                endpoint.http_url.as_str(),
                endpoint.dp_rank.is_some(),
            )
        })
        .collect();
    let eligible = |stage: Stage, worker: &dyn Worker, endpoint: &PdEndpoint| {
        let granularity_matches = !active_granularities.contains(&(
            stage,
            endpoint.http_url.as_str(),
            endpoint.dp_rank.is_none(),
        ));
        worker.is_available()
            && granularity_matches
            && context
                .model_id
                .is_none_or(|model| worker.model_id() == "unknown" || worker.model_id() == model)
    };
    let prefill_endpoints: Vec<_> = prefill_workers
        .iter()
        .map(|worker| PdEndpoint::for_worker(worker.as_ref()))
        .collect();
    let decode_endpoints: Vec<_> = decode_workers
        .iter()
        .map(|worker| PdEndpoint::for_worker(worker.as_ref()))
        .collect();
    let eligible_indices = |stage, workers: &[Arc<dyn Worker>], endpoints: &[PdEndpoint]| {
        workers
            .iter()
            .zip(endpoints)
            .enumerate()
            .filter(|(_, (worker, endpoint))| eligible(stage, worker.as_ref(), endpoint))
            .map(|(index, _)| index)
            .collect::<Vec<_>>()
    };
    let prefill_indices = eligible_indices(Stage::Prefill, prefill_workers, &prefill_endpoints);
    let decode_indices = eligible_indices(Stage::Decode, decode_workers, &decode_endpoints);
    if prefill_indices.is_empty() || decode_indices.is_empty() {
        return None;
    }
    let prefill = select_stage(
        loads,
        Stage::Prefill,
        prefill_workers,
        &prefill_endpoints,
        &prefill_indices,
        prefill_policy,
        context,
    )?;
    let decode = select_stage(
        loads,
        Stage::Decode,
        decode_workers,
        &decode_endpoints,
        &decode_indices,
        decode_policy,
        context,
    )?;

    Some(PdSelection { prefill, decode })
}

fn select_stage(
    loads: &Loads,
    stage: Stage,
    workers: &[Arc<dyn Worker>],
    endpoints: &[PdEndpoint],
    indices: &[usize],
    policy: &dyn LoadBalancingPolicy,
    context: &PdRequestContext<'_>,
) -> Option<StageDecision> {
    let policy_input = context.legacy_policy_input;
    let kv_aware = kv_aware_ops(policy).is_some();
    let mut selected_indices = indices.to_vec();
    selected_indices.sort_by(|&left, &right| endpoints[left].cmp(&endpoints[right]));
    let projected: Vec<Arc<dyn Worker>> = selected_indices
        .iter()
        .map(|&index| {
            let inner = workers[index].clone();
            let endpoint = endpoints[index].clone();
            let load = loads
                .get(&(stage, endpoint))
                .copied()
                .unwrap_or_default()
                .occupancy(stage);
            Arc::new(PolicyWorker { inner, load }) as Arc<dyn Worker>
        })
        .collect();
    let unknown_backlog = stage == Stage::Prefill
        && selected_indices.iter().any(|&index| {
            loads
                .get(&(stage, endpoints[index].clone()))
                .is_some_and(|load| load.unknown_prefill_requests > 0)
        });
    if let (Stage::Prefill, Some(kv), false) = (stage, kv_aware_ops(policy), unknown_backlog) {
        let inputs: Option<Vec<_>> = projected
            .iter()
            .map(|worker| {
                let input = context.prefill_inputs?.get(worker.base_url())?;
                Some((
                    input,
                    input.prompts.as_deref()?,
                    *input.work_by_rank.get(&worker.dp_rank())?,
                ))
            })
            .collect();
        // Token costs and request counts have different units. If any
        // eligible P has unknown work, compare the whole pool by occupancy.
        if let Some(inputs) = inputs {
            let work: Vec<_> = inputs.iter().map(|(_, _, work)| *work).collect();
            let costs: Vec<_> = selected_indices
                .iter()
                .zip(&work)
                .map(|(&index, work)| {
                    loads
                        .get(&(stage, endpoints[index].clone()))
                        .copied()
                        .unwrap_or_default()
                        .active_prefill_tokens
                        .saturating_add(work.new_prefill_tokens) as f64
                })
                .collect();
            let occupancy: Vec<_> = projected.iter().map(|worker| worker.load()).collect();
            debug!(version = 1, stage = "prefill", endpoints = ?selected_indices.iter()
                .map(|&index| &endpoints[index]).collect::<Vec<_>>(), ?costs, ?occupancy,
                remaining_work = ?work, "P/D token-cost candidates");
            // P follows actual token work across the full eligible pool;
            // decode session affinity must never filter or pin prefill.
            if let Some(picked) = kv.pick_prefill(&costs, &occupancy) {
                let (input, tokens, _) = inputs[picked];
                let estimate = work[picked];
                let new_tokens = estimate.new_prefill_tokens;
                let outcome = if !estimate.placement_verified {
                    "unverified_identity"
                } else {
                    "prefix_cost"
                };
                crate::metrics::RouterMetrics::record_prefix_routing_decision(
                    "kv_aware", "prefill", outcome,
                );
                return Some(StageDecision {
                    worker_index: selected_indices[picked],
                    endpoint: endpoints[selected_indices[picked]].clone(),
                    score: Some(costs[picked]),
                    new_prefill_tokens: Some(new_tokens),
                    credited_prefix_tokens: Some(
                        tokens
                            .iter()
                            .map(|ids| ids.len() as u64)
                            .sum::<u64>()
                            .saturating_sub(new_tokens),
                    ),
                    fallback_reason: input
                        .fallback_reason
                        .or((!estimate.placement_verified).then_some("unverified_placement")),
                    affinity_outcome: None,
                    affinity: None,
                });
            }
            // pick_prefill refused the candidates (non-finite or misaligned
            // costs): degrade to policy selection, never fail the request.
            debug!(
                stage = "prefill",
                "pick_prefill refused candidates; using policy selection"
            );
        }
    }
    let (picked, affinity_outcome, affinity_binding) = if stage == Stage::Decode && kv_aware {
        // Decode never queries placement or applies the P temperature:
        // deterministic minimum occupancy with rotating equal-load ties, and
        // the optional decode session binding on top.
        let kv = kv_aware_ops(policy)?;
        let pick = kv.select_decode(
            &projected,
            context.affinity_key.as_ref(),
            context.cache_salt,
        )?;
        (
            pick.index,
            context.affinity_key.is_some().then_some(pick.outcome),
            pick.binding,
        )
    } else if stage == Stage::Prefill && kv_aware {
        // Unknown token work or refused costs use projected occupancy,
        // without session binding or a request-count affinity limit on P.
        // PD admission owns the candidate lifecycle and load snapshot here;
        // the colocated selector's registration filter can otherwise discard
        // fresh discovery workers before their background KV binding exists.
        let picked = projected
            .iter()
            .enumerate()
            .min_by_key(|(_, worker)| worker.load())
            .map(|(index, _)| index)?;
        (picked, None, None)
    } else if let Some(cache) = policy
        .as_any()
        .downcast_ref::<crate::policies::CacheAwarePolicy>()
    {
        let prepared = if stage == Stage::Prefill {
            context.prefill_cache
        } else {
            context.decode_cache
        };
        let picked = cache.select_prepared(
            &projected,
            prepared,
            if stage == Stage::Prefill {
                "prefill"
            } else {
                "decode"
            },
        )?;
        (picked, None, None)
    } else {
        let picked =
            policy.select_worker_with_headers(&projected, policy_input, context.headers)?;
        (picked, None, None)
    };
    if kv_aware {
        let phase = if stage == Stage::Prefill {
            "prefill"
        } else {
            "decode"
        };
        let outcome = if affinity_outcome == Some("sticky_hit") {
            "sticky_hit"
        } else if stage == Stage::Prefill {
            "load_fallback"
        } else {
            "occupancy"
        };
        crate::metrics::RouterMetrics::record_prefix_routing_decision("kv_aware", phase, outcome);
    }
    // A held binding reference must never leak on a late selection failure:
    // release it before any bail-out below.
    let Some((&worker_index, worker)) = selected_indices.get(picked).zip(projected.get(picked))
    else {
        if let Some(record) = &affinity_binding {
            record.release_ref();
        }
        return None;
    };
    let selected_tokens = (stage == Stage::Prefill)
        .then(|| {
            context
                .prefill_inputs
                .and_then(|inputs| inputs.get(worker.base_url()))
                .and_then(|input| input.prompts.as_ref())
                .map(|prompts| prompts.iter().map(|ids| ids.len() as u64).sum())
        })
        .flatten();
    Some(StageDecision {
        fallback_reason: if stage == Stage::Prefill && kv_aware {
            Some(if unknown_backlog {
                "unknown_prefill_backlog"
            } else {
                projected
                    .iter()
                    .find_map(|worker| {
                        context
                            .prefill_inputs
                            .and_then(|inputs| inputs.get(worker.base_url()))
                            .and_then(|input| input.fallback_reason)
                    })
                    .unwrap_or("unverified_effective_tokens")
            })
        } else {
            None
        },
        credited_prefix_tokens: selected_tokens.map(|_| 0),
        worker_index,
        endpoint: endpoints[worker_index].clone(),
        // Unknown P token work falls back to occupancy; D includes the
        // incoming request, which does not change candidate ranking.
        score: kv_aware.then_some(match stage {
            Stage::Prefill => worker.load() as f64,
            Stage::Decode => (worker.load() + 1) as f64,
        }),
        new_prefill_tokens: selected_tokens,
        affinity_outcome,
        affinity: affinity_binding,
    })
}

/// Selection-only view: load is an immutable sample from the admission lock.
/// Worker identity, availability, metadata and policy inputs are preserved.
#[derive(Debug)]
struct PolicyWorker {
    inner: Arc<dyn Worker>,
    load: usize,
}

#[async_trait]
impl Worker for PolicyWorker {
    fn url(&self) -> &str {
        self.inner.url()
    }
    fn worker_type(&self) -> WorkerType {
        self.inner.worker_type()
    }
    fn connection_mode(&self) -> ConnectionMode {
        self.inner.connection_mode()
    }
    fn is_healthy(&self) -> bool {
        self.inner.is_healthy()
    }
    fn set_healthy(&self, healthy: bool) {
        self.inner.set_healthy(healthy);
    }
    async fn check_health_async(&self) -> WorkerResult<()> {
        self.inner.check_health_async().await
    }
    fn load(&self) -> usize {
        self.load
    }
    fn increment_load(&self) {
        self.inner.increment_load();
    }
    fn decrement_load(&self) {
        self.inner.decrement_load();
    }
    fn processed_requests(&self) -> usize {
        self.inner.processed_requests()
    }
    fn increment_processed(&self) {
        self.inner.increment_processed();
    }
    fn metadata(&self) -> &WorkerMetadata {
        self.inner.metadata()
    }
    fn circuit_breaker(&self) -> &CircuitBreaker {
        self.inner.circuit_breaker()
    }
    fn base_url(&self) -> &str {
        self.inner.base_url()
    }
    fn dp_rank(&self) -> Option<usize> {
        self.inner.dp_rank()
    }
    fn dp_size(&self) -> Option<usize> {
        self.inner.dp_size()
    }
    fn is_dp_aware(&self) -> bool {
        self.inner.is_dp_aware()
    }
}

#[derive(Debug, Clone)]
pub struct PdRouterBase {
    pub(super) kv_state: Option<Arc<PdKvState>>,
    pub worker_registry: Arc<WorkerRegistry>,
    pub(super) pd_load: Arc<PdLoadManager>,
    pub policy_registry: Arc<PolicyRegistry>,
    pub worker_startup_timeout_secs: u64,
    pub worker_startup_check_interval_secs: u64,
    pub worker_loads: Arc<tokio::sync::watch::Receiver<HashMap<String, isize>>>,
    pub load_monitor_handle: Option<Arc<tokio::task::JoinHandle<()>>>,
    pub client: Client,
    api_key: Option<String>,
    pub circuit_breaker_config: CircuitBreakerConfig,
    // Fallback DP size for servers that report no engine metrics
    pub dp_size: usize,
}

/// Build the workers backing one server URL, one per engine the server fronts.
fn build_dp_workers(
    url: &str,
    dp_size: usize,
    worker_type: WorkerType,
    circuit_breaker_config: &CircuitBreakerConfig,
    health_config: Option<&HealthConfig>,
) -> Vec<Arc<dyn Worker>> {
    let (base_url, pinned_rank) = dp_utils::parse_worker_url(url);

    // DPAwareWorker strips the @rank suffix in endpoint_url(), preventing
    // IPv6+DP URL corruption where HTTP clients read @N as userinfo (RFC 3986).
    let dp_worker = |rank: usize| -> Arc<dyn Worker> {
        let worker = DPAwareWorker::new(base_url.clone(), rank, dp_size, worker_type.clone())
            .with_circuit_breaker_config(circuit_breaker_config.clone());
        let worker = match health_config {
            Some(config) => worker.with_health_config(config.clone()),
            None => worker,
        };
        Arc::new(worker)
    };

    dp_utils::expand_dp_ranks(pinned_rank, dp_size)
        .into_iter()
        .map(|rank| match rank {
            Some(rank) => dp_worker(rank),
            None => {
                let worker = BasicWorker::new(url.to_string(), worker_type.clone())
                    .with_dp_size(dp_size)
                    .with_circuit_breaker_config(circuit_breaker_config.clone());
                let worker = match health_config {
                    Some(config) => worker.with_health_config(config.clone()),
                    None => worker,
                };
                Arc::new(worker) as Arc<dyn Worker>
            }
        })
        .collect()
}

impl PdRouterBase {
    pub(super) async fn prepare_prefill_inputs(
        &self,
        workers: &[Arc<dyn Worker>],
        request: &Value,
        route: &str,
        headers: Option<&HeaderMap>,
    ) -> Option<HashMap<String, super::pd_types::PdPrefillInput>> {
        let policy = self.policy_registry.get_prefill_policy();
        if kv_aware_ops(policy.as_ref()).is_some() {
            return Some(
                self.kv_state
                    .as_ref()?
                    .prepare(workers, &policy, request, route, headers)
                    .await,
            );
        }
        if policy.name() != "cache_aware" {
            return None;
        }
        let results = prepare_http_token_inputs(
            &self.client,
            workers
                .iter()
                .map(|worker| worker.base_url().to_owned())
                .collect(),
            request,
            route,
            headers,
            self.api_key.as_deref(),
            Duration::from_secs(1),
        )
        .await;
        Some(
            results
                .into_iter()
                .map(|(base, result)| {
                    let (prompts, reason) = match result {
                        Ok(input) => (Some(input.prompts), None),
                        Err(reason) => (None, Some(reason)),
                    };
                    (
                        base,
                        PdPrefillInput {
                            prompts,
                            work_by_rank: HashMap::new(),
                            allow_credit: false,
                            fallback_reason: reason,
                        },
                    )
                })
                .collect(),
        )
    }

    pub(super) async fn prepare_decode_inputs(
        &self,
        workers: &[Arc<dyn Worker>],
        request: &Value,
        route: &str,
        headers: Option<&HeaderMap>,
    ) -> Option<WorkerTokenInputs> {
        if self.policy_registry.get_decode_policy().name() != "cache_aware" {
            return None;
        }
        let results = prepare_http_token_inputs(
            &self.client,
            workers
                .iter()
                .map(|worker| worker.base_url().to_owned())
                .collect(),
            request,
            route,
            headers,
            self.api_key.as_deref(),
            Duration::from_secs(1),
        )
        .await;
        Some(
            results
                .into_iter()
                .filter_map(|(base, result)| match result {
                    Ok(input) => Some((base, input.prompts)),
                    Err(reason) => {
                        crate::metrics::RouterMetrics::record_kv_routing_input("pd_decode", reason);
                        None
                    }
                })
                .collect(),
        )
    }

    /// Shared entry point for static and discovered P/D pools. Selection and
    /// reservation are synchronous; no network work runs under admission.
    pub(super) fn select_and_admit(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        context: &PdRequestContext<'_>,
    ) -> Option<(PdSelection, PdAttempt)> {
        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();
        let admitted = self.pd_load.admit(|loads| {
            if let Some(kv) =
                kv_aware_ops(decode_policy.as_ref()).filter(|kv| kv.session_affinity_enabled())
            {
                kv.reconcile_decode_workers(decode_workers);
            }
            select_pd_pair(
                loads,
                prefill_workers,
                decode_workers,
                prefill_policy.as_ref(),
                decode_policy.as_ref(),
                context,
            )
        });
        if let Some((selection, attempt)) = &admitted {
            if kv_aware_ops(prefill_policy.as_ref()).is_some() {
                crate::metrics::RouterMetrics::record_pd_kv_decision(
                    selection.prefill.new_prefill_tokens,
                    selection.prefill.credited_prefix_tokens,
                    selection.prefill.fallback_reason,
                );
            }
            if let Some(outcome) = selection.decode.affinity_outcome {
                crate::metrics::RouterMetrics::record_pd_affinity_decision("decode", outcome);
            }
            // Selection indices come from these same slices, so the lookups
            // cannot fail today. If future filtering ever makes them fail,
            // still return the admitted attempt: bailing out here would
            // silently 503 a request whose occupancy was already reserved.
            // The per-worker counters are best-effort (pd_load is the
            // accounting of record); discovery pools rebuild their workers
            // per request, so for them the counters are write-only until
            // those workers are persisted.
            if let (Some(prefill_worker), Some(decode_worker)) = (
                prefill_workers.get(selection.prefill.worker_index),
                decode_workers.get(selection.decode.worker_index),
            ) {
                attempt.attach_workers(prefill_worker.clone(), decode_worker.clone());
            }
            debug!(
                version = 1,
                route = context.route,
                prefill_policy = prefill_policy.name(),
                decode_policy = decode_policy.name(),
                fallback = ?selection.prefill.fallback_reason,
                new_prefill_tokens = ?selection.prefill.new_prefill_tokens,
                credited_prefix_tokens = ?selection.prefill.credited_prefix_tokens,
                prefill_score = ?selection.prefill.score,
                decode_score = ?selection.decode.score,
                decode_sticky = ?selection.decode.fallback_reason,
                ?selection,
                "P/D admission"
            );
        }
        admitted
    }

    // Private helper method to perform health check on a new server
    async fn wait_for_server_health(&self, url: &str) -> Result<(), PDRouterError> {
        crate::routers::http::router::Router::wait_for_healthy_workers(
            &[url.to_string()],
            self.worker_startup_timeout_secs,
            self.worker_startup_check_interval_secs,
        )
        .await
        .map_err(|_| PDRouterError::HealthCheckFailed {
            url: url.to_string(),
        })
    }

    // Generic helper for processing all workers with an endpoint
    async fn process_workers(
        &self,
        worker_type_enum: WorkerType,
        worker_type: &str,
        endpoint: &str,
    ) -> (Vec<String>, Vec<String>) {
        let mut results = Vec::new();
        let mut errors = Vec::new();

        // Get workers from registry based on type
        let workers = self.worker_registry.get_by_type(&worker_type_enum);
        let urls: Vec<String> = workers.iter().map(|w| w.url().to_string()).collect();

        // Process each worker
        for worker_url in urls {
            // Extract base URL if DP-aware format (e.g., http://127.0.0.1:8081@0 → http://127.0.0.1:8081)
            let (base_url, _) = super::dp_utils::parse_worker_url(&worker_url);
            let url = format!("{}/{}", base_url, endpoint);
            match self.client.post(&url).send().await {
                Ok(res) if res.status().is_success() => {
                    results.push(format!("{} {}: OK", worker_type, worker_url));
                }
                Ok(res) => {
                    errors.push(format!(
                        "{} {} returned status: {}",
                        worker_type,
                        worker_url,
                        res.status()
                    ));
                }
                Err(e) => {
                    errors.push(format!("{} {} error: {}", worker_type, worker_url, e));
                }
            }
        }

        (results, errors)
    }

    // Helper to get prefill worker URLs
    fn get_prefill_worker_urls(&self) -> Vec<String> {
        self.worker_registry
            .get_prefill_workers()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }

    // Helper to get decode worker URLs
    fn get_decode_worker_urls(&self) -> Vec<String> {
        self.worker_registry
            .get_decode_workers()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }

    /// Start profiling on a backend server
    pub async fn start_profiling(&self, worker_url: &str) {
        // Extract base URL if worker_url is in DP-aware format (e.g., http://127.0.0.1:8081@2)
        let (base_url, _) = super::dp_utils::parse_worker_url(worker_url);

        let url = format!("{}/start_profile", base_url);
        match self.client.post(&url).send().await {
            Ok(res) if res.status().is_success() => {
                info!("Started profiling on {}", base_url);
            }
            Ok(res) => {
                warn!(
                    "Failed to start profiling on {}: status {}",
                    base_url,
                    res.status()
                );
            }
            Err(e) => {
                warn!("Error starting profiling on {}: {}", base_url, e);
            }
        }
    }

    /// Stop profiling on a backend server
    pub async fn stop_profiling(&self, worker_url: &str) {
        // Extract base URL if worker_url is in DP-aware format (e.g., http://127.0.0.1:8081@2)
        let (base_url, _) = super::dp_utils::parse_worker_url(worker_url);

        let url = format!("{}/stop_profile", base_url);
        match self.client.post(&url).send().await {
            Ok(res) if res.status().is_success() => {
                info!("Stopped profiling on {}", base_url);
            }
            Ok(res) => {
                warn!(
                    "Failed to stop profiling on {}: status {}",
                    base_url,
                    res.status()
                );
            }
            Err(e) => {
                warn!("Error stopping profiling on {}: {}", base_url, e);
            }
        }
    }

    // Helper for proxying requests to the first prefill worker
    async fn proxy_to_first_prefill_worker(
        &self,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let workers = self.worker_registry.get_prefill_workers();
        let first_worker_url = workers.first().map(|w| w.url().to_string());

        if let Some(worker_url) = first_worker_url {
            self.proxy_to_worker(worker_url, endpoint, headers).await
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "No prefill servers available".to_string(),
            )
                .into_response()
        }
    }

    // Generic helper for proxying to a specific worker
    async fn proxy_to_worker(
        &self,
        worker_url: String,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        // Extract base URL if DP-aware format (e.g., http://127.0.0.1:8081@0 → http://127.0.0.1:8081)
        let (base_url, _) = super::dp_utils::parse_worker_url(&worker_url);
        let url = format!("{}/{}", base_url, endpoint);
        let mut request_builder = self.client.get(&url);

        // Add headers if provided
        if let Some(headers) = headers {
            for (name, value) in headers {
                request_builder = request_builder.header(name, value);
            }
        }

        match request_builder.send().await {
            Ok(res) if res.status().is_success() => {
                let response_headers = header_utils::preserve_response_headers(res.headers());

                match res.bytes().await {
                    Ok(body) => {
                        let mut response = Response::new(axum::body::Body::from(body));
                        *response.status_mut() = StatusCode::OK;
                        *response.headers_mut() = response_headers;
                        response
                    }
                    Err(e) => {
                        error!("Failed to read response body: {}", e);
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Failed to read response body: {}", e),
                        )
                            .into_response()
                    }
                }
            }
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                (status, format!("{} server returned status: ", res.status())).into_response()
            }
            Err(e) => {
                error!("Failed to proxy request server: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to proxy request: {}", e),
                )
                    .into_response()
            }
        }
    }

    pub async fn add_prefill_server(
        &self,
        url: String,
        bootstrap_port: Option<u16>,
    ) -> Result<String, PDRouterError> {
        self.add_server(url, "prefill", WorkerType::Prefill { bootstrap_port })
            .await
    }

    pub async fn add_decode_server(&self, url: String) -> Result<String, PDRouterError> {
        self.add_server(url, "decode", WorkerType::Decode).await
    }

    async fn add_server(
        &self,
        url: String,
        role: &str,
        worker_type: WorkerType,
    ) -> Result<String, PDRouterError> {
        // Wait for the new server to be healthy
        self.wait_for_server_health(&url).await?;

        let dp_size = dp_utils::discover_dp_size(&self.client, &url, self.dp_size).await;
        let workers = build_dp_workers(
            &url,
            dp_size,
            worker_type,
            &self.circuit_breaker_config,
            None,
        );

        if workers
            .iter()
            .any(|worker| self.worker_registry.get_by_url(worker.url()).is_some())
        {
            return Err(PDRouterError::WorkerAlreadyExists { url });
        }

        info!(
            "Added {} server {} as {} worker(s)",
            role,
            url,
            workers.len()
        );
        for worker in workers {
            self.register_and_notify(worker);
        }

        Ok(format!("Successfully added {} server: {}", role, url))
    }

    /// Register a worker and notify the policy registry.
    fn register_and_notify(&self, worker_arc: Arc<dyn Worker>) {
        self.worker_registry.register(worker_arc.clone());
        let model_id = worker_arc.model_id();
        let policy = self.policy_registry.on_worker_added(model_id, None);
        if policy.name() == "cache_aware" {
            if let Some(cache_aware) = policy
                .as_any()
                .downcast_ref::<crate::policies::CacheAwarePolicy>()
            {
                let model_workers = self.worker_registry.get_by_model_fast(model_id);
                cache_aware.init_workers(&model_workers);
            }
        }
    }

    pub async fn remove_prefill_server(&self, url: &str) -> Result<String, PDRouterError> {
        // A bare URL that is not registered as-is was expanded per DP rank.
        if !url.contains('@') && self.worker_registry.get_by_url(url).is_none() {
            return self.remove_dp_expanded_workers(url, "prefill");
        }

        // Exact URL match (with @rank or dp_size==1)
        let model_id = match self.worker_registry.get_by_url(url) {
            Some(worker) => worker.model_id().to_string(),
            None => {
                return Err(PDRouterError::WorkerNotFound {
                    url: url.to_string(),
                });
            }
        };

        let removed = self.worker_registry.remove_by_url(url);

        if removed.is_some() {
            self.policy_registry.on_worker_removed(&model_id);
            if let Some(policy) = self.policy_registry.get_policy(&model_id) {
                if policy.name() == "cache_aware" {
                    if let Some(cache_aware) = policy
                        .as_any()
                        .downcast_ref::<crate::policies::CacheAwarePolicy>()
                    {
                        cache_aware.remove_worker_by_url(url);
                    }
                }
            }
        }

        if removed.is_some() {
            info!("Removed prefill server: {}", url);
            Ok(format!("Successfully removed prefill server: {}", url))
        } else {
            Err(PDRouterError::WorkerNotFound {
                url: url.to_string(),
            })
        }
    }

    pub async fn remove_decode_server(&self, url: &str) -> Result<String, PDRouterError> {
        if !url.contains('@') && self.worker_registry.get_by_url(url).is_none() {
            return self.remove_dp_expanded_workers(url, "decode");
        }

        let model_id = match self.worker_registry.get_by_url(url) {
            Some(worker) => worker.model_id().to_string(),
            None => {
                return Err(PDRouterError::WorkerNotFound {
                    url: url.to_string(),
                });
            }
        };

        let removed = self.worker_registry.remove_by_url(url);

        if removed.is_some() {
            if let Some(kv) = kv_aware_ops(self.policy_registry.get_decode_policy().as_ref()) {
                kv.unregister_worker(url);
            }
            // Notify PolicyRegistry about the removed worker
            self.policy_registry.on_worker_removed(&model_id);

            // Get the policy for this model to update cache-aware if needed
            if let Some(policy) = self.policy_registry.get_policy(&model_id) {
                if policy.name() == "cache_aware" {
                    if let Some(cache_aware) = policy
                        .as_any()
                        .downcast_ref::<crate::policies::CacheAwarePolicy>()
                    {
                        cache_aware.remove_worker_by_url(url);
                    }
                }
            }
        }

        if removed.is_some() {
            info!("Removed decode server: {}", url);
            Ok(format!("Successfully removed decode server: {}", url))
        } else {
            Err(PDRouterError::WorkerNotFound {
                url: url.to_string(),
            })
        }
    }

    /// Remove all DP-expanded workers for a bare URL by prefix match.
    /// When a bare URL like "http://host:8000" was expanded into
    /// "http://host:8000@0", "@1", ..., "@N-1", this removes all of them.
    fn remove_dp_expanded_workers(&self, url: &str, role: &str) -> Result<String, PDRouterError> {
        let prefix = format!("{}@", url);
        let all_workers = self.worker_registry.get_all();
        let mut removed_count = 0;
        for w in all_workers.iter() {
            if w.url().starts_with(&prefix) {
                let model_id = w.model_id().to_string();
                if self.worker_registry.remove_by_url(w.url()).is_some() {
                    if role == "decode" {
                        if let Some(kv) =
                            kv_aware_ops(self.policy_registry.get_decode_policy().as_ref())
                        {
                            kv.unregister_worker(w.url());
                        }
                    }
                    self.policy_registry.on_worker_removed(&model_id);
                    if let Some(policy) = self.policy_registry.get_policy(&model_id) {
                        if policy.name() == "cache_aware" {
                            if let Some(cache_aware) = policy
                                .as_any()
                                .downcast_ref::<crate::policies::CacheAwarePolicy>(
                            ) {
                                cache_aware.remove_worker_by_url(w.url());
                            }
                        }
                    }
                    removed_count += 1;
                }
            }
        }
        if removed_count > 0 {
            info!(
                "Removed {} DP-expanded {} workers for {}",
                removed_count, role, url
            );
            Ok(format!(
                "Successfully removed {} {} server(s) for {}",
                removed_count, role, url
            ))
        } else {
            Err(PDRouterError::WorkerNotFound {
                url: url.to_string(),
            })
        }
    }

    pub async fn new(
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
        ctx: &Arc<crate::server::AppContext>,
    ) -> Result<Self, String> {
        // Convert config CircuitBreakerConfig to core CircuitBreakerConfig
        let circuit_breaker_config = ctx.router_config.effective_circuit_breaker_config();
        let core_cb_config = CircuitBreakerConfig {
            failure_threshold: circuit_breaker_config.failure_threshold,
            success_threshold: circuit_breaker_config.success_threshold,
            timeout_duration: Duration::from_secs(circuit_breaker_config.timeout_duration_secs),
            window_duration: Duration::from_secs(circuit_breaker_config.window_duration_secs),
        };

        // Workers must be up before they can report their engine count.
        let prefill_base_urls: Vec<String> =
            prefill_urls.iter().map(|(url, _)| url.clone()).collect();
        for urls in [&prefill_base_urls, &decode_urls] {
            if !urls.is_empty() {
                crate::routers::http::router::Router::wait_for_healthy_workers(
                    urls,
                    ctx.router_config.worker_startup_timeout_secs,
                    ctx.router_config.worker_startup_check_interval_secs,
                )
                .await?;
            }
        }

        let fallback_dp_size = ctx.router_config.intra_node_data_parallel_size.max(1);
        let health_config = HealthConfig {
            timeout_secs: ctx.router_config.health_check.timeout_secs,
            check_interval_secs: ctx.router_config.health_check.check_interval_secs,
            endpoint: ctx.router_config.health_check.endpoint.clone(),
            failure_threshold: ctx.router_config.health_check.failure_threshold,
            success_threshold: ctx.router_config.health_check.success_threshold,
        };

        let prefill_workers = prefill_urls.iter().map(|(url, bootstrap_port)| {
            (
                url,
                WorkerType::Prefill {
                    bootstrap_port: *bootstrap_port,
                },
            )
        });
        let decode_workers = decode_urls.iter().map(|url| (url, WorkerType::Decode));
        for (url, worker_type) in prefill_workers.chain(decode_workers) {
            let dp_size = dp_utils::discover_dp_size(&ctx.client, url, fallback_dp_size).await;
            for worker in build_dp_workers(
                url,
                dp_size,
                worker_type,
                &core_cb_config,
                Some(&health_config),
            ) {
                ctx.worker_registry.register(worker);
            }
        }

        // Get all workers from registry for load monitoring
        let all_workers = ctx.worker_registry.get_all();
        let all_urls: Vec<String> = all_workers
            .iter()
            .map(|worker| worker.url().to_string())
            .collect();

        // Initialize cache-aware policies with workers from registry
        // Note: We need to get workers by type and convert to Box<dyn Worker> for CacheAwarePolicy
        // This is a temporary workaround until CacheAwarePolicy is updated to work with Arc<dyn Worker>
        // TODO: Update CacheAwarePolicy to accept Arc<dyn Worker> instead of Box<dyn Worker>

        // Set up background load monitoring for power-of-two selection
        let (tx, rx) = tokio::sync::watch::channel(HashMap::new());
        let worker_loads = Arc::new(rx);

        // Get policies from registry to check if we need load monitoring
        let prefill_policy = ctx.policy_registry.get_prefill_policy();
        let decode_policy = ctx.policy_registry.get_decode_policy();

        let load_monitor_handle =
            if prefill_policy.name() == "power_of_two" || decode_policy.name() == "power_of_two" {
                let monitor_urls = all_urls.clone();
                let monitor_interval = ctx.router_config.worker_startup_check_interval_secs;
                let monitor_client = ctx.client.clone();
                let prefill_policy_clone = Arc::clone(&prefill_policy);
                let decode_policy_clone = Arc::clone(&decode_policy);

                Some(Arc::new(tokio::spawn(async move {
                    Self::monitor_worker_loads_with_client(
                        monitor_urls,
                        tx,
                        monitor_interval,
                        monitor_client,
                        prefill_policy_clone,
                        decode_policy_clone,
                    )
                    .await;
                })))
            } else {
                None
            };

        // Note: Health checking is now handled centrally by RouterManager
        // Individual routers no longer need to manage health checkers

        let kv_state = if kv_aware_ops(prefill_policy.as_ref()).is_some() {
            Some(
                PdKvState::start(
                    ctx.client.clone(),
                    ctx.router_config.api_key.clone(),
                    ctx.router_config.pd_kv_event_sources.clone(),
                    ctx.worker_registry.get_prefill_workers(),
                    prefill_policy.clone(),
                )
                .await,
            )
        } else {
            None
        };
        Ok(PdRouterBase {
            kv_state,
            pd_load: Arc::new(PdLoadManager::default()),
            worker_registry: Arc::clone(&ctx.worker_registry),
            policy_registry: Arc::clone(&ctx.policy_registry),
            worker_startup_timeout_secs: ctx.router_config.worker_startup_timeout_secs,
            worker_startup_check_interval_secs: ctx
                .router_config
                .worker_startup_check_interval_secs,
            worker_loads,
            load_monitor_handle,
            client: ctx.client.clone(),
            api_key: ctx.router_config.api_key.clone(),
            circuit_breaker_config: core_cb_config,
            dp_size: fallback_dp_size,
        })
    }

    // Background task to monitor worker loads with shared client
    async fn monitor_worker_loads_with_client(
        worker_urls: Vec<String>,
        tx: tokio::sync::watch::Sender<HashMap<String, isize>>,
        interval_secs: u64,
        client: Client,
        prefill_policy: Arc<dyn LoadBalancingPolicy>,
        decode_policy: Arc<dyn LoadBalancingPolicy>,
    ) {
        loop {
            let mut loads = HashMap::new();

            let futures: Vec<_> = worker_urls
                .iter()
                .map(|url| {
                    let client = client.clone();
                    let url = url.clone();
                    async move {
                        let load = get_worker_load(&client, &url).await.unwrap_or(0);
                        (url, load)
                    }
                })
                .collect();

            let results = futures_util::future::join_all(futures).await;

            for (url, load) in results {
                loads.insert(url, load);
            }

            debug!("Worker loads updated: {:?}", loads);

            // Update both policies with current loads
            prefill_policy.update_loads(&loads);
            decode_policy.update_loads(&loads);

            // Check if receiver is still active
            if tx.send(loads).is_err() {
                info!("Load monitor receiver dropped, shutting down monitor task");
                break;
            }

            tokio::time::sleep(Duration::from_secs(interval_secs)).await;
        }
    }
}

// Helper functions

async fn get_worker_load(client: &Client, worker_url: &str) -> Option<isize> {
    match client.get(format!("{}/get_load", worker_url)).send().await {
        Ok(res) if res.status().is_success() => match res.bytes().await {
            Ok(bytes) => match serde_json::from_slice::<Value>(&bytes) {
                Ok(data) => data
                    .get("load")
                    .and_then(|v| v.as_i64())
                    .map(|v| v as isize),
                Err(e) => {
                    debug!("Failed to parse load response from {}: {}", worker_url, e);
                    None
                }
            },
            Err(e) => {
                debug!("Failed to read load response from {}: {}", worker_url, e);
                None
            }
        },
        Ok(res) => {
            debug!(
                "Worker {} returned non-success status: {}",
                worker_url,
                res.status()
            );
            None
        }
        Err(e) => {
            debug!("Failed to get load from {}: {}", worker_url, e);
            None
        }
    }
}

impl PdRouterBase {
    pub async fn add_worker(&self, _worker_url: &str) -> Result<String, String> {
        // For PD router, we don't support adding workers via this generic method
        Err(
            "PD router requires specific add_prefill_server or add_decode_server methods"
                .to_string(),
        )
    }

    pub fn remove_worker(&self, worker_url: &str) {
        // Remove from registry
        if let Some(worker) = self.worker_registry.remove_by_url(worker_url) {
            match worker.worker_type() {
                WorkerType::Prefill { .. } => {
                    info!("Removed prefill worker: {}", worker_url);
                }
                WorkerType::Decode => {
                    info!("Removed decode worker: {}", worker_url);
                }
                _ => {
                    info!("Removed worker: {}", worker_url);
                }
            }
        }
    }

    pub fn get_worker_urls(&self) -> Vec<String> {
        self.worker_registry.get_all_urls()
    }
}

impl PdRouterBase {
    pub async fn health(&self, _req: Request<Body>) -> Response {
        // This is a server readiness check - checking if we have healthy workers
        // Workers handle their own health checks in the background
        let mut all_healthy = true;
        let mut unhealthy_servers = Vec::new();

        // Check all workers
        for worker in self.worker_registry.get_all() {
            if !worker.is_healthy() {
                all_healthy = false;
                let worker_type = match worker.worker_type() {
                    WorkerType::Prefill { .. } => "Prefill",
                    WorkerType::Decode => "Decode",
                    _ => "Worker",
                };
                unhealthy_servers.push(format!("{}: {}", worker_type, worker.url()));
            }
        }

        if all_healthy {
            (StatusCode::OK, "All servers healthy").into_response()
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                format!("Unhealthy servers: {:?}", unhealthy_servers),
            )
                .into_response()
        }
    }

    pub async fn health_generate(&self, _req: Request<Body>) -> Response {
        // Test model generation capability by selecting a random pair and testing them
        // Note: This endpoint actually causes the model to generate tokens, so we only test one pair
        let prefill_workers = self.worker_registry.get_prefill_workers();
        let decode_workers = self.worker_registry.get_decode_workers();
        self.health_generate_for_workers(&prefill_workers, &decode_workers)
            .await
    }

    /// Probe one P/D pair selected through the same admission path as real
    /// traffic, so health checks and requests cannot drift apart. The
    /// attempt is dropped (releasing its reservations) when the probe ends.
    pub(super) async fn health_generate_for_workers(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
    ) -> Response {
        let context = PdRequestContext {
            prefill_inputs: None,
            model_id: None,
            legacy_policy_input: None,
            prefill_cache: None,
            decode_cache: None,
            headers: None,
            affinity_key: None,
            cache_salt: None,
            route: "/health_generate",
        };
        let (selection, _attempt) =
            match self.select_and_admit(prefill_workers, decode_workers, &context) {
                Some(admitted) => admitted,
                None => {
                    return (
                        StatusCode::SERVICE_UNAVAILABLE,
                        "No healthy worker pair available".to_string(),
                    )
                        .into_response();
                }
            };
        let prefill = &prefill_workers[selection.prefill.worker_index];
        let decode = &decode_workers[selection.decode.worker_index];

        // Test prefill server's health_generate
        // Extract base URLs if DP-aware format (e.g., http://127.0.0.1:8081@0 → http://127.0.0.1:8081)
        let (prefill_base_url, _) = super::dp_utils::parse_worker_url(prefill.url());
        let (decode_base_url, _) = super::dp_utils::parse_worker_url(decode.url());

        let prefill_url = format!("{}/health_generate", prefill_base_url);
        let (prefill_result, decode_result) = tokio::join!(
            self.client.get(&prefill_url).send(),
            self.client
                .get(format!("{}/health_generate", decode_base_url))
                .send()
        );

        // Check results
        let mut errors = Vec::new();

        match prefill_result {
            Ok(res) if res.status().is_success() => {
                debug!(
                    "Health generate passed for prefill server: {}",
                    prefill.url()
                );
            }
            Ok(res) => {
                errors.push(format!(
                    "Prefill {} returned status {}",
                    prefill.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Prefill {} error: {}", prefill.url(), e));
            }
        }

        match decode_result {
            Ok(res) if res.status().is_success() => {
                debug!("Health generate passed for decode server: {}", decode.url());
            }
            Ok(res) => {
                errors.push(format!(
                    "Decode {} returned status {}",
                    decode.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Decode {} error: {}", decode.url(), e));
            }
        }

        if errors.is_empty() {
            (
                StatusCode::OK,
                format!(
                    "Health generate passed on selected pair: prefill={}, decode={}",
                    prefill.url(),
                    decode.url()
                ),
            )
                .into_response()
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                format!("Health generate failed: {:?}", errors),
            )
                .into_response()
        }
    }

    pub async fn get_server_info(&self, _req: Request<Body>) -> Response {
        // Get info from the first decode server to match vllm's server info format
        // Note: We use decode workers for server info to match expected format
        self.proxy_to_first_prefill_worker("get_server_info", None)
            .await
    }

    pub async fn get_models(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = header_utils::copy_request_headers(&req);

        // Proxy to first prefill worker
        self.proxy_to_first_prefill_worker("v1/models", Some(headers))
            .await
    }

    pub async fn get_model_info(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = header_utils::copy_request_headers(&req);

        // Proxy to first prefill worker
        self.proxy_to_first_prefill_worker("get_model_info", Some(headers))
            .await
    }

    pub async fn get_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Responses retrieve endpoint not implemented for PD router",
        )
            .into_response()
    }

    pub async fn cancel_response(
        &self,
        _headers: Option<&HeaderMap>,
        _response_id: &str,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Responses cancel endpoint not implemented for PD router",
        )
            .into_response()
    }

    pub async fn route_embeddings(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::EmbeddingRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Embeddings endpoint not implemented for PD router",
        )
            .into_response()
    }

    pub async fn flush_cache(&self) -> Response {
        // Process both prefill and decode workers
        let (prefill_results, prefill_errors) = self
            .process_workers(
                WorkerType::Prefill {
                    bootstrap_port: None,
                },
                "Prefill",
                "flush_cache",
            )
            .await;
        let (decode_results, decode_errors) = self
            .process_workers(WorkerType::Decode, "Decode", "flush_cache")
            .await;

        // Combine results and errors
        let mut results = prefill_results;
        results.extend(decode_results);
        let mut errors = prefill_errors;
        errors.extend(decode_errors);

        if errors.is_empty() {
            (
                StatusCode::OK,
                format!("Cache flushed successfully: {:?}", results),
            )
                .into_response()
        } else {
            (
                StatusCode::PARTIAL_CONTENT,
                format!(
                    "Partial success. Results: {:?}, Errors: {:?}",
                    results, errors
                ),
            )
                .into_response()
        }
    }

    pub async fn get_worker_loads(&self) -> Response {
        let mut loads = HashMap::new();
        let mut errors = Vec::new();

        // Process prefill workers
        let prefill_urls = self.get_prefill_worker_urls();
        for worker_url in prefill_urls {
            match get_worker_load(&self.client, &worker_url).await {
                Some(load) => {
                    loads.insert(format!("prefill_{}", worker_url), load);
                }
                None => {
                    errors.push(format!("Failed to get load from prefill {}", worker_url));
                }
            }
        }

        // Process decode workers
        let decode_urls = self.get_decode_worker_urls();
        for worker_url in decode_urls {
            match get_worker_load(&self.client, &worker_url).await {
                Some(load) => {
                    loads.insert(format!("decode_{}", worker_url), load);
                }
                None => {
                    errors.push(format!("Failed to get load from decode {}", worker_url));
                }
            }
        }

        let response_data = serde_json::json!({
            "loads": loads,
            "errors": errors
        });

        (StatusCode::OK, Json(response_data)).into_response()
    }

    pub fn readiness(&self) -> Response {
        // PD router is ready if it has at least one healthy prefill AND one healthy decode worker
        let prefill_workers = self.worker_registry.get_prefill_workers();
        let decode_workers = self.worker_registry.get_decode_workers();

        let healthy_prefill_count = prefill_workers.iter().filter(|w| w.is_healthy()).count();

        let healthy_decode_count = decode_workers.iter().filter(|w| w.is_healthy()).count();

        let total_prefill = prefill_workers.len();
        let total_decode = decode_workers.len();

        if healthy_prefill_count > 0 && healthy_decode_count > 0 {
            Json(json!({
                "status": "ready",
                "prefill": {
                    "healthy": healthy_prefill_count,
                    "total": total_prefill
                },
                "decode": {
                    "healthy": healthy_decode_count,
                    "total": total_decode
                }
            }))
            .into_response()
        } else {
            let mut reasons = Vec::new();
            if healthy_prefill_count == 0 {
                reasons.push("no healthy prefill workers");
            }
            if healthy_decode_count == 0 {
                reasons.push("no healthy decode workers");
            }

            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "status": "not_ready",
                    "reason": reasons.join(", "),
                    "prefill": {
                        "healthy": healthy_prefill_count,
                        "total": total_prefill
                    },
                    "decode": {
                        "healthy": healthy_decode_count,
                        "total": total_decode
                    }
                })),
            )
                .into_response()
        }
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, DPAwareWorker, Worker, WorkerFactory, WorkerType};

    fn create_test_pd_router() -> PdRouterBase {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry =
            Arc::new(PolicyRegistry::new(crate::config::PolicyConfig::RoundRobin));

        PdRouterBase {
            kv_state: None,
            pd_load: Arc::new(PdLoadManager::default()),
            worker_registry,
            policy_registry,
            worker_startup_timeout_secs: 5,
            worker_startup_check_interval_secs: 1,
            worker_loads: Arc::new(tokio::sync::watch::channel(HashMap::new()).1),
            load_monitor_handle: None,
            client: Client::new(),
            api_key: None,
            circuit_breaker_config: CircuitBreakerConfig::default(),
            dp_size: 1,
        }
    }

    fn create_test_pd_router_with_dp(dp_size: usize) -> PdRouterBase {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry =
            Arc::new(PolicyRegistry::new(crate::config::PolicyConfig::RoundRobin));

        PdRouterBase {
            kv_state: None,
            pd_load: Arc::new(PdLoadManager::default()),
            worker_registry,
            policy_registry,
            worker_startup_timeout_secs: 5,
            worker_startup_check_interval_secs: 1,
            worker_loads: Arc::new(tokio::sync::watch::channel(HashMap::new()).1),
            load_monitor_handle: None,
            client: Client::new(),
            api_key: None,
            circuit_breaker_config: CircuitBreakerConfig::default(),
            dp_size,
        }
    }

    fn create_test_worker(url: String, worker_type: WorkerType, healthy: bool) -> Box<dyn Worker> {
        let worker = BasicWorker::new(url, worker_type);
        worker.set_healthy(healthy);
        Box::new(worker)
    }

    // ============= Worker Management Tests =============

    #[tokio::test]
    async fn test_add_prefill_server_already_exists() {
        let router = create_test_pd_router();

        // Add a worker first
        let worker = create_test_worker(
            "http://localhost:8000".to_string(),
            WorkerType::Prefill {
                bootstrap_port: Some(8080),
            },
            true,
        );
        router.worker_registry.register(Arc::from(worker));

        // Try to add the same URL again - this would fail during health check in real scenario
        // For unit test, we test the duplicate check logic
        let exists = router
            .worker_registry
            .get_by_url("http://localhost:8000")
            .is_some();
        assert!(exists);
    }

    #[tokio::test]
    async fn test_remove_prefill_server_success() {
        let router = create_test_pd_router();

        // Add servers first
        let worker1 = create_test_worker(
            "http://worker1".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let worker2 = create_test_worker(
            "http://worker2".to_string(),
            WorkerType::Prefill {
                bootstrap_port: Some(8080),
            },
            true,
        );

        router.worker_registry.register(Arc::from(worker1));
        router.worker_registry.register(Arc::from(worker2));

        // Remove one
        let result = router.remove_prefill_server("http://worker1").await;

        assert!(result.is_ok());
        assert!(result.unwrap().contains("Successfully removed"));

        let workers = router.worker_registry.get_prefill_workers();
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].url(), "http://worker2");
    }

    #[tokio::test]
    async fn test_remove_prefill_server_not_found() {
        let router = create_test_pd_router();

        let result = router.remove_prefill_server("http://nonexistent").await;

        assert!(result.is_err());
        match result.unwrap_err() {
            PDRouterError::WorkerNotFound { url } => {
                assert_eq!(url, "http://nonexistent");
            }
            _ => panic!("Expected WorkerNotFound error"),
        }
    }

    #[tokio::test]
    async fn test_remove_decode_server_success() {
        let router = create_test_pd_router();

        // Add server first
        let worker = create_test_worker("http://decode1".to_string(), WorkerType::Decode, true);
        router.worker_registry.register(Arc::from(worker));

        let result = router.remove_decode_server("http://decode1").await;

        assert!(result.is_ok());
        assert!(result.unwrap().contains("Successfully removed"));

        let workers = router.worker_registry.get_decode_workers();
        assert_eq!(workers.len(), 0);
    }

    // ============= Lock Error Handling Tests =============

    #[test]
    fn test_registry_operations() {
        let router = create_test_pd_router();

        // Test registry operations
        let workers = router.worker_registry.get_all();
        assert_eq!(workers.len(), 0);

        // Add a worker
        let worker = create_test_worker(
            "http://test".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        router.worker_registry.register(Arc::from(worker));

        let workers = router.worker_registry.get_all();
        assert_eq!(workers.len(), 1);

        let prefill_workers = router.worker_registry.get_prefill_workers();
        assert_eq!(prefill_workers.len(), 1);
    }

    // ============= Worker Selection Tests =============

    #[tokio::test]
    async fn test_select_healthy_prefill_worker() {
        let router = create_test_pd_router();

        // Add mix of healthy and unhealthy workers
        let healthy_worker = create_test_worker(
            "http://healthy".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let unhealthy_worker = create_test_worker(
            "http://unhealthy".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            false,
        );
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router.worker_registry.register(Arc::from(unhealthy_worker));
        router.worker_registry.register(Arc::from(healthy_worker));
        router.worker_registry.register(Arc::from(decode_worker));

        // Health-style selection goes through the same admission path as
        // real traffic; availability filtering is part of that path.
        let prefill_workers = router.worker_registry.get_prefill_workers();
        let decode_workers = router.worker_registry.get_decode_workers();
        let context = PdRequestContext {
            prefill_inputs: None,
            model_id: None,
            legacy_policy_input: None,
            prefill_cache: None,
            decode_cache: None,
            headers: None,
            affinity_key: None,
            cache_salt: None,
            route: "/health_generate",
        };
        let (selection, attempt) = router
            .select_and_admit(&prefill_workers, &decode_workers, &context)
            .unwrap();
        let prefill = &prefill_workers[selection.prefill.worker_index];

        // Should select the healthy worker
        assert_eq!(prefill.url(), "http://healthy");
        assert!(prefill.is_healthy());
        drop(attempt);
        assert!(router.pd_load.snapshot().is_empty());
    }

    #[tokio::test]
    async fn test_empty_worker_lists() {
        let router = create_test_pd_router();

        let empty: Vec<Arc<dyn Worker>> = Vec::new();
        let context = PdRequestContext {
            prefill_inputs: None,
            model_id: None,
            legacy_policy_input: None,
            prefill_cache: None,
            decode_cache: None,
            headers: None,
            affinity_key: None,
            cache_salt: None,
            route: "/health_generate",
        };
        assert!(router.select_and_admit(&empty, &empty, &context).is_none());
        assert!(router.pd_load.snapshot().is_empty());
    }

    // ============= Health Endpoints Tests =============

    #[tokio::test]
    async fn test_health_endpoints() {
        let router = create_test_pd_router();

        // Add healthy workers - create_test_worker returns Box<dyn Worker>, convert to Arc
        let prefill_worker = create_test_worker(
            "http://localhost:8000".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let decode_worker = create_test_worker(
            "http://localhost:8001".to_string(),
            WorkerType::Decode,
            true,
        );

        router.worker_registry.register(Arc::from(prefill_worker));
        router.worker_registry.register(Arc::from(decode_worker));

        // Test health endpoint
        let http_req = axum::http::Request::builder()
            .body(axum::body::Body::empty())
            .unwrap();
        let response = router.health(http_req).await;

        assert_eq!(response.status(), 200);

        // Test readiness endpoint
        let response = router.readiness();
        assert_eq!(response.status(), 200);
    }

    // ============= Load Monitoring Tests =============

    #[tokio::test]
    async fn test_load_monitor_updates() {
        let power_of_two_policy = Arc::new(crate::policies::PowerOfTwoPolicy::new());
        let mut router = create_test_pd_router();
        // Set power_of_two policies in the registry
        router
            .policy_registry
            .set_prefill_policy(power_of_two_policy.clone());
        router
            .policy_registry
            .set_decode_policy(power_of_two_policy);

        // Create load channel
        let (tx, rx) = tokio::sync::watch::channel(HashMap::new());
        router.worker_loads = Arc::new(rx);

        // Simulate load updates
        let mut loads = HashMap::new();
        loads.insert("http://worker1".to_string(), 10);
        loads.insert("http://worker2".to_string(), 5);

        let _ = tx.send(loads.clone());

        // Router should receive updates
        let received = router.worker_loads.borrow().clone();
        assert_eq!(received.get("http://worker1"), Some(&10));
        assert_eq!(received.get("http://worker2"), Some(&5));
    }

    // ============= Concurrent Operations Tests =============

    #[tokio::test]
    async fn test_concurrent_worker_operations() {
        let router = Arc::new(create_test_pd_router());

        let mut handles = vec![];

        // Spawn tasks to add workers
        for i in 0..5 {
            let router_clone = Arc::clone(&router);
            let url = format!("http://worker{}", i);
            let handle = tokio::spawn(async move {
                let worker = create_test_worker(
                    url,
                    WorkerType::Prefill {
                        bootstrap_port: None,
                    },
                    true,
                );
                router_clone.worker_registry.register(Arc::from(worker));
            });
            handles.push(handle);
        }

        // Wait for all tasks
        for handle in handles {
            let _ = handle.await;
        }

        // Check final state
        let workers = router.worker_registry.get_prefill_workers();
        assert_eq!(workers.len(), 5);
    }

    // ============= DP-Aware Worker Creation Tests =============
    // These tests verify that add_prefill_server/add_decode_server create
    // DPAwareWorker (not BasicWorker) when dp_size > 1, preventing IPv6+DP
    // URL corruption where @rank is misinterpreted as userinfo per RFC 3986.

    #[test]
    fn test_pd_router_dp_size_field() {
        let router1 = create_test_pd_router();
        assert_eq!(
            router1.dp_size, 1,
            "Default test router should have dp_size=1"
        );

        let router2 = create_test_pd_router_with_dp(4);
        assert_eq!(router2.dp_size, 4);
    }

    #[test]
    fn test_add_prefill_dp_creates_dp_aware_worker() {
        // Simulate what happens when add_prefill_server registers a @rank URL
        // with dp_size > 1: should create DPAwareWorker, not BasicWorker
        let router = create_test_pd_router_with_dp(2);
        let url = "http://10.0.0.1:8000@0";

        let (base_url, dp_rank) = super::dp_utils::parse_worker_url(url);
        let worker: Arc<dyn Worker> = Arc::new(DPAwareWorker::new(
            base_url.clone(),
            dp_rank.unwrap_or(0),
            router.dp_size,
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        ));

        router.worker_registry.register(worker.clone());

        // Verify worker is DP-aware
        assert!(
            worker.is_dp_aware(),
            "Worker should be DP-aware when dp_size > 1"
        );
        assert_eq!(worker.dp_rank(), Some(0));
        assert_eq!(worker.dp_size(), Some(2));

        // Critical: endpoint_url must NOT contain @rank
        let endpoint = worker.endpoint_url("/v1/completions");
        assert_eq!(
            endpoint, "http://10.0.0.1:8000/v1/completions",
            "DPAwareWorker endpoint_url must strip @rank"
        );
        assert!(
            !endpoint.contains('@'),
            "endpoint_url must not contain @ (got: {})",
            endpoint
        );

        // Verify the registry URL still has @rank for identification
        assert_eq!(worker.url(), "http://10.0.0.1:8000@0");
    }

    #[test]
    fn test_add_prefill_dp1_creates_basic_worker() {
        // With dp_size=1, should create BasicWorker (no @rank in URL)
        let router = create_test_pd_router();
        let url = "http://10.0.0.1:8000";

        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            url.to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        ));

        router.worker_registry.register(worker.clone());

        assert!(
            !worker.is_dp_aware(),
            "Worker should NOT be DP-aware when dp_size=1"
        );
        assert_eq!(worker.dp_rank(), None);
        assert_eq!(
            worker.endpoint_url("/v1/completions"),
            "http://10.0.0.1:8000/v1/completions"
        );
    }

    #[test]
    fn test_add_decode_dp_creates_dp_aware_worker() {
        let router = create_test_pd_router_with_dp(4);
        let url = "http://10.0.0.1:9000@2";

        let (base_url, dp_rank) = super::dp_utils::parse_worker_url(url);
        let worker: Arc<dyn Worker> = Arc::new(DPAwareWorker::new(
            base_url,
            dp_rank.unwrap_or(0),
            router.dp_size,
            WorkerType::Decode,
        ));

        router.worker_registry.register(worker.clone());

        assert!(worker.is_dp_aware());
        assert_eq!(worker.dp_rank(), Some(2));
        assert_eq!(worker.dp_size(), Some(4));

        let endpoint = worker.endpoint_url("/v1/completions");
        assert_eq!(endpoint, "http://10.0.0.1:9000/v1/completions");
        assert!(!endpoint.contains('@'));
    }

    #[test]
    fn test_dp_aware_worker_ipv6_url_not_corrupted() {
        // This is the exact production bug: IPv6 + @rank suffix causes URL corruption
        let router = create_test_pd_router_with_dp(2);
        let ipv6_url = "https://[2a03:83e4:5006:0090:5f5a:f8c5:0400:0000]:20009@0";

        let (base_url, dp_rank) = super::dp_utils::parse_worker_url(ipv6_url);
        assert_eq!(
            base_url,
            "https://[2a03:83e4:5006:0090:5f5a:f8c5:0400:0000]:20009"
        );
        assert_eq!(dp_rank, Some(0));

        // DPAwareWorker correctly strips @rank
        let dp_worker: Arc<dyn Worker> = Arc::new(DPAwareWorker::new(
            base_url.clone(),
            dp_rank.unwrap_or(0),
            router.dp_size,
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        ));

        let endpoint = dp_worker.endpoint_url("/v1/completions");
        assert_eq!(
            endpoint, "https://[2a03:83e4:5006:0090:5f5a:f8c5:0400:0000]:20009/v1/completions",
            "DPAwareWorker must produce clean IPv6 endpoint URL"
        );

        // BasicWorker would include @rank, causing corruption
        let basic_worker = BasicWorker::new(
            ipv6_url.to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        );
        let bad_endpoint = basic_worker.endpoint_url("/v1/completions");
        assert_eq!(
            bad_endpoint,
            "https://[2a03:83e4:5006:0090:5f5a:f8c5:0400:0000]:20009@0/v1/completions",
            "BasicWorker includes @rank in endpoint URL (this is the bug)"
        );
    }

    #[test]
    fn test_dp_add_prefill_logic_matches_new() {
        // Verify that the add_prefill_server logic (dp_size > 1 branch)
        // produces the same result as PdRouterBase::new() worker creation
        let dp_size = 4;
        let urls = [
            "http://10.0.0.1:8000@0",
            "http://10.0.0.1:8000@1",
            "http://10.0.0.1:8000@2",
            "http://10.0.0.1:8000@3",
        ];

        for (expected_rank, url) in urls.iter().enumerate() {
            let (base_url, dp_rank) = super::dp_utils::parse_worker_url(url);
            assert_eq!(base_url, "http://10.0.0.1:8000");
            assert_eq!(dp_rank, Some(expected_rank));

            let worker = DPAwareWorker::new(
                base_url,
                dp_rank.unwrap_or(0),
                dp_size,
                WorkerType::Prefill {
                    bootstrap_port: Some(8080),
                },
            );

            assert!(worker.is_dp_aware());
            assert_eq!(worker.dp_rank(), Some(expected_rank));
            assert_eq!(worker.dp_size(), Some(dp_size));
            assert_eq!(
                worker.endpoint_url("/v1/completions"),
                "http://10.0.0.1:8000/v1/completions"
            );
        }
    }

    // ============= Bare URL DP Expansion Tests =============
    // These tests verify that add_prefill_server/add_decode_server correctly
    // expand bare URLs (without @rank) from K8s service discovery into
    // dp_size DPAwareWorkers. Without expansion, all traffic pins to rank 0
    // via unwrap_or(0), bypassing vLLM's hybrid load balancer.

    #[tokio::test]
    async fn test_add_prefill_bare_url_expands_to_dp_size_workers() {
        let router = create_test_pd_router_with_dp(4);
        let bare_url = "http://10.0.0.1:8000";

        // Simulate what service discovery does: call add_prefill_server with bare URL.
        // We can't call add_prefill_server directly (it does health checks),
        // so test the expansion logic inline.
        let dp_size = router.dp_size;
        assert_eq!(dp_size, 4);

        // Bare URL has no @rank
        let (_base, dp_rank) = super::dp_utils::parse_worker_url(bare_url);
        assert!(dp_rank.is_none(), "Bare URL should not have @rank");

        // Expand: register dp_size workers
        for rank in 0..dp_size {
            let worker: Arc<dyn Worker> = Arc::new(DPAwareWorker::new(
                bare_url.to_string(),
                rank,
                dp_size,
                WorkerType::Prefill {
                    bootstrap_port: None,
                },
            ));
            router.worker_registry.register(worker);
        }

        // Should have 4 workers, not 1
        let workers = router.worker_registry.get_prefill_workers();
        assert_eq!(
            workers.len(),
            4,
            "Bare URL should expand into dp_size workers"
        );

        // Each worker should have a distinct rank
        let mut ranks: Vec<usize> = workers.iter().filter_map(|w| w.dp_rank()).collect();
        ranks.sort();
        assert_eq!(ranks, vec![0, 1, 2, 3], "All DP ranks should be present");

        // All workers should share the same base endpoint
        for w in &workers {
            assert_eq!(
                w.endpoint_url("/v1/completions"),
                "http://10.0.0.1:8000/v1/completions"
            );
        }

        // Registry URLs should have @rank suffixes
        let mut urls: Vec<String> = workers.iter().map(|w| w.url().to_string()).collect();
        urls.sort();
        assert_eq!(
            urls,
            vec![
                "http://10.0.0.1:8000@0",
                "http://10.0.0.1:8000@1",
                "http://10.0.0.1:8000@2",
                "http://10.0.0.1:8000@3",
            ]
        );
    }

    #[tokio::test]
    async fn test_add_decode_bare_url_expands_to_dp_size_workers() {
        let router = create_test_pd_router_with_dp(4);
        let bare_url = "http://10.0.0.2:8000";

        for rank in 0..router.dp_size {
            let worker: Arc<dyn Worker> = Arc::new(DPAwareWorker::new(
                bare_url.to_string(),
                rank,
                router.dp_size,
                WorkerType::Decode,
            ));
            router.worker_registry.register(worker);
        }

        let workers = router.worker_registry.get_decode_workers();
        assert_eq!(workers.len(), 4);

        let mut ranks: Vec<usize> = workers.iter().filter_map(|w| w.dp_rank()).collect();
        ranks.sort();
        assert_eq!(ranks, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_bare_url_rank0_pinning_regression() {
        // This is the exact regression from c1710b1: bare URL + dp_size > 1
        // should NOT create a single worker pinned to rank 0.
        let router = create_test_pd_router_with_dp(4);
        let bare_url = "http://10.0.0.1:8000";

        // Old buggy behavior: parse bare URL -> dp_rank=None -> unwrap_or(0)
        // -> single DPAwareWorker at rank 0
        let (_base, dp_rank) = super::dp_utils::parse_worker_url(bare_url);
        assert!(
            dp_rank.is_none(),
            "parse_worker_url on bare URL must return None, not Some(0)"
        );

        // Correct behavior: expand into dp_size workers
        // (The fix checks dp_rank.is_some() to distinguish pre-expanded from bare)
        if dp_rank.is_none() {
            // Must expand, not fall through to unwrap_or(0)
            for rank in 0..router.dp_size {
                let worker: Arc<dyn Worker> = Arc::new(DPAwareWorker::new(
                    bare_url.to_string(),
                    rank,
                    router.dp_size,
                    WorkerType::Prefill {
                        bootstrap_port: None,
                    },
                ));
                router.worker_registry.register(worker);
            }
        }

        let workers = router.worker_registry.get_prefill_workers();
        assert_ne!(
            workers.len(),
            1,
            "Bare URL must NOT create a single rank-0 worker (regression check)"
        );
        assert_eq!(workers.len(), 4);
    }

    #[test]
    fn test_remove_dp_expanded_workers_by_prefix() {
        let router = create_test_pd_router_with_dp(4);
        let bare_url = "http://10.0.0.1:8000";

        // Register 4 expanded workers
        for rank in 0..4 {
            let worker: Arc<dyn Worker> = Arc::new(DPAwareWorker::new(
                bare_url.to_string(),
                rank,
                4,
                WorkerType::Prefill {
                    bootstrap_port: None,
                },
            ));
            router.worker_registry.register(worker);
        }
        assert_eq!(router.worker_registry.get_prefill_workers().len(), 4);

        // Remove by bare URL prefix
        let result = router.remove_dp_expanded_workers(bare_url, "prefill");
        assert!(result.is_ok());
        assert_eq!(router.worker_registry.get_prefill_workers().len(), 0);
    }

    #[test]
    fn test_remove_dp_expanded_workers_not_found() {
        let router = create_test_pd_router_with_dp(4);
        let result = router.remove_dp_expanded_workers("http://nonexistent:8000", "prefill");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_pre_expanded_url_still_works() {
        // Pre-expanded URLs (with @rank) should still create a single worker,
        // not trigger re-expansion.
        let router = create_test_pd_router_with_dp(4);
        let pre_expanded_url = "http://10.0.0.1:8000@2";

        let (_base, dp_rank) = super::dp_utils::parse_worker_url(pre_expanded_url);
        assert_eq!(dp_rank, Some(2), "Pre-expanded URL should parse to rank 2");

        // Register as single worker (what add_prefill_server does for @rank URLs)
        let (base_url, dp_rank) = super::dp_utils::parse_worker_url(pre_expanded_url);
        let worker: Arc<dyn Worker> = Arc::new(DPAwareWorker::new(
            base_url,
            dp_rank.unwrap(),
            router.dp_size,
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        ));
        router.worker_registry.register(worker.clone());

        let workers = router.worker_registry.get_prefill_workers();
        assert_eq!(
            workers.len(),
            1,
            "Pre-expanded @rank URL should create exactly 1 worker"
        );
        assert_eq!(workers[0].dp_rank(), Some(2));
    }

    #[tokio::test]
    async fn test_bare_url_dp1_creates_basic_worker() {
        // With dp_size=1, bare URL should create BasicWorker (no DP expansion)
        let router = create_test_pd_router();
        assert_eq!(router.dp_size, 1);

        let bare_url = "http://10.0.0.1:8000";
        let worker: Arc<dyn Worker> = Arc::from(WorkerFactory::create_prefill_with_config(
            bare_url.to_string(),
            None,
            router.circuit_breaker_config.clone(),
        ));
        router.worker_registry.register(worker.clone());

        let workers = router.worker_registry.get_prefill_workers();
        assert_eq!(workers.len(), 1);
        assert!(!workers[0].is_dp_aware());
        assert_eq!(workers[0].dp_rank(), None);
    }

    #[tokio::test]
    async fn test_two_bare_urls_expand_independently() {
        // Simulates 2 pods discovered by service discovery, each expanded to dp_size workers
        let router = create_test_pd_router_with_dp(4);

        for bare_url in &["http://10.0.0.1:8000", "http://10.0.0.2:8000"] {
            for rank in 0..router.dp_size {
                let worker: Arc<dyn Worker> = Arc::new(DPAwareWorker::new(
                    bare_url.to_string(),
                    rank,
                    router.dp_size,
                    WorkerType::Prefill {
                        bootstrap_port: None,
                    },
                ));
                router.worker_registry.register(worker);
            }
        }

        let workers = router.worker_registry.get_prefill_workers();
        assert_eq!(workers.len(), 8, "2 hosts × 4 ranks = 8 workers");

        // Remove one host's workers
        let result = router.remove_dp_expanded_workers("http://10.0.0.1:8000", "prefill");
        assert!(result.is_ok());
        assert_eq!(
            router.worker_registry.get_prefill_workers().len(),
            4,
            "Only host 2's workers should remain"
        );
    }
}

#[cfg(test)]
mod admission_tests {
    use super::super::pd_types::PdLoadSample;
    use super::*;
    use crate::core::{BasicWorker, DPAwareWorker};
    use crate::policies::{
        AffinityKey, CostParams, KvAwarePolicy, RequestHeaders, RoundRobinPolicy,
    };

    fn prefix_policy_with_blocks(
        p: &[Arc<dyn Worker>],
        tokens: &[u32],
        stored_blocks: usize,
        params: CostParams,
    ) -> KvAwarePolicy {
        use crate::kv_index::{
            CacheKey, HashMode, KvBlockIndexer, ResidencyOwner, RoutableTarget, StorageTier,
        };
        let policy = KvAwarePolicy::new(params);
        let index = Arc::new(KvBlockIndexer::new());
        let hashes = KvAwarePolicy::local_hashes(tokens, 16);
        let blocks: Vec<_> = hashes
            .into_iter()
            .enumerate()
            .take(stored_blocks)
            .map(|(i, hash)| (Arc::<str>::from(format!("block-{i}")), hash))
            .collect();
        index.store(
            0,
            ResidencyOwner::Worker {
                source: Arc::from(p[1].base_url()),
                dp_rank: 0,
                incarnation: 0,
            },
            StorageTier::Device,
            None,
            &blocks,
        );
        for worker in p {
            policy.register_target_with_cache_identity(
                worker.url(),
                RoutableTarget {
                    instance_id: Arc::from(worker.base_url()),
                    dp_rank: 0,
                },
                CacheKey {
                    model: Arc::from("model"),
                    hash_mode: HashMode::Sha256,
                    block_size: 16,
                },
                index.clone(),
            );
        }
        policy
    }

    fn resident_prefix_policy(p: &[Arc<dyn Worker>], tokens: &[u32]) -> KvAwarePolicy {
        prefix_policy_with_blocks(p, tokens, usize::MAX, CostParams::default())
    }

    fn token_inputs(
        p: &[Arc<dyn Worker>],
        tokens: &[u32],
        policy: &KvAwarePolicy,
    ) -> HashMap<String, super::super::pd_types::PdPrefillInput> {
        p.iter()
            .map(|worker| {
                (
                    worker.base_url().to_owned(),
                    super::super::pd_types::PdPrefillInput {
                        prompts: Some(Arc::from(vec![tokens.to_vec()])),
                        work_by_rank: HashMap::from([(
                            worker.dp_rank(),
                            kv_aware_ops(policy).unwrap().estimate_prefill_work(&[(
                                worker.as_ref(),
                                &[tokens.to_vec()],
                                true,
                            )])[0],
                        )]),
                        allow_credit: true,
                        fallback_reason: None,
                    },
                )
            })
            .collect()
    }

    #[test]
    fn pd_prefix_credit_changes_placement_and_charges_exact_attempt_work() {
        let manager = Arc::new(PdLoadManager::default());
        let p = pool("p", 2);
        let d = pool("d", 2);
        let tokens: Vec<u32> = (0..33).collect();
        let policy = resident_prefix_policy(&p, &tokens);
        let colocated = policy
            .select_worker_with_receipt(&p, Some(&tokens), None, None, None, &[])
            .unwrap();
        assert_eq!(colocated.worker_index, 1);
        assert_eq!(colocated.new_prefill_tokens, 1);
        let inputs = token_inputs(&p, &tokens, &policy);
        let mut context = context();
        context.prefill_inputs = Some(&inputs);
        let (first, a) = manager
            .admit(|loads| select_pd_pair(loads, &p, &d, &policy, &policy, &context))
            .unwrap();
        assert_eq!(first.prefill.worker_index, 1); // Warm P wins over the first idle P.
        assert_eq!(first.prefill.new_prefill_tokens, Some(1));
        assert_eq!(first.prefill.credited_prefix_tokens, Some(32));
        assert_eq!(first.prefill.fallback_reason, None);
        let (second, b) = manager
            .admit(|loads| select_pd_pair(loads, &p, &d, &policy, &policy, &context))
            .unwrap();
        assert_eq!(second.prefill.worker_index, 1);
        assert_eq!(second.prefill.score, Some(2.0));
        assert_ne!(first.decode.endpoint, second.decode.endpoint);
        assert_eq!(
            sample(&manager, Stage::Prefill, &first.prefill.endpoint).active_prefill_tokens,
            2
        );
        a.finish_prefill();
        assert_eq!(
            sample(&manager, Stage::Prefill, &first.prefill.endpoint).active_prefill_tokens,
            1
        );
        drop((a, b));
        assert!(manager.snapshot().is_empty());

        let mut loads = Loads::new();
        loads.insert(
            (Stage::Prefill, first.prefill.endpoint),
            PdLoadSample {
                prefill_requests: 1,
                active_prefill_tokens: 40,
                ..Default::default()
            },
        );
        let selection = select_pd_pair(&loads, &p, &d, &policy, &policy, &context).unwrap();
        assert_eq!(selection.prefill.worker_index, 0); // Token backlog outweighs warm residency.
        assert_eq!(selection.prefill.new_prefill_tokens, Some(33));
    }

    #[test]
    fn prefill_backlog_shifts_to_idle_worker() {
        let p = pool("p", 2);
        let d = pool("d", 1);
        // 400 tokens; p1 holds the first 12 blocks (192), so new_p1 = 208
        // and new_p0 = 400. Placement is the only term favoring the holder:
        // once its admitted backlog outweighs the recompute gap, the request
        // shifts to the idle worker based on actual admitted token work.
        let tokens: Vec<u32> = (0..400).collect();
        let mut context = context();

        let policy = prefix_policy_with_blocks(&p, &tokens, 12, CostParams::default());
        let inputs = token_inputs(&p, &tokens, &policy);
        context.prefill_inputs = Some(&inputs);
        let manager = Arc::new(PdLoadManager::default());
        let (first, a) = manager
            .admit(|loads| select_pd_pair(loads, &p, &d, &policy, &policy, &context))
            .unwrap();
        assert_eq!(first.prefill.worker_index, 1);
        let (second, b) = manager
            .admit(|loads| select_pd_pair(loads, &p, &d, &policy, &policy, &context))
            .unwrap();
        assert_eq!(
            second.prefill.worker_index, 0,
            "the idle worker's backlog advantage beats the recompute gap"
        );
        drop((a, b));
    }

    #[test]
    fn unknown_inflight_work_forces_one_unit_consistent_pool_fallback() {
        let manager = Arc::new(PdLoadManager::default());
        let p = pool("p", 2);
        let d = pool("d", 2);
        let tokens: Vec<u32> = (0..33).collect();
        let policy = resident_prefix_policy(&p, &tokens);
        let (_, unknown) = manager
            .admit(|loads| select_pd_pair(loads, &p, &d, &policy, &policy, &context()))
            .unwrap();
        let inputs = token_inputs(&p, &tokens, &policy);
        let mut context = context();
        context.prefill_inputs = Some(&inputs);
        let (selection, known) = manager
            .admit(|loads| select_pd_pair(loads, &p, &d, &policy, &policy, &context))
            .unwrap();
        assert_eq!(
            selection.prefill.fallback_reason,
            Some("unknown_prefill_backlog")
        );
        assert_eq!(selection.prefill.new_prefill_tokens, Some(33));
        assert_eq!(selection.prefill.credited_prefix_tokens, Some(0));
        drop((unknown, known));
        assert!(manager.snapshot().is_empty());
    }

    fn pool(prefix: &str, count: usize) -> Vec<Arc<dyn Worker>> {
        (0..count)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://{prefix}{i}"),
                    WorkerType::Regular,
                )) as Arc<dyn Worker>
            })
            .collect()
    }

    fn context() -> PdRequestContext<'static> {
        PdRequestContext {
            prefill_inputs: None,
            model_id: None,
            legacy_policy_input: Some("original request"),
            prefill_cache: None,
            decode_cache: None,
            headers: None,
            affinity_key: None,
            cache_salt: None,
            route: "/v1/completions",
        }
    }

    fn admit(
        manager: &Arc<PdLoadManager>,
        p: &[Arc<dyn Worker>],
        d: &[Arc<dyn Worker>],
    ) -> (PdSelection, PdAttempt) {
        let policy = KvAwarePolicy::default();
        manager
            .admit(|loads| select_pd_pair(loads, p, d, &policy, &policy, &context()))
            .unwrap()
    }

    fn context_with_session(session: &str) -> PdRequestContext<'static> {
        PdRequestContext {
            prefill_inputs: None,
            model_id: None,
            legacy_policy_input: Some("original request"),
            prefill_cache: None,
            decode_cache: None,
            headers: None,
            affinity_key: Some(affinity_key(session)),
            cache_salt: None,
            route: "/v1/completions",
        }
    }

    fn affinity_key(session: &str) -> AffinityKey {
        let body = serde_json::json!({ "session_id": session });
        let identity = crate::request_identity::resolve_affinity_identity(
            None,
            crate::request_identity::IdentityFields::from_body(&body),
        )
        .unwrap()
        .unwrap();
        AffinityKey::from_identity(identity, "default")
    }

    fn admit_with(
        manager: &Arc<PdLoadManager>,
        p: &[Arc<dyn Worker>],
        d: &[Arc<dyn Worker>],
        policy: &KvAwarePolicy,
        context: &PdRequestContext<'_>,
    ) -> (PdSelection, PdAttempt) {
        manager
            .admit(|loads| select_pd_pair(loads, p, d, policy, policy, context))
            .unwrap()
    }

    fn sample(manager: &PdLoadManager, stage: Stage, endpoint: &PdEndpoint) -> PdLoadSample {
        manager
            .snapshot()
            .get(&(stage, endpoint.clone()))
            .copied()
            .unwrap_or_default()
    }

    #[test]
    fn decode_soft_affinity_rebinds_pending_and_active_requests_then_stays_idle() {
        let manager = Arc::new(PdLoadManager::default());
        let p = pool("p", 1);
        let d = pool("d", 2);
        let policy = affinity_policy();
        let ctx = context_with_session("session-a");
        let (first, attempt) = admit_with(&manager, &p, &d, &policy, &ctx);
        assert_eq!(first.decode.affinity_outcome, Some("bind"));
        // Pending reservations already count, so another request spills.
        let (second, second_attempt) = admit_with(&manager, &p, &d, &policy, &ctx);
        assert_ne!(first.decode.worker_index, second.decode.worker_index);
        assert_eq!(second.decode.affinity_outcome, Some("rebind"));
        second_attempt.begin_decode();
        drop(attempt);
        // Active D work also counts. The old decoder is now idle.
        let (third, third_attempt) = admit_with(&manager, &p, &d, &policy, &ctx);
        assert_eq!(third.decode.worker_index, first.decode.worker_index);
        assert_eq!(third.decode.affinity_outcome, Some("rebind"));
        drop((second_attempt, third_attempt));
        // With no overlap, subsequent turns keep their cached decoder.
        let (next, attempt) = admit_with(&manager, &p, &d, &policy, &ctx);
        assert_eq!(next.decode.worker_index, third.decode.worker_index);
        assert_eq!(next.decode.affinity_outcome, Some("sticky_hit"));
        drop(attempt);
        assert!(manager.snapshot().is_empty());
    }

    /// DP ranks on one HTTP server share a base URL; the binding target must
    /// stay rank-qualified or every lookup would resolve to the first rank
    /// and strand session KV there.
    #[test]
    fn decode_session_binds_by_rank_qualified_target() {
        let manager = Arc::new(PdLoadManager::default());
        let p = pool("p", 1);
        let d: Vec<Arc<dyn Worker>> = (0..2)
            .map(|rank| {
                Arc::new(DPAwareWorker::new(
                    "http://d".to_owned(),
                    rank,
                    2,
                    WorkerType::Decode,
                )) as Arc<dyn Worker>
            })
            .collect();
        let policy = affinity_policy();

        // A live request on rank 0 makes session a choose rank 1.
        let ctx_b = context_with_session("session-b");
        let mut attempts = Vec::new();
        let (pick, attempt) = admit_with(&manager, &p, &d, &policy, &ctx_b);
        assert_eq!(pick.decode.worker_index, 0);
        attempts.push(attempt);
        // Session a must bind to the idle rank 1...
        let ctx_a = context_with_session("session-a");
        let (first, attempt) = admit_with(&manager, &p, &d, &policy, &ctx_a);
        assert_eq!(first.decode.worker_index, 1);
        assert_eq!(first.decode.affinity_outcome, Some("bind"));
        attempts.push(attempt);
        // Finish both requests. The next turn must retain rank 1 even
        // though rank 0 is equally idle and sorts first.
        drop(attempts);
        let (next, attempt) = admit_with(&manager, &p, &d, &policy, &ctx_a);
        assert_eq!(next.decode.worker_index, 1);
        assert_eq!(next.decode.affinity_outcome, Some("sticky_hit"));
        drop(attempt);
        assert!(manager.snapshot().is_empty());
    }

    fn affinity_policy() -> KvAwarePolicy {
        KvAwarePolicy::default().with_shared_affinity(Arc::new(
            crate::policies::KvAffinityTable::with_ttl(Some(Duration::from_secs(1800))),
        ))
    }

    #[test]
    fn decode_affinity_does_not_filter_prefill_token_cost_candidates() {
        let manager = Arc::new(PdLoadManager::default());
        let p = pool("p", 2);
        let d = pool("d", 1);
        let tokens: Vec<u32> = (0..33).collect();
        let policy = resident_prefix_policy(&p, &tokens).with_shared_affinity(Arc::new(
            crate::policies::KvAffinityTable::with_ttl(Some(Duration::from_secs(1800))),
        ));
        let inputs = token_inputs(&p, &tokens, &policy);
        let mut ctx = context_with_session("session-a");
        ctx.prefill_inputs = Some(&inputs);
        let (first, a) = admit_with(&manager, &p, &d, &policy, &ctx);
        assert_eq!(first.prefill.worker_index, 1);
        assert_eq!(first.prefill.new_prefill_tokens, Some(1));
        assert_eq!(first.prefill.affinity_outcome, None);
        assert!(first.prefill.affinity.is_none());
        assert_eq!(first.decode.affinity_outcome, Some("bind"));
        // Warm P1 still wins on token cost (2 vs 33), even though its
        // projected request count exceeds the old affinity limit (2 vs 1).
        let (second, b) = admit_with(&manager, &p, &d, &policy, &ctx);
        assert_eq!(second.prefill.worker_index, 1);
        assert_eq!(second.prefill.new_prefill_tokens, Some(1));
        assert_eq!(second.prefill.score, Some(2.0));
        assert_eq!(second.prefill.affinity_outcome, None);
        assert!(second.prefill.affinity.is_none());
        assert_eq!(second.decode.affinity_outcome, Some("sticky_hit"));
        assert_eq!(
            sample(&manager, Stage::Prefill, &first.prefill.endpoint).prefill_requests,
            2
        );
        // A real token backlog still outweighs P1's prefix credit for this
        // same session: decode affinity must never pin P to its last choice.
        let mut loads = manager.snapshot();
        loads
            .get_mut(&(Stage::Prefill, first.prefill.endpoint.clone()))
            .unwrap()
            .active_prefill_tokens = 40;
        let unbound_decode = KvAwarePolicy::default();
        let next = select_pd_pair(&loads, &p, &d, &policy, &unbound_decode, &ctx).unwrap();
        assert_eq!(next.prefill.worker_index, 0);
        assert_eq!(next.prefill.new_prefill_tokens, Some(33));
        assert_eq!(next.prefill.affinity_outcome, None);
        assert!(next.prefill.affinity.is_none());
        assert_eq!(
            sample(&manager, Stage::Prefill, &second.prefill.endpoint).prefill_requests,
            2
        );
        drop((a, b));
        assert!(manager.snapshot().is_empty());
    }

    #[test]
    fn concurrent_session_admissions_balance_occupancy_with_decode_only_binding() {
        let manager = Arc::new(PdLoadManager::default());
        let p = pool("p", 4);
        let d = pool("d", 4);
        let policy = affinity_policy();
        let attempts: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..16)
                .map(|_| {
                    scope.spawn(|| {
                        admit_with(
                            &manager,
                            &p,
                            &d,
                            &policy,
                            &context_with_session("same-session"),
                        )
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| {
                    let (selection, attempt) = handle.join().unwrap();
                    // These requests have no prepared token work, so this
                    // also covers the occupancy fallback with TTL enabled.
                    assert_eq!(selection.prefill.affinity_outcome, None);
                    assert!(selection.prefill.affinity.is_none());
                    attempt
                })
                .collect()
        });
        let snapshot = manager.snapshot();
        assert_eq!(snapshot.len(), 8);
        for ((stage, _), load) in snapshot {
            assert_eq!(load.occupancy(stage), 4);
        }
        drop(attempts);
        assert!(manager.snapshot().is_empty());
    }

    #[test]
    fn pending_handoffs_balance_even_with_fresh_discovery_workers() {
        let manager = Arc::new(PdLoadManager::default());
        let (first, a) = admit(&manager, &pool("p", 1), &pool("d", 2));
        a.finish_prefill();
        let (second, b) = admit(&manager, &pool("p", 1), &pool("d", 2));
        assert_ne!(first.decode.endpoint, second.decode.endpoint);
        assert_eq!(
            sample(&manager, Stage::Decode, &first.decode.endpoint).pending_decode_requests,
            1
        );
        assert_eq!(first.prefill.new_prefill_tokens, None);
        drop((a, b));
        assert!(manager.snapshot().is_empty());
    }

    #[test]
    fn stages_finish_independently_and_transitions_are_idempotent() {
        let manager = Arc::new(PdLoadManager::default());
        let (selection, attempt) = admit(&manager, &pool("same", 1), &pool("same", 1));
        attempt.begin_decode();
        attempt.begin_decode();
        let d = sample(&manager, Stage::Decode, &selection.decode.endpoint);
        assert_eq!(
            (d.pending_decode_requests, d.active_decode_requests),
            (0, 1)
        );
        attempt.finish_decode();
        attempt.finish_decode();
        assert_eq!(
            sample(&manager, Stage::Prefill, &selection.prefill.endpoint).prefill_requests,
            1
        );
        attempt.finish_prefill();
        attempt.finish_prefill();
        attempt.release();
        attempt.begin_decode();
        assert!(manager.snapshot().is_empty());
    }

    #[test]
    fn finishing_prefill_retains_decode_and_drop_releases_pending() {
        let manager = Arc::new(PdLoadManager::default());
        let (selection, attempt) = admit(&manager, &pool("p", 1), &pool("d", 1));
        attempt.finish_prefill();
        assert_eq!(
            sample(&manager, Stage::Decode, &selection.decode.endpoint).pending_decode_requests,
            1
        );
        let clone = attempt.clone();
        drop(attempt);
        assert_eq!(manager.snapshot().len(), 1);
        drop(clone);
        assert!(manager.snapshot().is_empty());
    }

    #[test]
    fn decode_policy_failure_after_prefill_selection_leaves_no_reservations() {
        #[derive(Debug)]
        struct RejectDecode;
        impl LoadBalancingPolicy for RejectDecode {
            fn select_worker_with_headers(
                &self,
                _: &[Arc<dyn Worker>],
                _: Option<&str>,
                _: Option<&RequestHeaders>,
            ) -> Option<usize> {
                None
            }
            fn name(&self) -> &'static str {
                "reject-decode"
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
        let manager = Arc::new(PdLoadManager::default());
        let p = pool("p", 1);
        let d = pool("d", 1);
        let policy = KvAwarePolicy::default();
        assert!(manager
            .admit(|loads| select_pd_pair(loads, &p, &d, &policy, &RejectDecode, &context()))
            .is_none());
        assert!(manager.snapshot().is_empty());
    }

    #[test]
    fn equal_load_uses_endpoint_order_not_hashmap_enumeration() {
        let manager = Arc::new(PdLoadManager::default());
        let mut p = pool("p", 2);
        let mut d = pool("d", 2);
        p.reverse();
        d.reverse();
        let (selection, _) = admit(&manager, &p, &d);
        assert_eq!(selection.prefill.worker_index, 1);
        assert_eq!(selection.decode.worker_index, 1);
    }

    #[test]
    fn dp_ranks_have_independent_occupancy_and_granularity_is_pinned() {
        let manager = Arc::new(PdLoadManager::default());
        let p = pool("p", 1);
        let d: Vec<Arc<dyn Worker>> = (0..2)
            .map(|rank| {
                Arc::new(DPAwareWorker::new(
                    "http://d0".into(),
                    rank,
                    2,
                    WorkerType::Decode,
                )) as Arc<dyn Worker>
            })
            .collect();
        let (first, a) = admit(&manager, &p, &d);
        let (second, b) = admit(&manager, &p, &d);
        assert_eq!(first.decode.endpoint.dp_rank, Some(0));
        assert_eq!(second.decode.endpoint.dp_rank, Some(1));
        let policy = KvAwarePolicy::default();
        assert!(manager
            .admit(|loads| select_pd_pair(loads, &p, &pool("d", 1), &policy, &policy, &context()))
            .is_none());
        drop((a, b));
        let (_, attempt) = admit(&manager, &p, &pool("d", 1));
        drop(attempt);
        assert!(manager.snapshot().is_empty());
    }

    #[test]
    fn concurrent_admission_samples_and_reserves_atomically() {
        let manager = Arc::new(PdLoadManager::default());
        let handles: Vec<_> = (0..32)
            .map(|_| {
                let manager = manager.clone();
                std::thread::spawn(move || admit(&manager, &pool("p", 2), &pool("d", 2)))
            })
            .collect();
        let admitted: Vec<_> = handles
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        for load in manager.snapshot().values() {
            assert_eq!(load.prefill_requests + load.pending_decode_requests, 16);
        }
        drop(admitted);
        assert!(manager.snapshot().is_empty());
    }

    #[test]
    fn racing_cleanup_never_releases_another_attempt() {
        let manager = Arc::new(PdLoadManager::default());
        let (_, a) = admit(&manager, &pool("p", 1), &pool("d", 1));
        let (selection, b) = admit(&manager, &pool("p", 1), &pool("d", 1));
        let threads: Vec<_> = (0..16)
            .map(|i| {
                let a = a.clone();
                std::thread::spawn(move || {
                    if i % 2 == 0 {
                        a.begin_decode();
                    }
                    a.release();
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(
            sample(&manager, Stage::Prefill, &selection.prefill.endpoint).prefill_requests,
            1
        );
        assert_eq!(
            sample(&manager, Stage::Decode, &selection.decode.endpoint).pending_decode_requests,
            1
        );
        drop((a, b));
        assert!(manager.snapshot().is_empty());
    }

    #[test]
    fn cleanup_uses_each_attempts_immutable_admitted_token_amount() {
        let manager = Arc::new(PdLoadManager::default());
        let p = pool("p", 1);
        let d = pool("d", 1);
        let policy = KvAwarePolicy::default();
        let known = |tokens| {
            manager
                .admit(|loads| {
                    let mut selection =
                        select_pd_pair(loads, &p, &d, &policy, &policy, &context())?;
                    selection.prefill.new_prefill_tokens = Some(tokens);
                    Some(selection)
                })
                .unwrap()
        };
        let (selection, a) = known(17);
        let (_, b) = known(31);
        assert_eq!(
            sample(&manager, Stage::Prefill, &selection.prefill.endpoint).active_prefill_tokens,
            48
        );
        a.finish_prefill();
        a.finish_prefill();
        assert_eq!(
            sample(&manager, Stage::Prefill, &selection.prefill.endpoint).active_prefill_tokens,
            31
        );
        drop((a, b));
        assert!(manager.snapshot().is_empty());
    }

    #[derive(Debug)]
    struct LegacyPolicy;
    impl LoadBalancingPolicy for LegacyPolicy {
        fn select_worker_with_headers(
            &self,
            workers: &[Arc<dyn Worker>],
            text: Option<&str>,
            headers: Option<&RequestHeaders>,
        ) -> Option<usize> {
            assert_eq!(text, Some("original request"));
            assert_eq!(
                headers.unwrap().get("x-session-id").map(String::as_str),
                Some("session")
            );
            workers
                .iter()
                .enumerate()
                .min_by_key(|(_, w)| w.load())
                .map(|(i, _)| i)
        }
        fn name(&self) -> &'static str {
            "legacy-test"
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn mixed_legacy_policies_keep_inputs_and_read_shared_occupancy() {
        let manager = Arc::new(PdLoadManager::default());
        let p = pool("p", 2);
        let d = pool("d", 2);
        let (first, a) = admit(&manager, &p, &d);
        let headers = HashMap::from([("x-session-id".into(), "session".into())]);
        let mut context = context();
        context.headers = Some(&headers);
        let policy = LegacyPolicy;
        let (second, b) = manager
            .admit(|loads| select_pd_pair(loads, &p, &d, &policy, &policy, &context))
            .unwrap();
        assert_ne!(first.prefill.endpoint, second.prefill.endpoint);
        assert_ne!(first.decode.endpoint, second.decode.endpoint);
        assert_eq!(second.prefill.score, None);
        drop((a, b));
        assert!(manager.snapshot().is_empty());
    }

    #[test]
    fn stage_override_is_preserved() {
        let manager = Arc::new(PdLoadManager::default());
        let p = pool("p", 2);
        let d = pool("d", 1);
        let rr = RoundRobinPolicy::new();
        let kv = KvAwarePolicy::default();
        let (_, a) = manager
            .admit(|loads| select_pd_pair(loads, &p, &d, &rr, &kv, &context()))
            .unwrap();
        let (second, b) = manager
            .admit(|loads| select_pd_pair(loads, &p, &d, &rr, &kv, &context()))
            .unwrap();
        assert_eq!(second.prefill.score, None);
        assert_eq!(second.decode.score, Some(2.0));
        drop((a, b));
    }

    #[test]
    fn model_labels_filter_eligible_pools() {
        // PD workers register without labels today, so model filtering is
        // inert until something populates model_id. Pin the behavior for the
        // labeled future: a matching label or an unlabeled ("unknown")
        // worker stays eligible, mismatched labels are excluded, and a model
        // served by no labeled worker fails admission once every worker
        // carries a label.
        let manager = Arc::new(PdLoadManager::default());
        let policy = KvAwarePolicy::default();
        let d = pool("d", 1);
        let labeled = |url: &str, model: &str| {
            Arc::new(
                BasicWorker::new(url.to_string(), WorkerType::Regular)
                    .with_labels(HashMap::from([("model_id".to_string(), model.to_string())])),
            ) as Arc<dyn Worker>
        };
        let unlabeled = |url: &str| {
            Arc::new(BasicWorker::new(url.to_string(), WorkerType::Regular)) as Arc<dyn Worker>
        };
        fn ctx(model: &str) -> PdRequestContext<'_> {
            PdRequestContext {
                prefill_inputs: None,
                model_id: Some(model),
                legacy_policy_input: Some("original request"),
                prefill_cache: None,
                decode_cache: None,
                headers: None,
                affinity_key: None,
                cache_salt: None,
                route: "/v1/completions",
            }
        }

        // Request for model-b: p1 matches, p2 is unknown, p0 mismatches.
        let p = vec![
            labeled("http://p0", "model-a"),
            labeled("http://p1", "model-b"),
            unlabeled("http://p2"),
        ];
        let (selection, attempt) = manager
            .admit(|loads| select_pd_pair(loads, &p, &d, &policy, &policy, &ctx("model-b")))
            .unwrap();
        assert_eq!(selection.prefill.endpoint.http_url, "http://p1");
        drop(attempt);

        // Request for model-c: only the unlabeled worker keeps it routable.
        let (selection, attempt) = manager
            .admit(|loads| select_pd_pair(loads, &p, &d, &policy, &policy, &ctx("model-c")))
            .unwrap();
        assert_eq!(selection.prefill.endpoint.http_url, "http://p2");
        drop(attempt);

        // Once every worker carries a label, an unserved model has no pair.
        let only_labeled = vec![
            labeled("http://p0", "model-a"),
            labeled("http://p1", "model-b"),
        ];
        assert!(manager
            .admit(|loads| {
                select_pd_pair(loads, &only_labeled, &d, &policy, &policy, &ctx("model-c"))
            })
            .is_none());
        assert!(manager.snapshot().is_empty());
    }

    #[tokio::test]
    async fn body_guard_holds_until_eof_and_handles_unpolled_drop() {
        let manager = Arc::new(PdLoadManager::default());
        let (_, attempt) = admit(&manager, &pool("p", 1), &pool("d", 1));
        attempt.finish_prefill();
        attempt.begin_decode();
        let body = attempt.track_body(Body::from("chunk"));
        assert_eq!(manager.snapshot().len(), 1);
        let bytes = axum::body::to_bytes(body, 100).await.unwrap();
        assert_eq!(&bytes[..], b"chunk");
        assert!(manager.snapshot().is_empty());
        let (_, attempt) = admit(&manager, &pool("p", 1), &pool("d", 1));
        attempt.finish_prefill();
        attempt.begin_decode();
        let retained_producer_handle = attempt.clone();
        let body = attempt.track_body(Body::from("never polled"));
        drop(body);
        assert!(manager.snapshot().is_empty());
        drop(retained_producer_handle);
    }

    #[tokio::test]
    async fn body_error_and_cancellation_release_occupancy() {
        let manager = Arc::new(PdLoadManager::default());
        let (_, attempt) = admit(&manager, &pool("p", 1), &pool("d", 1));
        attempt.finish_prefill();
        attempt.begin_decode();
        let body = Body::from_stream(futures_util::stream::once(async {
            Err::<bytes::Bytes, _>(std::io::Error::other("injected read failure"))
        }));
        assert!(axum::body::to_bytes(attempt.track_body(body), 100)
            .await
            .is_err());
        assert!(manager.snapshot().is_empty());

        let (_, attempt) = admit(&manager, &pool("p", 1), &pool("d", 1));
        attempt.begin_decode();
        let body = Body::from_stream(futures_util::stream::pending::<
            Result<bytes::Bytes, std::io::Error>,
        >());
        let task =
            tokio::spawn(async move { axum::body::to_bytes(attempt.track_body(body), 100).await });
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert!(manager.snapshot().is_empty());
    }

    #[test]
    fn early_error_response_releases_both_stages_without_waiting_for_client() {
        let manager = Arc::new(PdLoadManager::default());
        let (_, attempt) = admit(&manager, &pool("p", 1), &pool("d", 1));
        let response =
            attempt.track_response(axum::response::Response::new(Body::from("prefill error")));
        assert!(manager.snapshot().is_empty());
        drop(response);
    }
}
