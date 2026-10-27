use super::completion_route::{placement_eligibility, prepare_completion_route, CompletionRoute};
use super::program_adapter::ProgramCompletion;
use crate::backend::preprocess::prepare_http_kv_inputs;
use crate::backend::{
    stream_output::{output_frame_tracking, OutputFrameConsumer, OutputFrameProducer},
    PrepareError,
};
use crate::config::types::RetryConfig;
use crate::core::{
    is_retryable_status, BasicWorker, CircuitBreakerConfig, DPAwareWorker, HealthConfig,
    RetryExecutor, Worker, WorkerRegistry, WorkerType,
};
use crate::metrics::RouterMetrics;
use crate::otel_http::{self, ClientRequestOptions};
use crate::policies::KvRequestTokens;
use crate::policies::{
    kv_aware_ops, kv_aware_policy, AffinityKey, KvRequestSlot, KvSelectionReceipt,
    LoadBalancingPolicy, PolicyRegistry, QualifiedKvInput,
};
use crate::program_scheduling::{
    BackendObservationProvider, ProgramIdentity, ProgramScheduler, ProgramSchedulerConfig,
    ProgramTarget, ScheduleError, VllmMetricsObservationProvider,
};
use crate::protocols::spec::{
    ChatCompletionRequest, CompletionRequest, EmbeddingRequest, GenerateRequest, GenerationRequest,
    GrpcRequestRef, InferenceGenerateRequest, RerankRequest, RerankResponse, RerankResult,
    ResponsesRequest,
};
use crate::request_identity::{resolve_affinity_identity, IdentityFields};
use crate::routers::header_utils;
use crate::routers::http::dp_utils;
use crate::routers::{RouterTrait, WorkerManagement};
use crate::token_estimator::{MomentumTokenEstimator, TokenEstimateScope};
use axum::body::to_bytes;
use axum::{
    body::Body,
    extract::Request,
    http::{
        header::CONTENT_LENGTH, header::CONTENT_TYPE, HeaderMap, HeaderValue, Method, StatusCode,
    },
    response::{IntoResponse, Response},
    Json,
};
use futures_util::StreamExt;
use parking_lot::Mutex;
use reqwest::Client;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, info, warn};

fn prepare_error_status(error: &PrepareError) -> StatusCode {
    match error {
        PrepareError::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
        PrepareError::Invalid(_) => StatusCode::BAD_REQUEST,
    }
}

fn insert_router_stages(headers: &mut HeaderMap, stages: &serde_json::Value) {
    if let Ok(value) = HeaderValue::from_str(&stages.to_string()) {
        headers.insert("x-router-stages", value);
    }
}

// Diagnostic only. For HTTP this measures router -> worker header wait and
// first SSE byte; `engine_ms` is a residual, not EngineCore telemetry.
fn http_stages_json(http_ttfb_ms: f64, first_sse_ms: f64) -> serde_json::Value {
    serde_json::json!({
        "path": "http_proxy",
        "http_ttfb_ms": http_ttfb_ms,
        "http_first_sse_ms": first_sse_ms,
        "xfer_ms": http_ttfb_ms,
        "first_token_ms": first_sse_ms,
        "engine_ms": first_sse_ms - http_ttfb_ms,
    })
}

fn emit_http_first_sse<E>(
    tx: &tokio::sync::mpsc::UnboundedSender<Result<bytes::Bytes, E>>,
    stages: &serde_json::Value,
    output_frame: Option<&mut OutputFrameProducer>,
) -> bool {
    let comment = bytes::Bytes::from(format!(": router-stages {stages}\n\n"));
    info!(%stages, "http proxy stages");
    match output_frame {
        Some(output_frame) => output_frame.send(tx, comment, false),
        None => tx.send(Ok(comment)).is_ok(),
    }
}

struct LoadTrackedBody {
    inner: Pin<
        Box<dyn futures_util::Stream<Item = Result<bytes::Bytes, axum::Error>> + Send + 'static>,
    >,
    load: RequestLoadGuard,
    /// gRPC and Completion streams report the frame containing model output.
    /// Other HTTP streams retain their first-frame behavior.
    output_frame: Option<OutputFrameConsumer>,
    producer_abort: Option<tokio::task::AbortHandle>,
}

struct RequestLoadState {
    worker: Option<Arc<dyn Worker>>,
    kv_slot: Option<KvRequestSlot>,
    released: AtomicBool,
}

/// Owns an admitted attempt across dispatch and response consumption. Any
/// cancellation or completion releases the shared counters exactly once.
#[derive(Clone)]
struct RequestLoadGuard {
    state: Arc<RequestLoadState>,
}

impl RequestLoadGuard {
    fn new(worker: Option<Arc<dyn Worker>>, kv_slot: Option<KvRequestSlot>) -> Self {
        Self {
            state: Arc::new(RequestLoadState {
                worker,
                kv_slot,
                released: AtomicBool::new(false),
            }),
        }
    }

    fn from_selection(
        worker: Arc<dyn Worker>,
        receipt: Option<KvSelectionReceipt>,
        tracked: bool,
    ) -> Self {
        let tracked = tracked || receipt.is_some();
        if tracked {
            if receipt.is_none() {
                worker.increment_load();
            }
            RouterMetrics::set_running_requests(worker.url(), worker.load());
        }
        Self::new(
            tracked.then_some(worker),
            receipt.and_then(|receipt| receipt.slot),
        )
    }

    fn tracks_worker(&self) -> bool {
        self.state.worker.is_some()
    }

    fn begin_decode(&self) {
        if let Some(slot) = &self.state.kv_slot {
            slot.begin_decode();
        }
    }

    fn release(&self) {
        if self.state.released.swap(true, Ordering::AcqRel) {
            return;
        }
        if let Some(slot) = &self.state.kv_slot {
            slot.release();
        }
        if let Some(worker) = &self.state.worker {
            worker.decrement_load();
            RouterMetrics::set_running_requests(worker.url(), worker.load());
        }
    }
}

impl Drop for RequestLoadGuard {
    fn drop(&mut self) {
        self.release();
    }
}

impl LoadTrackedBody {
    fn release(&mut self) {
        self.load.release();
    }
}

impl futures_util::Stream for LoadTrackedBody {
    type Item = Result<bytes::Bytes, axum::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let item = self.inner.as_mut().poll_next(cx);
        if matches!(&item, Poll::Ready(Some(Ok(_)))) {
            let output_started = self
                .output_frame
                .as_mut()
                .is_none_or(OutputFrameConsumer::on_frame);
            if output_started {
                self.load.begin_decode();
            }
        }
        if matches!(item, Poll::Ready(None)) {
            self.release();
            self.producer_abort.take();
        }
        item
    }
}

impl Drop for LoadTrackedBody {
    fn drop(&mut self) {
        if let Some(abort) = self.producer_abort.take() {
            abort.abort();
        }
        self.release();
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
fn hold_load_until_body_done(
    response: Response,
    worker: Arc<dyn Worker>,
    kv_slot: Option<KvRequestSlot>,
) -> Response {
    hold_request_load_until_body_done(response, RequestLoadGuard::new(Some(worker), kv_slot))
}

fn hold_request_load_until_body_done(mut response: Response, load: RequestLoadGuard) -> Response {
    let task = response
        .extensions_mut()
        .remove::<crate::backend::grpc::GrpcStreamTask>();
    let output_frame = task.as_ref().and_then(|task| task.output_frame());
    let producer = task.and_then(|task| task.take());
    hold_request_load_with_output_frame(response, load, producer, output_frame)
}

fn hold_request_load_with_output_frame(
    response: Response,
    load: RequestLoadGuard,
    producer: Option<tokio::task::JoinHandle<()>>,
    output_frame: Option<OutputFrameConsumer>,
) -> Response {
    let producer_abort = producer.as_ref().map(tokio::task::JoinHandle::abort_handle);
    if let Some(producer) = producer {
        let producer_load = load.clone();
        tokio::spawn(async move {
            let _ = producer.await;
            producer_load.release();
        });
    }
    let (parts, body) = response.into_parts();
    let stream = LoadTrackedBody {
        inner: Box::pin(body.into_data_stream()),
        load,
        output_frame,
        producer_abort,
    };
    Response::from_parts(parts, Body::from_stream(stream))
}

struct TypedDispatch<'a> {
    headers: Option<&'a HeaderMap>,
    route: &'a str,
    worker_url: &'a str,
    is_stream: bool,
    completion: Option<&'a CompletionRoute<'a>>,
    load: RequestLoadGuard,
    prepared: Option<crate::backend::PreparedRequest>,
}

struct SelectedWorker {
    worker: Arc<dyn Worker>,
    kv_receipt: Option<KvSelectionReceipt>,
}

/// Regular router that uses injected load balancing policies
#[derive(Debug)]
pub struct Router {
    epd: Option<super::epd::EncoderStage>,
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    client: Client,
    worker_startup_timeout_secs: u64,
    worker_startup_check_interval_secs: u64,
    intra_node_data_parallel_size: usize,
    api_key: Option<String>,
    retry_config: RetryConfig,
    circuit_breaker_config: CircuitBreakerConfig,
    kv_aware_session_affinity_ttl_secs: Option<u64>,
    health_config: HealthConfig,
    frontend: crate::backend::EngineFrontend,
    completion_input: Option<crate::backend::completion_input::CompletionInput>,
    completion_preparation: crate::backend::completion_input::CompletionPreparation,
    _worker_loads: Arc<tokio::sync::watch::Receiver<HashMap<String, isize>>>,
    _load_monitor_handle: Option<Arc<tokio::task::JoinHandle<()>>>,
    kv_index_supervisor: Option<Arc<crate::kv_index::KvIndexSupervisor>>,
    _kv_signal_handle: Option<Arc<tokio::task::JoinHandle<()>>>,
    program_scheduler: Option<Arc<ProgramScheduler>>,
    _program_observation_handle: Option<Arc<tokio::task::JoinHandle<()>>>,
    program_targets_cache: Mutex<HashMap<String, ProgramTargetCacheEntry>>,
    program_token_estimator: Arc<MomentumTokenEstimator>,
}

#[derive(Debug, Clone)]
struct ProgramTargetCacheEntry {
    registry_revision: u64,
    targets: Arc<[ProgramTarget]>,
}

const PROGRAM_MODEL_POOL_ALL: &str = "program:all";
const PROGRAM_MODEL_POOL_FALLBACK: &str = "program:fallback:unlabeled";
const PROGRAM_MODEL_POOL_NAMED_PREFIX: &str = "program:model:";
const UNLABELED_WORKER_MODEL_ID: &str = "unknown";

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedProgramModelPool {
    scheduler_key: String,
    registry_model: Option<String>,
}

impl Router {
    /// Create a new router with injected policy and client
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        worker_urls: Vec<String>,
        ctx: &Arc<crate::server::AppContext>,
    ) -> Result<Self, String> {
        // Update active workers gauge
        RouterMetrics::set_active_workers(worker_urls.len());

        // All-http or all-grpc. Mixed schemes fail here (not a silent fallback).
        let pool_kind = crate::backend::classify_worker_urls(&worker_urls)?;
        let default_policy = ctx.policy_registry.get_default_policy();
        let pending_kv_supervisor = if kv_aware_ops(default_policy.as_ref()).is_some() {
            let (supervisor, signals) = crate::kv_index::KvIndexSupervisor::new_http_dense(
                ctx.client.clone(),
                crate::kv_index::HashMode::Sha256,
            );
            let static_sources = match &ctx.router_config.policy {
                crate::config::PolicyConfig::KvAware {
                    static_kv_event_sources,
                    ..
                } => static_kv_event_sources.as_ref().clone(),
                _ => HashMap::new(),
            };
            let supervisor = supervisor.with_static_sources(static_sources);
            for worker_url in &worker_urls {
                supervisor.validate_http_worker(
                    worker_url,
                    ctx.router_config.intra_node_data_parallel_size,
                )?;
            }
            Some((supervisor, signals))
        } else {
            None
        };
        let request_timeout = Duration::from_secs(ctx.router_config.request_timeout_secs);
        let completion_enabled = std::env::var_os("VLLM_ROUTER_COMPLETION_MODEL_ALIAS").is_some();
        let frontend = if pool_kind == Some(crate::backend::WorkerPoolKind::Http) {
            crate::backend::frontend::warn_ignored_encoding_caches();
            crate::backend::EngineFrontend::with_request_timeout(request_timeout)
        } else {
            crate::backend::EngineFrontend::from_env(request_timeout)
                .map_err(|error| format!("tokenizer cache: {error}"))?
        };
        let completion_input = if completion_enabled {
            if pool_kind != Some(crate::backend::WorkerPoolKind::Http)
                || ctx.router_config.intra_node_data_parallel_size != 1
                || kv_aware_ops(ctx.policy_registry.get_default_policy().as_ref()).is_none()
            {
                return Err("Completion input binding requires static HTTP Regular kv_aware with independent DP1 Workers".into());
            }
            crate::backend::completion_input::CompletionInput::from_env(&frontend)
                .await
                .map_err(|error| format!("Completion input: {error:#}"))?
        } else {
            None
        };

        // Wait for workers to be healthy (skip if empty - for service discovery mode)
        if !worker_urls.is_empty() {
            Self::wait_for_healthy_workers(
                &worker_urls,
                ctx.router_config.worker_startup_timeout_secs,
                ctx.router_config.worker_startup_check_interval_secs,
            )
            .await?;
        }

        let kv_discovery_urls = worker_urls.clone();

        // Automatically expand to DP-aware workers when intra_node_data_parallel_size > 1
        let worker_urls = if ctx.router_config.intra_node_data_parallel_size > 1 {
            // worker address now in the format of "http://host:port@dp_rank"
            dp_utils::get_dp_aware_workers(
                &worker_urls,
                &ctx.router_config.api_key,
                ctx.router_config.intra_node_data_parallel_size,
            )
            .await
            .map_err(|e| format!("Failed to get dp-aware workers: {}", e))?
        } else {
            worker_urls
        };

        // Convert config CircuitBreakerConfig to core CircuitBreakerConfig
        let circuit_breaker_config = ctx.router_config.effective_circuit_breaker_config();
        let core_cb_config = CircuitBreakerConfig {
            failure_threshold: circuit_breaker_config.failure_threshold,
            success_threshold: circuit_breaker_config.success_threshold,
            timeout_duration: Duration::from_secs(circuit_breaker_config.timeout_duration_secs),
            window_duration: Duration::from_secs(circuit_breaker_config.window_duration_secs),
        };

        // Register workers in the registry
        // In IGW mode, we need to fetch model info from workers
        let dp_size = ctx.router_config.intra_node_data_parallel_size;
        let health_config = HealthConfig {
            timeout_secs: ctx.router_config.health_check.timeout_secs,
            check_interval_secs: ctx.router_config.health_check.check_interval_secs,
            endpoint: ctx.router_config.health_check.endpoint.clone(),
            failure_threshold: ctx.router_config.health_check.failure_threshold,
            success_threshold: ctx.router_config.health_check.success_threshold,
        };
        let mut kv_worker_info = HashMap::new();
        let (kv_index_supervisor, kv_signal_handle) =
            if let Some((supervisor, mut signals)) = pending_kv_supervisor {
                let supervisor = Arc::new(supervisor);
                for worker_url in &kv_discovery_urls {
                    match supervisor.on_worker_added(worker_url).await {
                        Ok(info) => {
                            kv_worker_info.insert(worker_url.clone(), info);
                        }
                        Err(error) => warn!(
                            worker_url,
                            %error,
                            "kv_aware discovery failed; worker will use load-only scoring"
                        ),
                    }
                }
                let signal_handle = Arc::new(tokio::spawn(async move {
                    while signals.recv().await.is_some() {}
                }));
                (Some(supervisor), Some(signal_handle))
            } else {
                (None, None)
            };
        for url in &worker_urls {
            // TODO: In IGW mode, fetch model_id from worker's /get_model_info endpoint
            // For now, create worker without model_id
            let worker_arc: Arc<dyn Worker> = if dp_size > 1 {
                let (base_url, dp_rank) = dp_utils::parse_worker_url(url);
                Arc::new(
                    DPAwareWorker::new(
                        base_url,
                        dp_rank.unwrap_or(0),
                        dp_size,
                        WorkerType::Regular,
                    )
                    .with_circuit_breaker_config(core_cb_config.clone())
                    .with_health_config(health_config.clone()),
                )
            } else {
                Arc::new(
                    BasicWorker::new(url.clone(), WorkerType::Regular)
                        .with_circuit_breaker_config(core_cb_config.clone())
                        .with_health_config(health_config.clone()),
                )
            };
            ctx.worker_registry.register(worker_arc.clone());

            if let Some(kv_aware) = kv_aware_ops(default_policy.as_ref()) {
                let provider = kv_index_supervisor.as_ref().and_then(|supervisor| {
                    supervisor.match_provider_for_worker(worker_arc.base_url())
                });
                kv_aware.register_worker(
                    worker_arc.as_ref(),
                    kv_worker_info.get(worker_arc.base_url()),
                    provider,
                );
            }

            // Notify PolicyRegistry about the new worker
            let model_id = worker_arc.model_id();
            let policy = ctx.policy_registry.on_worker_added(model_id, None);

            // If this is a cache-aware policy and it's the first worker for this model,
            // initialize it with the worker
            if policy.needs_cache_routing_key() {
                if let Some(cache_aware) = policy
                    .as_any()
                    .downcast_ref::<crate::policies::CacheAwarePolicy>()
                {
                    let worker_dyn: Arc<dyn Worker> = worker_arc.clone();
                    cache_aware.init_workers(std::slice::from_ref(&worker_dyn));
                }
            }
        }

        // Setup load monitoring for PowerOfTwo policy
        let (tx, rx) = tokio::sync::watch::channel(HashMap::new());
        let worker_loads = Arc::new(rx);

        // Check if default policy is power_of_two for load monitoring
        let load_monitor_handle = if default_policy.name() == "power_of_two" {
            let monitor_urls = worker_urls.clone();
            let monitor_interval = ctx.router_config.worker_startup_check_interval_secs;
            let policy_clone = default_policy.clone();
            let client_clone = ctx.client.clone();

            Some(Arc::new(tokio::spawn(async move {
                Self::monitor_worker_loads(
                    monitor_urls,
                    tx,
                    monitor_interval,
                    policy_clone,
                    client_clone,
                )
                .await;
            })))
        } else {
            None
        };

        if let Some(config) = ctx.router_config.program_scheduling.as_ref() {
            Self::warn_on_default_program_calibration(config);
        }
        let program_scheduler = ctx
            .router_config
            .program_scheduling
            .as_ref()
            .map(ProgramSchedulerConfig::from)
            .map(ProgramScheduler::new)
            .map(Arc::new);
        let program_observation_handle = program_scheduler.as_ref().map(|scheduler| {
            let scheduler = scheduler.clone();
            let worker_registry = ctx.worker_registry.clone();
            let provider = VllmMetricsObservationProvider::new(
                ctx.client.clone(),
                ctx.router_config.api_key.clone(),
            );
            Arc::new(tokio::spawn(async move {
                let mut interval = tokio::time::interval(scheduler.metrics_interval());
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                loop {
                    interval.tick().await;
                    Self::sync_program_targets_from_registry(&scheduler, &worker_registry);
                    let targets = scheduler.all_targets();
                    if targets.is_empty() {
                        continue;
                    }
                    let epoch = scheduler.begin_observation(&targets);
                    let (observations, failures) = provider.observe(&targets).await;
                    for failure in failures {
                        warn!(
                            base_url = failure.base_url,
                            error = failure.error,
                            "Program scheduling backend observation failed"
                        );
                    }
                    scheduler.apply_observations(epoch, observations);
                    scheduler.tick();
                }
            }))
        });

        Ok(Router {
            epd: ctx
                .router_config
                .epd
                .clone()
                .map(super::epd::EncoderStage::new),
            worker_registry: ctx.worker_registry.clone(),
            policy_registry: ctx.policy_registry.clone(),
            client: ctx.client.clone(),
            worker_startup_timeout_secs: ctx.router_config.worker_startup_timeout_secs,
            worker_startup_check_interval_secs: ctx
                .router_config
                .worker_startup_check_interval_secs,
            intra_node_data_parallel_size: ctx.router_config.intra_node_data_parallel_size,
            api_key: ctx.router_config.api_key.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
            circuit_breaker_config: core_cb_config,
            kv_aware_session_affinity_ttl_secs: ctx
                .router_config
                .kv_aware_session_affinity_ttl_secs,
            health_config,
            frontend,
            completion_input,
            completion_preparation: crate::backend::completion_input::CompletionPreparation::new(
                ctx.router_config.max_concurrent_requests,
                request_timeout,
                Duration::from_secs(ctx.router_config.queue_timeout_secs),
            ),
            _worker_loads: worker_loads,
            _load_monitor_handle: load_monitor_handle,
            kv_index_supervisor,
            _kv_signal_handle: kv_signal_handle,
            program_scheduler,
            _program_observation_handle: program_observation_handle,
            program_targets_cache: Mutex::new(HashMap::new()),
            program_token_estimator: Arc::new(MomentumTokenEstimator::default()),
        })
    }

    /// Test hook: pin token ids so gRPC e2e does not load a model.
    pub fn pin_test_token_ids(&self, token_ids: Vec<u32>) {
        self.frontend.pin_test_token_ids(token_ids);
    }

    async fn discover_kv_worker(&self, worker_url: &str) -> Option<crate::kv_index::WorkerKvInfo> {
        let supervisor = self.kv_index_supervisor.as_ref()?;
        let policy = self.policy_registry.get_default_policy();
        kv_aware_ops(policy.as_ref())?;
        match supervisor.on_worker_added(worker_url).await {
            Ok(info) => Some(info),
            Err(error) => {
                warn!(
                    worker_url,
                    %error,
                    "kv_aware discovery failed; worker will use load-only scoring"
                );
                None
            }
        }
    }

    fn register_kv_target(
        &self,
        worker: &dyn Worker,
        info: Option<&crate::kv_index::WorkerKvInfo>,
    ) {
        let policy = self.policy_registry.get_default_policy();
        let Some(kv_aware) = kv_aware_ops(policy.as_ref()) else {
            return;
        };
        let provider = self
            .kv_index_supervisor
            .as_ref()
            .and_then(|supervisor| supervisor.match_provider_for_worker(worker.base_url()));
        kv_aware.register_worker(worker, info, provider);
    }

    fn remove_kv_worker(&self, worker_url: &str, target_urls: &[String]) {
        if let Some(supervisor) = &self.kv_index_supervisor {
            supervisor.on_worker_removed(worker_url);
        }
        let policy = self.policy_registry.get_default_policy();
        if let Some(kv_aware) = kv_aware_ops(policy.as_ref()) {
            for target_url in target_urls {
                kv_aware.unregister_worker(target_url);
            }
        }
    }

    fn warn_on_default_program_calibration(config: &crate::config::types::ProgramSchedulingConfig) {
        if config.binding_only {
            return;
        }
        let defaulted_fields = config.defaulted_calibration_fields();
        if defaulted_fields.is_empty() {
            return;
        }

        const REFERENCE_PROMPT_TOKENS: f64 = 50_000.0;
        const REFERENCE_DECODE_BATCH_SIZE: f64 = 4.0;
        const REFERENCE_DECODE_CONTEXT_TOKENS_PER_REQUEST: f64 = 50_000.0;
        const REFERENCE_DECODE_CONTEXT_TOKENS: f64 = 200_000.0;
        let prompt_1k = REFERENCE_PROMPT_TOKENS / 1_000.0;
        let prefill = config.prefill_cost_model;
        let reference_prefill_seconds = prefill.intercept_seconds
            + prefill.linear_seconds_per_1k_tokens * prompt_1k
            + prefill.quadratic_seconds_per_1k_tokens_squared * prompt_1k * prompt_1k;
        let decode = config.decode_throughput_model;
        let reference_decode_throughput_tokens_per_second = REFERENCE_DECODE_BATCH_SIZE
            / (decode.fixed_step_seconds
                + decode.batch_step_seconds_per_request * REFERENCE_DECODE_BATCH_SIZE
                + decode.context_step_seconds_per_token * REFERENCE_DECODE_CONTEXT_TOKENS);

        warn!(
            event = "program_scheduling_reference_calibration",
            defaulted_fields = %defaulted_fields.join(","),
            reference_prefill_tokens = REFERENCE_PROMPT_TOKENS as u64,
            reference_prefill_seconds,
            reference_decode_batch_size = REFERENCE_DECODE_BATCH_SIZE as u64,
            reference_decode_context_tokens_per_request = REFERENCE_DECODE_CONTEXT_TOKENS_PER_REQUEST as u64,
            reference_decode_context_tokens = REFERENCE_DECODE_CONTEXT_TOKENS as u64,
            reference_decode_throughput_tokens_per_second,
            calibration_docs = "docs/program_scheduling.md#offline-calibration-models",
            "Program scheduling is using one or more reference calibration coefficients; deployment-specific performance differences may degrade scheduling decisions"
        );
    }

    fn program_targets(workers: &[Arc<dyn Worker>]) -> Vec<ProgramTarget> {
        workers
            .iter()
            .filter(|worker| worker.is_available())
            .map(|worker| {
                let (base_url, parsed_rank) = dp_utils::parse_worker_url(worker.url());
                ProgramTarget {
                    id: worker.url().to_string(),
                    base_url,
                    dp_rank: worker.dp_rank().or(parsed_rank),
                }
            })
            .collect()
    }

    /// Refresh every tracked model pool from one revision-stable registry view.
    fn sync_program_targets_from_registry(
        scheduler: &ProgramScheduler,
        worker_registry: &WorkerRegistry,
    ) {
        Self::sync_program_targets_from_registry_with_hook(scheduler, worker_registry, || {});
    }

    fn sync_program_targets_from_registry_with_hook(
        scheduler: &ProgramScheduler,
        worker_registry: &WorkerRegistry,
        mut after_collection: impl FnMut(),
    ) {
        let model_pools = scheduler.model_pools();
        if model_pools.is_empty() {
            return;
        }
        loop {
            let revision = worker_registry.revision();
            let snapshots = model_pools
                .iter()
                .map(|model_pool| {
                    let workers = Self::workers_for_program_model_pool(worker_registry, model_pool);
                    let targets: Arc<[ProgramTarget]> = Self::program_targets(&workers).into();
                    (model_pool.clone(), targets)
                })
                .collect::<Vec<_>>();
            if worker_registry.revision() != revision {
                continue;
            }
            after_collection();
            scheduler.sync_target_snapshots(snapshots);
            if worker_registry.revision() != revision {
                continue;
            }
            return;
        }
    }

    fn resolved_program_model_pool(&self, model_id: Option<&str>) -> ResolvedProgramModelPool {
        match model_id {
            Some(model) if self.worker_registry.has_model(model) => ResolvedProgramModelPool {
                scheduler_key: format!("{PROGRAM_MODEL_POOL_NAMED_PREFIX}{model}"),
                registry_model: Some(model.to_string()),
            },
            Some(_) => ResolvedProgramModelPool {
                scheduler_key: PROGRAM_MODEL_POOL_FALLBACK.to_string(),
                registry_model: Some(UNLABELED_WORKER_MODEL_ID.to_string()),
            },
            None => ResolvedProgramModelPool {
                scheduler_key: PROGRAM_MODEL_POOL_ALL.to_string(),
                registry_model: None,
            },
        }
    }

    fn workers_for_program_model_pool(
        worker_registry: &WorkerRegistry,
        model_pool: &str,
    ) -> Vec<Arc<dyn Worker>> {
        if model_pool == PROGRAM_MODEL_POOL_ALL {
            worker_registry.get_all()
        } else if model_pool == PROGRAM_MODEL_POOL_FALLBACK {
            worker_registry.get_by_model_fast(UNLABELED_WORKER_MODEL_ID)
        } else if let Some(model) = model_pool.strip_prefix(PROGRAM_MODEL_POOL_NAMED_PREFIX) {
            worker_registry.get_by_model_fast(model)
        } else {
            Vec::new()
        }
    }

    fn program_targets_for_model(
        &self,
        model_pool: &ResolvedProgramModelPool,
    ) -> Arc<[ProgramTarget]> {
        loop {
            let registry_revision = self.worker_registry.revision();
            let cache_key = model_pool.scheduler_key.clone();
            if let Some(entry) = self.program_targets_cache.lock().get(&cache_key) {
                if entry.registry_revision == registry_revision {
                    return entry.targets.clone();
                }
            }
            let workers = match model_pool.registry_model.as_deref() {
                Some(model) => self.worker_registry.get_by_model_fast(model),
                None => self.worker_registry.get_all(),
            };
            let targets: Arc<[ProgramTarget]> = Self::program_targets(&workers).into();
            if self.worker_registry.revision() != registry_revision {
                continue;
            }
            self.program_targets_cache.lock().insert(
                cache_key,
                ProgramTargetCacheEntry {
                    registry_revision,
                    targets: targets.clone(),
                },
            );
            return targets;
        }
    }

    fn schedule_error_response(error: ScheduleError) -> Response {
        match error {
            ScheduleError::InvalidIdentity(_) => {
                (StatusCode::BAD_REQUEST, error.to_string()).into_response()
            }
            ScheduleError::NoTargets => {
                (StatusCode::SERVICE_UNAVAILABLE, error.to_string()).into_response()
            }
            ScheduleError::QueueTimeout => {
                (StatusCode::TOO_MANY_REQUESTS, error.to_string()).into_response()
            }
        }
    }

    fn program_target_unavailable_response(route: &str) -> Response {
        RouterMetrics::record_request_error(route, "program_target_unavailable");
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Program-bound worker is unavailable",
        )
            .into_response()
    }

    fn should_buffer_transparent_response(is_stream: bool, tracked: bool) -> bool {
        !is_stream && tracked
    }

    fn should_proxy_transparent_directly(tracked: bool) -> bool {
        !tracked
    }

    async fn acquire_program_completion<T: GenerationRequest>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        model_id: Option<&str>,
        endpoint: &str,
    ) -> Result<Option<ProgramCompletion>, ScheduleError> {
        let Some(scheduler) = &self.program_scheduler else {
            return Ok(None);
        };
        let request = typed_req.extract_program_identity_payload();
        let model_pool = self.resolved_program_model_pool(model_id);
        let Some(identity) = ProgramIdentity::from_request_with_enable_key(
            headers,
            request.as_ref(),
            Some(&model_pool.scheduler_key),
            scheduler.enable_key(),
        )?
        else {
            return Ok(None);
        };
        let input_text = typed_req.extract_text_for_program_scheduling();
        self.acquire_program_completion_for_identity(
            scheduler,
            identity,
            &model_pool,
            endpoint,
            &input_text,
        )
        .await
    }

    async fn acquire_program_completion_from_payload(
        &self,
        headers: Option<&HeaderMap>,
        request: Option<&serde_json::Value>,
        model_id: Option<&str>,
        endpoint: &str,
        input_text: &str,
    ) -> Result<Option<ProgramCompletion>, ScheduleError> {
        let Some(scheduler) = &self.program_scheduler else {
            return Ok(None);
        };
        let model_pool = self.resolved_program_model_pool(model_id);
        let Some(identity) = ProgramIdentity::from_request_with_enable_key(
            headers,
            request,
            Some(&model_pool.scheduler_key),
            scheduler.enable_key(),
        )?
        else {
            return Ok(None);
        };
        self.acquire_program_completion_for_identity(
            scheduler,
            identity,
            &model_pool,
            endpoint,
            input_text,
        )
        .await
    }

    async fn acquire_program_completion_for_identity(
        &self,
        scheduler: &Arc<ProgramScheduler>,
        identity: ProgramIdentity,
        model_pool: &ResolvedProgramModelPool,
        endpoint: &str,
        input_text: &str,
    ) -> Result<Option<ProgramCompletion>, ScheduleError> {
        let (estimated_context_tokens, calibration) = self.program_token_estimator.estimate(
            TokenEstimateScope::new(&model_pool.scheduler_key, endpoint),
            input_text,
        );
        let targets = self.program_targets_for_model(model_pool);
        let routing_text = scheduler
            .uses_cache_aware_binding()
            .then(|| input_text.to_string());
        let dispatch = scheduler
            .acquire_from_snapshot(identity, estimated_context_tokens, targets, routing_text)
            .await?;
        Ok(Some(ProgramCompletion::new(
            scheduler.clone(),
            dispatch,
            self.program_token_estimator.clone(),
            calibration,
        )))
    }

    /// Get the current list of worker URLs
    pub fn get_worker_urls(&self) -> Vec<String> {
        self.worker_registry.get_all_urls()
    }

    /// Get worker URLs for a specific model
    pub fn get_worker_urls_for_model(&self, model_id: Option<&str>) -> Vec<String> {
        let workers = match model_id {
            Some(model) => self.worker_registry.get_by_model_fast(model),
            None => self.worker_registry.get_all(),
        };
        workers.iter().map(|w| w.url().to_string()).collect()
    }

    pub async fn wait_for_healthy_workers(
        worker_urls: &[String],
        worker_startup_timeout_secs: u64,
        worker_startup_check_interval_secs: u64,
    ) -> Result<(), String> {
        if worker_urls.is_empty() {
            return Err(
                "Timeout waiting for workers to become healthy: no workers provided".to_string(),
            );
        }

        // Perform health check asynchronously
        Self::wait_for_healthy_workers_async(
            worker_urls,
            worker_startup_timeout_secs,
            worker_startup_check_interval_secs,
        )
        .await
    }

    async fn wait_for_healthy_workers_async(
        worker_urls: &[String],
        worker_startup_timeout_secs: u64,
        worker_startup_check_interval_secs: u64,
    ) -> Result<(), String> {
        // Extract unique base URLs (hosts) for health checks
        // This deduplicates DP-aware URLs like http://host:8081@0, @1, @2, @3
        // to only check http://host:8081 once
        use std::collections::HashSet;
        let mut unique_hosts = HashSet::new();
        let mut host_to_workers: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();

        for url in worker_urls {
            // Extract base URL by removing @rank suffix if present
            let base_url = if let Some(at_pos) = url.rfind('@') {
                url[..at_pos].to_string()
            } else {
                url.clone()
            };

            unique_hosts.insert(base_url.clone());
            host_to_workers
                .entry(base_url)
                .or_default()
                .push(url.clone());
        }

        let unique_hosts_vec: Vec<String> = unique_hosts.into_iter().collect();

        info!(
            "Waiting for {} unique hosts (representing {} workers) to become healthy (timeout: {}s)",
            unique_hosts_vec.len(),
            worker_urls.len(),
            worker_startup_timeout_secs
        );

        let start_time = std::time::Instant::now();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        loop {
            if start_time.elapsed() > Duration::from_secs(worker_startup_timeout_secs) {
                error!(
                    "Timeout {}s waiting for hosts {:?} to become healthy. Please set --router-worker-startup-timeout-secs (vllm_router.launch_server) or --worker-startup-timeout-secs (vllm_worker.router) to a larger value",
                    worker_startup_timeout_secs, unique_hosts_vec
                );
                return Err(format!(
                    "Timeout {}s waiting for hosts {:?} to become healthy. Please set --router-worker-startup-timeout-secs (vllm_router.launch_server) or --worker-startup-timeout-secs (vllm_worker.router) to a larger value",
                    worker_startup_timeout_secs, unique_hosts_vec
                ));
            }

            // Perform health checks only on unique hosts (not per DP rank)
            let mut health_checks = Vec::new();
            for base_url in &unique_hosts_vec {
                let client_clone = client.clone();
                let url_clone = base_url.clone();

                let check_health = tokio::spawn(async move {
                    if crate::backend::is_grpc_url(&url_clone) {
                        match crate::backend::check_grpc_health(&url_clone, Duration::from_secs(2))
                            .await
                        {
                            Ok(()) => None,
                            Err(e) => Some((url_clone, e)),
                        }
                    } else {
                        let health_url = format!("{}/health", url_clone);
                        match client_clone.get(&health_url).send().await {
                            Ok(res) => {
                                if res.status().is_success() {
                                    None
                                } else {
                                    Some((url_clone, format!("status: {}", res.status())))
                                }
                            }
                            Err(_) => Some((url_clone, "not ready".to_string())),
                        }
                    }
                });

                health_checks.push(check_health);
            }

            // Wait for all health checks to complete
            let results = futures::future::join_all(health_checks).await;

            let mut unhealthy_hosts = Vec::new();
            let mut healthy_host_count = 0;

            for result in results {
                match result {
                    Ok(None) => {
                        healthy_host_count += 1;
                        // Host is healthy
                    }
                    Ok(Some((url, reason))) => {
                        unhealthy_hosts.push((url, reason));
                    }
                    Err(e) => {
                        unhealthy_hosts.push(("unknown".to_string(), format!("task error: {}", e)));
                    }
                }
            }

            if healthy_host_count > 0 {
                info!(
                    "{} out of {} unique hosts are healthy (representing {} workers)",
                    healthy_host_count,
                    unique_hosts_vec.len(),
                    worker_urls.len()
                );
                return Ok(());
            } else {
                debug!(
                   "Waiting for at least 1 of {} unique hosts to become healthy ({} unhealthy: {:?})",
                    unique_hosts_vec.len(),
                    unhealthy_hosts.len(),
                    unhealthy_hosts
                );
                tokio::time::sleep(Duration::from_secs(worker_startup_check_interval_secs)).await;
            }
        }
    }

    fn select_first_worker(&self) -> Result<String, String> {
        // Prefer an available worker so proxy GETs (e.g. /v1/models) don't 500
        // when an arbitrary registered worker is temporarily down or its
        // circuit breaker is open.
        let workers = self.worker_registry.get_all();
        match workers.into_iter().find(|w| w.is_available()) {
            Some(worker) => Ok(worker.url().to_string()),
            None => Err("No workers are currently available".to_string()),
        }
    }

    #[allow(dead_code)]
    fn select_first_worker_for_model(&self, model_id: Option<&str>) -> Result<String, String> {
        let workers = match model_id {
            Some(model) => self.worker_registry.get_by_model_fast(model),
            None => self.worker_registry.get_all(),
        };
        if workers.is_empty() {
            Err(format!(
                "No workers are available for model: {:?}",
                model_id
            ))
        } else {
            Ok(workers[0].url().to_string())
        }
    }

    pub async fn send_health_check(&self, worker_url: &str) -> Response {
        let health_url = if self.intra_node_data_parallel_size > 1 {
            // Need to extract the URL from "http://host:port@dp_rank"
            match dp_utils::extract_dp_rank(worker_url) {
                Ok((worker_url_prefix, _dp_rank)) => worker_url_prefix,
                Err(e) => {
                    error!("Failed to extract dp_rank for health check: {}", e);
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to extract dp_rank: {}", e),
                    )
                        .into_response();
                }
            }
        } else {
            worker_url
        };

        if crate::backend::is_grpc_url(health_url) {
            return match crate::backend::check_grpc_health(health_url, Duration::from_secs(2)).await
            {
                Ok(()) => StatusCode::OK.into_response(),
                Err(e) => (StatusCode::BAD_GATEWAY, e).into_response(),
            };
        }

        let request_builder = self.client.get(format!("{}/health", health_url));

        let response = match request_builder.send().await {
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                match res.bytes().await {
                    Ok(body) => (status, body).into_response(),
                    Err(e) => {
                        error!(
                            worker_url = %health_url,
                            error = %e,
                            "Failed to read health response body"
                        );
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Failed to read response body: {}", e),
                        )
                            .into_response()
                    }
                }
            }
            Err(e) => {
                error!(
                    worker_url = %health_url,
                    error = %e,
                    "Failed to send health request to worker"
                );
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to send request to worker {}: {}", health_url, e),
                )
                    .into_response()
            }
        };

        // Don't record metrics for health checks
        response
    }

    // Helper method to proxy GET requests to the first available worker
    async fn proxy_get_request(&self, req: Request<Body>, endpoint: &str) -> Response {
        let incoming_headers = req.headers();
        let headers = header_utils::copy_request_headers(&req);

        match self.select_first_worker() {
            Ok(worker_url) => {
                let (base_url, dp_rank) = dp_utils::parse_worker_url(&worker_url);
                let url = format!("{}/{}", base_url, endpoint);
                let route_name = format!("/{}", endpoint);
                let mut request_builder =
                    dp_utils::add_dp_rank_header(self.client.get(&url), dp_rank);

                for (name, value) in headers {
                    let name_lc = name.to_lowercase();
                    // When the router selects a DP rank, it owns the
                    // X-data-parallel-rank header: skip any client-supplied
                    // value so the worker sees exactly one rank (ours).
                    if name_lc != "content-type"
                        && name_lc != "content-length"
                        && !(dp_rank.is_some() && name_lc == "x-data-parallel-rank")
                        && !header_utils::TRACE_HEADER_NAMES.contains(&name_lc.as_str())
                    {
                        request_builder = request_builder.header(name, value);
                    }
                }

                match otel_http::send_client_request(
                    request_builder,
                    Some(incoming_headers),
                    ClientRequestOptions {
                        method: "GET",
                        url: &url,
                        route: Some(&route_name),
                        request_phase: None,
                    },
                )
                .await
                {
                    Ok(res) => {
                        let status = StatusCode::from_u16(res.status().as_u16())
                            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                        // Preserve headers from backend
                        let response_headers =
                            header_utils::preserve_response_headers(res.headers());

                        match res.bytes().await {
                            Ok(body) => {
                                let mut response = Response::new(axum::body::Body::from(body));
                                *response.status_mut() = status;
                                *response.headers_mut() = response_headers;
                                response
                            }
                            Err(e) => (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                format!("Failed to read response: {}", e),
                            )
                                .into_response(),
                        }
                    }
                    Err(e) => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Request failed: {}", e),
                    )
                        .into_response(),
                }
            }
            Err(e) => (StatusCode::SERVICE_UNAVAILABLE, e).into_response(),
        }
    }

    /// Convert axum HeaderMap to policy RequestHeaders (HashMap<String, String>)
    fn headers_to_request_headers(
        headers: Option<&HeaderMap>,
    ) -> Option<crate::policies::RequestHeaders> {
        headers.map(|h| {
            h.iter()
                .filter_map(|(name, value)| {
                    value
                        .to_str()
                        .ok()
                        .map(|v| (name.as_str().to_lowercase(), v.to_string()))
                })
                .collect()
        })
    }

    fn policy_for_model(&self, model_id: Option<&str>) -> Arc<dyn LoadBalancingPolicy> {
        match model_id {
            Some(model) => self.policy_registry.get_policy_or_default(model),
            None => self.policy_registry.get_default_policy(),
        }
    }

    fn available_workers(&self, model_id: Option<&str>) -> Vec<Arc<dyn Worker>> {
        let workers = match model_id {
            Some(model) => self.worker_registry.get_by_model_fast(model),
            None => self.worker_registry.get_all(),
        };
        workers
            .into_iter()
            .filter(|worker| worker.is_available())
            .collect()
    }

    /// Select worker for a specific model considering circuit breaker state
    #[allow(clippy::too_many_arguments)]
    fn select_worker_for_model(
        &self,
        model_id: Option<&str>,
        available: &[Arc<dyn Worker>],
        text: Option<&str>,
        headers: Option<&HeaderMap>,
        token_ids: KvRequestTokens<'_>,
        eligibility: Option<&[bool]>,
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
        affinity_key: Option<&AffinityKey>,
        excluded_workers: &[String],
    ) -> Option<SelectedWorker> {
        if available.is_empty() {
            return None;
        }

        let policy = self.policy_for_model(model_id);

        // Convert headers for policies that need them (e.g., consistent_hash)
        let request_headers = Self::headers_to_request_headers(headers);

        if let Some(kv_aware) = kv_aware_policy(policy.as_ref()) {
            // The policy sees the full list: a retry exclusion must not
            // rewrite a still-valid binding.
            let receipt = kv_aware.select_worker_with_qualified_inputs(
                available,
                QualifiedKvInput {
                    tokens: token_ids,
                    eligibility,
                },
                lora_name,
                cache_salt,
                affinity_key,
                excluded_workers,
            )?;
            return Some(SelectedWorker {
                worker: available.get(receipt.worker_index)?.clone(),
                kv_receipt: Some(receipt),
            });
        }

        let idx = policy.select_worker_with_headers(available, text, request_headers.as_ref())?;
        Some(SelectedWorker {
            worker: available.get(idx)?.clone(),
            kv_receipt: None,
        })
    }

    pub async fn route_typed_request<T: GenerationRequest + serde::Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &str,
        model_id: Option<&str>,
    ) -> Response {
        self.route_prepared_request(headers, typed_req, route, model_id, None, None)
            .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn route_prepared_request<T: GenerationRequest + serde::Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &str,
        model_id: Option<&str>,
        completion: Option<Arc<CompletionRequest>>,
        raw_body: Option<&axum::body::Bytes>,
    ) -> Response {
        let start = Instant::now();
        let is_stream = typed_req.is_stream();
        let model_id = model_id
            .or_else(|| typed_req.get_model())
            .filter(|model| self.worker_registry.has_model(model));

        // Re-check live URLs. Mix is rejected at init / add_worker; this
        // catches a registry that somehow became mixed.
        let pool = match crate::backend::classify_worker_urls(&self.get_worker_urls()) {
            Ok(kind) => kind,
            Err(e) => {
                return (StatusCode::INTERNAL_SERVER_ERROR, e).into_response();
            }
        };

        // Reject unsupported gRPC routes before admitting a request or
        // incrementing the selected worker's load.
        if matches!(pool, Some(crate::backend::WorkerPoolKind::Grpc))
            && !matches!(route, "/v1/chat/completions" | "/v1/completions")
        {
            return (
                StatusCode::BAD_REQUEST,
                format!("gRPC backend currently supports /v1/chat/completions and /v1/completions, not {route}"),
            )
                .into_response();
        }

        // All-grpc chat: tokenize once, outside policy and retry.
        // Cache-aware placement uses the same full content/session key as
        // HTTP. PreparedRequest retains full token IDs for backend dispatch.
        //
        // The concrete request comes from `GenerationRequest::grpc_request`
        // rather than a `serde_json::Value` roundtrip: the only two routes a
        // gRPC pool admits are exactly the two concrete request types.
        let prepared = match (pool, typed_req.grpc_request()) {
            (Some(crate::backend::WorkerPoolKind::Grpc), Some(GrpcRequestRef::Chat(chat)))
                if route == "/v1/chat/completions" =>
            {
                match self.frontend.prepare(chat.clone()).await {
                    Ok(prepared) => Some(prepared),
                    Err(error) => {
                        return (prepare_error_status(&error), error.to_string()).into_response();
                    }
                }
            }
            (
                Some(crate::backend::WorkerPoolKind::Grpc),
                Some(GrpcRequestRef::Completion(completion)),
            ) if route == "/v1/completions" => {
                let lowered = match crate::backend::prepare_grpc_completion(completion) {
                    Ok(lowered) => lowered,
                    Err(error) => return (StatusCode::BAD_REQUEST, error).into_response(),
                };
                match self
                    .frontend
                    .prepare_token_ids(lowered.chat, lowered.token_ids, lowered.emit_token_ids)
                    .await
                {
                    Ok(prepared) => Some(prepared),
                    Err(error) => {
                        return (prepare_error_status(&error), error.to_string()).into_response();
                    }
                }
            }
            (Some(crate::backend::WorkerPoolKind::Grpc), _) => {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("gRPC backend received a request body that does not match {route}"),
                )
                    .into_response();
            }
            _ => None,
        };

        let completion = match completion
            .filter(|_| matches!(pool, Some(crate::backend::WorkerPoolKind::Http)))
        {
            Some(request) => {
                let policy = match model_id {
                    Some(model) => self.policy_registry.get_policy_or_default(model),
                    None => self.policy_registry.get_default_policy(),
                };
                match prepare_completion_route(
                    self.completion_input.as_ref(),
                    &self.completion_preparation,
                    policy.as_ref(),
                    request,
                    raw_body,
                )
                .await
                {
                    Ok(completion) => Some(completion),
                    Err(error) => return error.into_response(),
                }
            }
            None => None,
        };
        let text = match (pool, typed_req.grpc_request()) {
            (
                Some(crate::backend::WorkerPoolKind::Grpc),
                Some(GrpcRequestRef::Completion(completion)),
            ) if route == "/v1/completions" => completion.extract_text_for_grpc_routing(),
            _ => typed_req.extract_text_for_routing(),
        };
        let text = &text;
        let prepared = &prepared;
        let completion = completion.as_ref();
        let cache_key = std::sync::OnceLock::new();
        let cache_key = &cache_key;

        let http_inputs = tokio::sync::OnceCell::new();

        // KV-aware retries may bypass a target without rewriting its binding.
        let failed_workers = Mutex::new(Vec::new());
        let (lora_name, cache_salt) = typed_req.cache_identity();
        let affinity_key = self
            .kv_aware_session_affinity_ttl_secs
            .filter(|_| kv_aware_ops(self.policy_for_model(model_id).as_ref()).is_some())
            .and_then(|_| {
                resolve_affinity_identity(headers, typed_req.affinity_identity())
                    .ok()
                    .flatten()
                    .map(|identity| {
                        AffinityKey::from_identity(
                            identity,
                            model_id
                                .or_else(|| typed_req.get_model())
                                .unwrap_or("default"),
                        )
                    })
            });

        let http_inputs = &http_inputs;
        let failed_workers = &failed_workers;
        let affinity_key = &affinity_key;
        let response = RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            // operation per attempt
            |attempt: u32| async move {
                // Each backend attempt is a fresh scheduling arrival. The
                // previous ProgramCompletion finishes exactly once before a
                // retry invokes this closure again.
                let program_completion = match self
                    .acquire_program_completion(headers, typed_req, model_id, route)
                    .await
                {
                    Ok(completion) => completion,
                    Err(error) => return Self::schedule_error_response(error),
                };
                let forced_worker_url = program_completion
                    .as_ref()
                    .map(|completion| completion.dispatch().target_id.as_str());
                let policy = self.policy_for_model(model_id);
                let kv_ops = kv_aware_ops(policy.as_ref());
                let workers = self.available_workers(model_id);
                let excluded_workers = failed_workers.lock().clone();
                let inputs = if prepared.is_none()
                    && !completion.is_some_and(|completion| completion.input_active)
                    && kv_ops.is_some()
                    && !(excluded_workers.is_empty()
                        && affinity_key.as_ref().is_some_and(|key| {
                            kv_ops.is_some_and(|ops| {
                                ops.has_affinity(&workers, lora_name, cache_salt, key)
                            })
                        })) {
                    http_inputs
                        .get_or_init(|| async {
                            match serde_json::to_value(typed_req) {
                                Ok(body) => {
                                    prepare_http_kv_inputs(
                                        &self.client,
                                        &workers,
                                        headers,
                                        &body,
                                        route,
                                        self.api_key.as_deref(),
                                        Duration::from_secs(1),
                                    )
                                    .await
                                }
                                Err(_) => None,
                            }
                        })
                        .await
                        .as_ref()
                } else {
                    None
                };
                let request_tokens = if let Some(prepared) = &prepared {
                    if kv_ops.is_some() {
                        RouterMetrics::record_kv_routing_input("grpc", "prepared");
                    }
                    KvRequestTokens::Shared(Some(prepared.tokenized.token_ids.as_ref()))
                } else if let Some(completion) = completion.filter(|input| input.input_active) {
                    KvRequestTokens::Shared(completion.tokens.as_deref())
                } else {
                    inputs
                        .map(KvRequestTokens::PerWorker)
                        .unwrap_or(KvRequestTokens::Shared(None))
                };
                let eligibility = completion
                    .filter(|input| input.input_active)
                    .and_then(|_| {
                        placement_eligibility(
                            self.completion_input.as_ref(),
                            self.kv_index_supervisor.as_deref(),
                            &workers,
                        )
                    });
                let selected_worker = if let Some(target) = forced_worker_url {
                    let worker = self
                        .worker_registry
                        .get_by_url(target)
                        .filter(|worker| worker.is_available());
                    let Some(worker) = worker else {
                        return Self::program_target_unavailable_response(route);
                    };
                    if model_id.is_some_and(|model| {
                        worker.model_id() != model && worker.model_id() != UNLABELED_WORKER_MODEL_ID
                    }) {
                        return Self::program_target_unavailable_response(route);
                    }
                    let eligibility = completion
                        .filter(|input| input.input_active)
                        .and_then(|_| {
                            placement_eligibility(
                                self.completion_input.as_ref(),
                                self.kv_index_supervisor.as_deref(),
                                std::slice::from_ref(&worker),
                            )
                        });
                    let kv_receipt = kv_aware_policy(policy.as_ref()).and_then(|kv_aware| {
                        kv_aware.select_worker_with_qualified_inputs(
                            std::slice::from_ref(&worker),
                            QualifiedKvInput {
                                tokens: request_tokens,
                                eligibility: eligibility.as_deref(),
                            },
                            lora_name,
                            cache_salt,
                            affinity_key.as_ref(),
                            &[],
                        )
                    });
                    Some(SelectedWorker { worker, kv_receipt })
                } else {
                    self.select_worker_for_model(
                        model_id,
                        &workers,
                        Some(if policy.needs_cache_routing_key() {
                            cache_key.get_or_init(|| typed_req.extract_cache_routing_key())
                        } else {
                            text
                        }),
                        headers,
                        request_tokens,
                        eligibility.as_deref(),
                        lora_name,
                        cache_salt,
                        affinity_key.as_ref(),
                        &excluded_workers,
                    )
                };
                let SelectedWorker { worker, kv_receipt } = match selected_worker {
                    Some(selection) => selection,
                    None => {
                        RouterMetrics::record_request_error(route, "no_available_workers");
                        return (
                            StatusCode::SERVICE_UNAVAILABLE,
                            "No available workers (all circuits open or unhealthy)",
                        )
                            .into_response();
                    }
                };

                if let Some(receipt) = &kv_receipt {
                    let input_tokens = match request_tokens {
                        KvRequestTokens::Shared(ids) => ids.map(<[u32]>::len),
                        KvRequestTokens::PerWorker(_) => None,
                    };
                    debug!(attempt, worker_url = worker.url(), target = %receipt.decision.target.instance_id, rank = receipt.decision.target.dp_rank, new_prefill_tokens = receipt.new_prefill_tokens, input_known = input_tokens.is_some(), input_tokens, "KV request selected");
                }
                let load = RequestLoadGuard::from_selection(
                    worker.clone(),
                    kv_receipt,
                    policy.tracks_request_load() || program_completion.is_some(),
                );

                let response = self
                    .send_typed_request(
                        typed_req,
                        TypedDispatch {
                            headers,
                            route,
                            worker_url: worker.url(),
                            is_stream,
                            completion,
                            load,
                            prepared: (*prepared).clone(),
                        },
                        program_completion.clone(),
                    )
                    .await;

                // Client errors (4xx) are not worker failures - only server errors (5xx)
                // should count against the circuit breaker.
                let status = response.status();
                let was_available = worker.is_available();
                worker.record_outcome(status.is_success() || status.is_client_error());
                if was_available != worker.is_available() {
                    self.worker_registry.notify_worker_state_change();
                }
                if is_retryable_status(status) && kv_ops.is_some() {
                    failed_workers.lock().push(worker.url().to_string());
                }

                response
            },
            // should_retry predicate
            |res, _attempt| is_retryable_status(res.status()),
            // on_backoff hook
            |delay, attempt| {
                RouterMetrics::record_retry(route);
                RouterMetrics::record_retry_backoff_duration(delay, attempt);
            },
            // on_exhausted hook
            || RouterMetrics::record_retries_exhausted(route),
        )
        .await;

        if response.status().is_success() {
            let duration = start.elapsed();
            RouterMetrics::record_request(route);
            RouterMetrics::record_generate_duration(duration);
        } else if !is_retryable_status(response.status()) {
            RouterMetrics::record_request_error(route, "non_retryable_error");
        }

        response
    }

    // Helper: return base worker URL (strips DP suffix when enabled)
    fn worker_base_url(&self, worker_url: &str) -> String {
        if self.intra_node_data_parallel_size > 1 {
            if let Ok((prefix, _)) = dp_utils::extract_dp_rank(worker_url) {
                return prefix.to_string();
            }
        }
        worker_url.to_string()
    }

    // Generic simple routing for GET/POST without JSON body
    async fn route_simple_request(
        &self,
        headers: Option<&HeaderMap>,
        endpoint: &str,
        method: Method,
    ) -> Response {
        // TODO: currently the vllm worker is using in-memory state management, so this implementation has to fan out to all workers.
        // Eventually, we need to have router to manage the chat history with a proper database, will update this implementation accordingly.
        let worker_urls = self.get_worker_urls();
        if worker_urls.is_empty() {
            return (StatusCode::SERVICE_UNAVAILABLE, "No available workers").into_response();
        }

        let mut last_response: Option<Response> = None;
        for worker_url in worker_urls {
            let base = self.worker_base_url(&worker_url);

            let url = format!("{}/{}", base, endpoint);
            let route_name = format!("/{}", endpoint);
            let method_name = method.as_str().to_string();
            let mut request_builder = match method.clone() {
                Method::GET => self.client.get(&url),
                Method::POST => self.client.post(&url),
                _ => {
                    return (
                        StatusCode::METHOD_NOT_ALLOWED,
                        "Unsupported method for simple routing",
                    )
                        .into_response()
                }
            };

            if let Some(hdrs) = headers {
                let forwarded = header_utils::sanitize_request_headers(hdrs);
                for (name, value) in &forwarded {
                    let name_lc = name.as_str().to_lowercase();
                    if name_lc != "content-type"
                        && name_lc != "content-length"
                        && !header_utils::TRACE_HEADER_NAMES.contains(&name_lc.as_str())
                    {
                        request_builder = request_builder.header(name, value);
                    }
                }
            }

            match otel_http::send_client_request(
                request_builder,
                headers,
                ClientRequestOptions {
                    method: &method_name,
                    url: &url,
                    route: Some(&route_name),
                    request_phase: None,
                },
            )
            .await
            {
                Ok(res) => {
                    let status = StatusCode::from_u16(res.status().as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    let response_headers = header_utils::preserve_response_headers(res.headers());
                    match res.bytes().await {
                        Ok(body) => {
                            let mut response = Response::new(axum::body::Body::from(body));
                            *response.status_mut() = status;
                            *response.headers_mut() = response_headers;
                            if status.is_success() {
                                return response;
                            }
                            last_response = Some(response);
                        }
                        Err(e) => {
                            last_response = Some(
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    format!("Failed to read response: {}", e),
                                )
                                    .into_response(),
                            );
                        }
                    }
                }
                Err(e) => {
                    last_response = Some(
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Request failed: {}", e),
                        )
                            .into_response(),
                    );
                }
            }
        }

        last_response
            .unwrap_or_else(|| (StatusCode::BAD_GATEWAY, "No worker response").into_response())
    }

    // Route a GET request with provided headers to a specific endpoint
    async fn route_get_request(&self, headers: Option<&HeaderMap>, endpoint: &str) -> Response {
        self.route_simple_request(headers, endpoint, Method::GET)
            .await
    }

    // Route a POST request with empty body to a specific endpoint
    async fn route_post_empty_request(
        &self,
        headers: Option<&HeaderMap>,
        endpoint: &str,
    ) -> Response {
        self.route_simple_request(headers, endpoint, Method::POST)
            .await
    }

    // Send typed request directly without conversion
    #[allow(clippy::too_many_arguments)]
    async fn send_typed_request<T: serde::Serialize>(
        &self,
        typed_req: &T,
        dispatch: TypedDispatch<'_>,
        program_completion: Option<ProgramCompletion>,
    ) -> Response {
        let TypedDispatch {
            headers,
            route,
            worker_url,
            is_stream,
            completion,
            load,
            prepared,
        } = dispatch;
        let completion_echo = completion.map(|input| input.echo);
        let load_incremented = load.tracks_worker();
        if crate::backend::is_grpc_url(worker_url) {
            // gRPC workers currently support only the chat and completions
            // endpoints. Treat all other routes as client errors so they do not
            // count against healthy workers in circuit-breaker accounting.
            if !matches!(route, "/v1/chat/completions" | "/v1/completions") {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("gRPC backend currently supports /v1/chat/completions and /v1/completions, not {route}"),
                )
                    .into_response();
            }
            let Some(prepared) = prepared else {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "gRPC chat requires Frontend.prepare before dispatch",
                )
                    .into_response();
            };
            let on_first_output = if is_stream {
                None
            } else {
                let load = load.clone();
                Some(Arc::new(move || load.begin_decode())
                    as crate::backend::grpc::FirstOutputCallback)
            };
            let mut response = self
                .frontend
                .dispatch_with_first_output(worker_url, prepared, on_first_output)
                .await;
            if is_stream && response.status().is_success() {
                response = hold_request_load_until_body_done(response, load);
            }
            if let Some(completion) = &program_completion {
                completion.finish(response.status().is_success());
            }
            return response;
        }

        let (mut request_builder, extracted_dp_rank, request_url) =
            if self.intra_node_data_parallel_size > 1 {
                let (worker_url_prefix, dp_rank) = match dp_utils::extract_dp_rank(worker_url) {
                    Ok(tup) => tup,
                    Err(e) => {
                        error!("Failed to extract dp_rank: {}", e);
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Failed to extract dp_rank: {}", e),
                        )
                            .into_response();
                    }
                };

                // Parse the request body
                let json_val = match serde_json::to_value(typed_req) {
                    Ok(j) => j,
                    Err(e) => {
                        return (
                            StatusCode::BAD_REQUEST,
                            format!("Convert into serde_json::Value failed: {}", e),
                        )
                            .into_response();
                    }
                };

                // Use the original json_val without modification

                let request_url = format!("{}{}", worker_url_prefix, route);
                (
                    self.client.post(&request_url).json(&json_val),
                    Some(dp_rank),
                    request_url,
                )
            } else {
                let request_url = format!("{}{}", worker_url, route);
                (
                    if let Some(completion) = completion {
                        completion.forward_request(&self.client, &request_url, headers, typed_req)
                    } else {
                        self.client.post(&request_url).json(typed_req)
                    },
                    None,
                    request_url,
                )
            };

        // Copy all headers from original request if provided, skipping
        // Content-Type/Content-Length (.json() sets them) and trace headers
        // (propagate_trace_headers below injects fresh context).
        if let Some(headers) = headers {
            let forwarded = header_utils::sanitize_request_headers(headers);
            for (name, value) in &forwarded {
                if *name != CONTENT_TYPE
                    && *name != CONTENT_LENGTH
                    && !header_utils::TRACE_HEADER_NAMES
                        .iter()
                        .any(|&th| name.as_str().eq_ignore_ascii_case(th))
                {
                    request_builder = request_builder.header(name, value);
                }
            }
        }

        // Add X-data-parallel-rank header for DP-aware routing
        if let Some(dp_rank) = extracted_dp_rank {
            request_builder = request_builder.header("X-data-parallel-rank", dp_rank.to_string());
        }

        // Opt-in: VLLM_ROUTER_STAGES=1. HTTP remains a transparent proxy; the
        // stage split reports worker header wait and first SSE byte only.
        let stages_on = crate::backend::stages_enabled();

        let t_send = Instant::now();
        let res = match otel_http::send_client_request(
            request_builder,
            headers,
            ClientRequestOptions {
                method: "POST",
                url: &request_url,
                route: Some(route),
                request_phase: Some("inference"),
            },
        )
        .await
        {
            Ok(res) => res,
            Err(e) => {
                error!(
                    "Failed to send typed request worker_url={} route={} error={}",
                    worker_url, route, e
                );

                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Request failed: {}", e),
                )
                    .into_response();
            }
        };

        let status = StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let http_ttfb_ms = t_send.elapsed().as_secs_f64() * 1000.0;

        if !is_stream {
            // For non-streaming requests, preserve headers
            let mut response_headers = header_utils::preserve_response_headers(res.headers());
            let first_ms = t_send.elapsed().as_secs_f64() * 1000.0;
            if stages_on {
                let http_stages = http_stages_json(http_ttfb_ms, first_ms);
                insert_router_stages(&mut response_headers, &http_stages);
                info!(stages = %http_stages, "http proxy stages");
            }

            let response = match res.bytes().await {
                Ok(body) => {
                    if let Some(completion) = &program_completion {
                        completion.observe_json(&body);
                    }
                    let mut response = Response::new(axum::body::Body::from(body));
                    *response.status_mut() = status;
                    *response.headers_mut() = response_headers;
                    response
                }
                Err(e) => {
                    let error_msg = format!("Failed to get response body: {}", e);
                    (StatusCode::INTERNAL_SERVER_ERROR, error_msg).into_response()
                }
            };

            if let Some(completion) = &program_completion {
                completion.finish(response.status().is_success());
            }

            response
        } else if load_incremented {
            let mut completion_output = completion.map(|input| input.output_observer(status));
            // The forwarding task and response body share one cleanup receipt.
            let completion = if status.is_success() {
                program_completion
            } else {
                None
            };

            // Preserve headers for streaming response
            let mut response_headers = header_utils::preserve_response_headers(res.headers());
            // Ensure we set the correct content-type for SSE
            response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
            if stages_on {
                let header_stages = http_stages_json(http_ttfb_ms, http_ttfb_ms);
                insert_router_stages(&mut response_headers, &header_stages);
            }

            let stream = res.bytes_stream();
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            let (mut producer_output_frame, output_frame) = if completion_echo.is_some() {
                let (producer, consumer) = output_frame_tracking();
                (Some(producer), Some(consumer))
            } else {
                (None, None)
            };

            // Spawn task to forward stream and detect completion
            let producer_load = load.clone();
            let producer = tokio::spawn(async move {
                let mut stream = stream;
                let mut first_sse = true;
                let mut stream_succeeded = true;
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(bytes) => {
                            if completion_echo.is_none() {
                                producer_load.begin_decode();
                            }
                            if first_sse {
                                first_sse = false;
                                let first_ms = t_send.elapsed().as_secs_f64() * 1000.0;
                                if stages_on {
                                    let stages = http_stages_json(http_ttfb_ms, first_ms);
                                    if !emit_http_first_sse(
                                        &tx,
                                        &stages,
                                        producer_output_frame.as_mut(),
                                    ) && completion_echo.is_some()
                                    {
                                        stream_succeeded = false;
                                        break;
                                    }
                                }
                            }
                            if let Some(completion) = &completion {
                                completion.observe_sse_chunk(&bytes);
                            }
                            // Check for stream end marker
                            if completion_echo.is_none()
                                && bytes
                                    .as_ref()
                                    .windows(12)
                                    .any(|window| window == b"data: [DONE]")
                            {
                                producer_load.release();
                            }
                            let has_output = completion_output
                                .as_mut()
                                .is_some_and(|observer| observer.observe(&bytes));
                            let sent = match producer_output_frame.as_mut() {
                                Some(output_frame) => output_frame.send(&tx, bytes, has_output),
                                None => tx.send(Ok(bytes)).is_ok(),
                            };
                            if !sent {
                                stream_succeeded = false;
                                break;
                            }
                            if has_output {
                                producer_load.begin_decode();
                            }
                        }
                        Err(e) => {
                            stream_succeeded = false;
                            let _ = tx.send(Err(format!("Stream error: {}", e)));
                            break;
                        }
                    }
                }
                if let Some(completion) = &completion {
                    completion.finish(stream_succeeded);
                }
            });

            let stream = UnboundedReceiverStream::new(rx);
            let body = Body::from_stream(stream);

            let mut response = Response::new(body);
            *response.status_mut() = status;
            *response.headers_mut() = response_headers;
            hold_request_load_with_output_frame(response, load, Some(producer), output_frame)
        } else {
            // For requests without load tracking, just stream
            // Preserve headers for streaming response
            let mut response_headers = header_utils::preserve_response_headers(res.headers());
            // Ensure we set the correct content-type for SSE
            response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
            if stages_on {
                let header_stages = http_stages_json(http_ttfb_ms, http_ttfb_ms);
                insert_router_stages(&mut response_headers, &header_stages);
            }

            let stream = res.bytes_stream();
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            let completion = if status.is_success() {
                program_completion
            } else {
                None
            };

            // Spawn task to forward stream
            tokio::spawn(async move {
                let mut stream = stream;
                let mut first_sse = true;
                let mut stream_succeeded = true;
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(bytes) => {
                            if first_sse {
                                first_sse = false;
                                let first_ms = t_send.elapsed().as_secs_f64() * 1000.0;
                                if stages_on {
                                    let stages = http_stages_json(http_ttfb_ms, first_ms);
                                    emit_http_first_sse(&tx, &stages, None);
                                }
                            }
                            if let Some(completion) = &completion {
                                completion.observe_sse_chunk(&bytes);
                            }
                            if tx.send(Ok(bytes)).is_err() {
                                stream_succeeded = false;
                                break;
                            }
                        }
                        Err(e) => {
                            stream_succeeded = false;
                            let _ = tx.send(Err(format!("Stream error: {}", e)));
                            break;
                        }
                    }
                }
                if let Some(completion) = &completion {
                    completion.finish(stream_succeeded);
                }
            });

            let stream = UnboundedReceiverStream::new(rx);
            let body = Body::from_stream(stream);

            let mut response = Response::new(body);
            *response.status_mut() = status;
            *response.headers_mut() = response_headers;
            response
        }
    }

    pub async fn add_worker(&self, worker_url: &str) -> Result<String, String> {
        let mut urls = self.get_worker_urls();
        urls.push(worker_url.to_string());
        crate::backend::classify_worker_urls(&urls)?;
        if let Some(supervisor) = &self.kv_index_supervisor {
            supervisor.validate_http_worker(worker_url, self.intra_node_data_parallel_size)?;
        }

        let start_time = std::time::Instant::now();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(self.worker_startup_timeout_secs))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        loop {
            if start_time.elapsed() > Duration::from_secs(self.worker_startup_timeout_secs) {
                error!(
                    "Timeout {}s waiting for worker {} to become healthy. Please set --router-worker-startup-timeout-secs (vllm_router.launch_server) or --worker-startup-timeout-secs (vllm_worker.router) to a larger value",
                    self.worker_startup_timeout_secs, worker_url
                );
                return Err(format!(
                    "Timeout {}s waiting for worker {} to become healthy. Please set --router-worker-startup-timeout-secs (vllm_router.launch_server) or --worker-startup-timeout-secs (vllm_worker.router) to a larger value",
                    self.worker_startup_timeout_secs, worker_url
                ));
            }

            let health_result = if crate::backend::is_grpc_url(worker_url) {
                crate::backend::check_grpc_health(worker_url, Duration::from_secs(2)).await
            } else {
                match client.get(format!("{}/health", worker_url)).send().await {
                    Ok(res) if res.status().is_success() => Ok(()),
                    Ok(res) => Err(format!("HTTP health status {}", res.status())),
                    Err(error) => Err(error.to_string()),
                }
            };

            match health_result {
                Ok(()) => {
                    if self.intra_node_data_parallel_size > 1 {
                        // Expand worker URL into multiple DP-aware URLs based on configured intra_node_data_parallel_size
                        // (e.g., "http://host:8000" → "http://host:8000@0", "@1", etc.)
                        // without querying the worker
                        let url_vec = vec![String::from(worker_url)];
                        let dp_url_vec = dp_utils::get_dp_aware_workers(
                            &url_vec,
                            &self.api_key,
                            self.intra_node_data_parallel_size,
                        )
                        .await
                        .map_err(|e| format!("Failed to get dp-aware workers: {}", e))?;
                        let mut worker_added: bool = false;
                        let mut kv_discovery_attempted = false;
                        let mut kv_info = None;
                        for dp_url in &dp_url_vec {
                            if self.worker_registry.get_by_url(dp_url).is_some() {
                                warn!("Worker {} already exists", dp_url);
                                continue;
                            }
                            if !kv_discovery_attempted {
                                kv_info = self.discover_kv_worker(worker_url).await;
                                kv_discovery_attempted = true;
                            }
                            info!("Added worker: {}", dp_url);
                            // TODO: In IGW mode, fetch model_id from worker's /get_model_info endpoint
                            let (base_url, dp_rank) = dp_utils::parse_worker_url(dp_url);
                            let new_worker = DPAwareWorker::new(
                                base_url,
                                dp_rank.unwrap_or(0),
                                self.intra_node_data_parallel_size,
                                WorkerType::Regular,
                            )
                            .with_circuit_breaker_config(self.circuit_breaker_config.clone())
                            .with_health_config(self.health_config.clone());

                            let worker_arc: Arc<dyn Worker> = Arc::new(new_worker);
                            self.worker_registry.register(worker_arc.clone());
                            self.register_kv_target(worker_arc.as_ref(), kv_info.as_ref());

                            // Notify PolicyRegistry about the new worker
                            let model_id = worker_arc.model_id();
                            let policy = self.policy_registry.on_worker_added(model_id, None);

                            // If this is a cache-aware policy, update it with all workers for this model
                            if policy.needs_cache_routing_key() {
                                if let Some(cache_aware) = policy
                                    .as_any()
                                    .downcast_ref::<crate::policies::CacheAwarePolicy>(
                                ) {
                                    let model_workers =
                                        self.worker_registry.get_by_model_fast(model_id);
                                    cache_aware.init_workers(&model_workers);
                                }
                            }

                            worker_added = true;
                        }
                        if !worker_added {
                            return Err(format!("No worker added for {}", worker_url));
                        }
                    } else {
                        if self.worker_registry.get_by_url(worker_url).is_some() {
                            return Err(format!("Worker {} already exists", worker_url));
                        }
                        let kv_info = self.discover_kv_worker(worker_url).await;
                        info!("Added worker: {}", worker_url);

                        // TODO: In IGW mode, fetch model_id from worker's /get_model_info endpoint
                        let new_worker =
                            BasicWorker::new(worker_url.to_string(), WorkerType::Regular)
                                .with_circuit_breaker_config(self.circuit_breaker_config.clone())
                                .with_health_config(self.health_config.clone());

                        let worker_arc = Arc::new(new_worker);
                        self.worker_registry.register(worker_arc.clone());
                        self.register_kv_target(worker_arc.as_ref(), kv_info.as_ref());

                        // Notify PolicyRegistry about the new worker
                        let model_id = worker_arc.model_id();
                        let policy = self.policy_registry.on_worker_added(model_id, None);

                        // If this is a cache-aware policy, add this worker to it
                        if policy.needs_cache_routing_key() {
                            if let Some(cache_aware) = policy
                                .as_any()
                                .downcast_ref::<crate::policies::CacheAwarePolicy>(
                            ) {
                                // Get all workers for this model
                                let model_workers =
                                    self.worker_registry.get_by_model_fast(model_id);
                                cache_aware.init_workers(&model_workers);
                            }
                        }
                    }

                    RouterMetrics::set_active_workers(self.worker_registry.get_all().len());

                    return Ok(format!("Successfully added worker: {}", worker_url));
                }
                Err(e) => {
                    debug!("Worker {} health check pending - error: {}", worker_url, e);

                    if !worker_url.starts_with("http://")
                        && !worker_url.starts_with("https://")
                        && !crate::backend::is_grpc_url(worker_url)
                    {
                        warn!("The worker url {} does not have http or https prefix. Please add the prefix to the url.", worker_url);
                    }

                    tokio::time::sleep(Duration::from_secs(
                        self.worker_startup_check_interval_secs,
                    ))
                    .await;
                    continue;
                }
            }
        }
    }

    pub fn remove_worker(&self, worker_url: &str) {
        if self.intra_node_data_parallel_size > 1 {
            // remove dp-aware workers in a prefix-matching fashion
            // without contacting the remote worker
            let mut removed_workers: Vec<String> = Vec::new();
            let worker_url_prefix = format!("{}@", worker_url);

            // Find and remove all workers with matching prefix
            let all_workers = self.worker_registry.get_all();
            for w in all_workers.iter() {
                if w.url().starts_with(&worker_url_prefix) {
                    let removed_url = w.url().to_string();
                    // Get model_id before removing
                    let model_id = w.model_id().to_string();

                    if self.worker_registry.remove_by_url(&removed_url).is_some() {
                        self.frontend.remove_worker(&removed_url);
                        info!("Removed worker: {}", removed_url);
                        removed_workers.push(removed_url);

                        // Notify PolicyRegistry about the removed worker
                        self.policy_registry.on_worker_removed(&model_id);
                    } else {
                        warn!("Worker {} not found, skipping removal", w.url());
                    }
                }
            }

            RouterMetrics::set_active_workers(self.worker_registry.get_all().len());
            if !removed_workers.is_empty() {
                self.remove_kv_worker(worker_url, &removed_workers);
            }

            // If any models are using cache aware policy, remove the workers from the tree
            // Check each removed worker's model and get its policy
            for dp_url in removed_workers.iter() {
                if let Some(worker) = self.worker_registry.get_by_url(dp_url) {
                    let model_id = worker.model_id();
                    if let Some(policy) = self.policy_registry.get_policy(model_id) {
                        if let Some(cache_aware) = policy
                            .as_any()
                            .downcast_ref::<crate::policies::CacheAwarePolicy>()
                        {
                            cache_aware.remove_worker_by_url(dp_url);
                            info!("Removed worker from cache-aware tree: {}", dp_url);
                        }
                    }
                }
            }
        } else {
            // Get the worker first to extract model_id
            let model_id = if let Some(worker) = self.worker_registry.get_by_url(worker_url) {
                worker.model_id().to_string()
            } else {
                warn!("Worker {} not found, skipping removal", worker_url);
                return;
            };

            if self.worker_registry.remove_by_url(worker_url).is_some() {
                self.frontend.remove_worker(worker_url);
                info!("Removed worker: {}", worker_url);

                // Notify PolicyRegistry about the removed worker
                self.policy_registry.on_worker_removed(&model_id);

                RouterMetrics::set_active_workers(self.worker_registry.get_all().len());
                self.remove_kv_worker(worker_url, &[worker_url.to_string()]);
            }

            // If the model is using cache aware policy, remove the worker from the tree
            if let Some(policy) = self.policy_registry.get_policy(&model_id) {
                if let Some(cache_aware) = policy
                    .as_any()
                    .downcast_ref::<crate::policies::CacheAwarePolicy>()
                {
                    cache_aware.remove_worker_by_url(worker_url);
                    info!("Removed worker from cache-aware tree: {}", worker_url);
                }
            }
        }
    }

    async fn get_worker_load(&self, worker_url: &str) -> Option<isize> {
        let worker_url = if self.intra_node_data_parallel_size > 1 {
            // Need to extract the URL from "http://host:port@dp_rank"
            let (worker_url_prefix, _dp_rank) = match dp_utils::extract_dp_rank(worker_url) {
                Ok(tup) => tup,
                Err(e) => {
                    error!("Failed to extract dp_rank: {}", e);
                    return None;
                }
            };
            worker_url_prefix
        } else {
            worker_url
        };

        match self
            .client
            .get(format!("{}/get_load", worker_url))
            .send()
            .await
        {
            Ok(res) if res.status().is_success() => match res.bytes().await {
                Ok(bytes) => match serde_json::from_slice::<serde_json::Value>(&bytes) {
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

    // Background task to monitor worker loads
    async fn monitor_worker_loads(
        worker_urls: Vec<String>,
        tx: tokio::sync::watch::Sender<HashMap<String, isize>>,
        interval_secs: u64,
        policy: Arc<dyn LoadBalancingPolicy>,
        client: Client,
    ) {
        let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));

        loop {
            interval.tick().await;

            let mut loads = HashMap::new();
            for url in &worker_urls {
                if let Some(load) = Self::get_worker_load_static(&client, url).await {
                    loads.insert(url.clone(), load);
                }
            }

            if !loads.is_empty() {
                // Update policy with new loads
                policy.update_loads(&loads);

                // Send to watchers
                if let Err(e) = tx.send(loads) {
                    error!("Failed to send load update: {}", e);
                }
            }
        }
    }

    // Static version of get_worker_load for use in monitoring task
    async fn get_worker_load_static(client: &reqwest::Client, worker_url: &str) -> Option<isize> {
        let worker_url = if worker_url.contains("@") {
            // Need to extract the URL from "http://host:port@dp_rank"
            let (worker_url_prefix, _dp_rank) = match dp_utils::extract_dp_rank(worker_url) {
                Ok(tup) => tup,
                Err(e) => {
                    debug!("Failed to extract dp_rank: {}", e);
                    return None;
                }
            };
            worker_url_prefix
        } else {
            worker_url
        };

        match client.get(format!("{}/get_load", worker_url)).send().await {
            Ok(res) if res.status().is_success() => match res.bytes().await {
                Ok(bytes) => match serde_json::from_slice::<serde_json::Value>(&bytes) {
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

    async fn build_rerank_response(
        req: &RerankRequest,
        response: Response,
    ) -> anyhow::Result<Response> {
        let (_, response_body) = response.into_parts();
        let body_bytes = to_bytes(response_body, usize::MAX).await?;
        let rerank_results = serde_json::from_slice::<Vec<RerankResult>>(&body_bytes)?;
        let mut rerank_response =
            RerankResponse::new(rerank_results, req.model.clone(), req.rid.clone());
        rerank_response.sort_by_score();
        if let Some(top_k) = req.top_k {
            rerank_response.apply_top_k(top_k);
        }
        if !req.return_documents {
            rerank_response.drop_documents();
        }
        Ok(Json(rerank_response).into_response())
    }
}

use async_trait::async_trait;

#[async_trait]
impl WorkerManagement for Router {
    async fn add_worker(&self, worker_url: &str) -> Result<String, String> {
        Router::add_worker(self, worker_url).await
    }

    fn remove_worker(&self, worker_url: &str) {
        Router::remove_worker(self, worker_url)
    }

    fn get_worker_urls(&self) -> Vec<String> {
        Router::get_worker_urls(self)
    }
}

#[async_trait]
impl RouterTrait for Router {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn scheduling_diagnostics(&self) -> Option<serde_json::Value> {
        let scheduler = self.program_scheduler.as_ref()?;
        let mut diagnostics = serde_json::to_value(scheduler.diagnostics()).ok()?;
        diagnostics["token_estimation"] =
            serde_json::to_value(self.program_token_estimator.diagnostics()).ok()?;
        Some(diagnostics)
    }

    async fn health(&self, _req: Request<Body>) -> Response {
        let workers = self.worker_registry.get_all();
        let unhealthy_servers: Vec<_> = workers
            .iter()
            .filter(|w| !w.is_healthy())
            .map(|w| w.url().to_string())
            .collect();

        if unhealthy_servers.is_empty() {
            (StatusCode::OK, "All servers healthy").into_response()
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                format!("Unhealthy servers: {:?}", unhealthy_servers),
            )
                .into_response()
        }
    }

    async fn health_generate(&self, req: Request<Body>) -> Response {
        match self.select_first_worker() {
            Ok(worker_url) if crate::backend::is_grpc_url(&worker_url) => {
                // vLLM Rust gRPC exposes canonical grpc.health.v1 for
                // readiness. That is not equivalent to /health_generate,
                // which router HTTP/PD paths treat as a generation-capability
                // probe, so all-gRPC mode returns an explicit 501.
                return (
                    StatusCode::NOT_IMPLEMENTED,
                    "/health_generate is not implemented for gRPC workers; use /health for gRPC readiness",
                )
                    .into_response();
            }
            Ok(_) => {}
            Err(e) => return (StatusCode::SERVICE_UNAVAILABLE, e).into_response(),
        }
        self.proxy_get_request(req, "health_generate").await
    }

    async fn get_server_info(&self, req: Request<Body>) -> Response {
        match self.select_first_worker() {
            Ok(worker_url) if crate::backend::is_grpc_url(&worker_url) => {
                return match crate::backend::get_grpc_server_info(
                    &worker_url,
                    Duration::from_secs(2),
                )
                .await
                {
                    Ok(info) => (
                        StatusCode::OK,
                        Json(crate::backend::server_info_json(&info)),
                    )
                        .into_response(),
                    Err(e) => (StatusCode::BAD_GATEWAY, e).into_response(),
                };
            }
            Ok(_) => {}
            Err(e) => return (StatusCode::SERVICE_UNAVAILABLE, e).into_response(),
        }
        self.proxy_get_request(req, "get_server_info").await
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        match self.select_first_worker() {
            Ok(worker_url) if crate::backend::is_grpc_url(&worker_url) => {
                return match crate::backend::get_grpc_model_info(
                    &worker_url,
                    Duration::from_secs(2),
                )
                .await
                {
                    Ok(info) => (
                        StatusCode::OK,
                        Json(crate::backend::openai_models_json(&info)),
                    )
                        .into_response(),
                    Err(e) => (StatusCode::BAD_GATEWAY, e).into_response(),
                };
            }
            Ok(_) => {}
            Err(e) => return (StatusCode::SERVICE_UNAVAILABLE, e).into_response(),
        }
        self.proxy_get_request(req, "v1/models").await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        match self.select_first_worker() {
            Ok(worker_url) if crate::backend::is_grpc_url(&worker_url) => {
                return match crate::backend::get_grpc_model_info(
                    &worker_url,
                    Duration::from_secs(2),
                )
                .await
                {
                    Ok(info) => (StatusCode::OK, Json(crate::backend::model_info_json(&info)))
                        .into_response(),
                    Err(e) => (StatusCode::BAD_GATEWAY, e).into_response(),
                };
            }
            Ok(_) => {}
            Err(e) => return (StatusCode::SERVICE_UNAVAILABLE, e).into_response(),
        }
        self.proxy_get_request(req, "get_model_info").await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/generate", model_id)
            .await
    }

    async fn route_inference_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &InferenceGenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/inference/v1/generate", model_id)
            .await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/chat/completions", model_id)
            .await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: Arc<CompletionRequest>,
        model_id: Option<&str>,
    ) -> Response {
        self.route_prepared_request(
            headers,
            body.as_ref(),
            "/v1/completions",
            model_id,
            Some(body.clone()),
            None,
        )
        .await
    }

    async fn route_completion_raw(
        &self,
        headers: Option<&HeaderMap>,
        body: Arc<CompletionRequest>,
        raw: &axum::body::Bytes,
        model_id: Option<&str>,
    ) -> Response {
        // DP rewrites and non-KV routes preserve their pre-existing typed wire.
        let policy = match model_id {
            Some(model) => self.policy_registry.get_policy_or_default(model),
            None => self.policy_registry.get_default_policy(),
        };
        let preserve_raw =
            self.intra_node_data_parallel_size == 1 && kv_aware_ops(policy.as_ref()).is_some();
        self.route_prepared_request(
            headers,
            body.as_ref(),
            "/v1/completions",
            model_id,
            Some(body.clone()),
            preserve_raw.then_some(raw),
        )
        .await
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/responses", model_id)
            .await
    }

    async fn get_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let endpoint = format!("v1/responses/{}", response_id);
        self.route_get_request(headers, &endpoint).await
    }

    async fn cancel_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let endpoint = format!("v1/responses/{}/cancel", response_id);
        self.route_post_empty_request(headers, &endpoint).await
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Record embeddings-specific metrics in addition to general request metrics
        let start = Instant::now();
        let res = self
            .route_typed_request(headers, body, "/v1/embeddings", model_id)
            .await;

        // Embedding specific metrics
        if res.status().is_success() {
            RouterMetrics::record_embeddings_request();
            RouterMetrics::record_embeddings_duration(start.elapsed());
        } else {
            let error_type = format!("http_{}", res.status().as_u16());
            RouterMetrics::record_embeddings_error(&error_type);
        }

        res
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: Option<&str>,
    ) -> Response {
        if let Err(e) = body.validate() {
            return (StatusCode::BAD_REQUEST, e).into_response();
        }
        let response = self
            .route_typed_request(headers, body, "/v1/rerank", model_id)
            .await;
        if response.status().is_success() {
            match Self::build_rerank_response(body, response).await {
                Ok(rerank_response) => rerank_response,
                Err(e) => {
                    error!("Failed to build rerank response: {}", e);
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Failed to build rerank response".to_string(),
                    )
                        .into_response();
                }
            }
        } else {
            response
        }
    }

    async fn flush_cache(&self) -> Response {
        // Get all worker URLs
        let worker_urls = self.get_worker_urls();

        // Send requests to all workers concurrently without headers
        let mut tasks = Vec::new();
        for worker_url in &worker_urls {
            let worker_url = if self.intra_node_data_parallel_size > 1 {
                // Need to extract the URL from "http://host:port@dp_rank"
                let (worker_url_prefix, _dp_rank) = match dp_utils::extract_dp_rank(worker_url) {
                    Ok(tup) => tup,
                    Err(e) => {
                        error!("Failed to extract dp_rank: {}", e);
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Failed to extract dp_rank: {}", e),
                        )
                            .into_response();
                    }
                };
                worker_url_prefix
            } else {
                worker_url
            };
            let request_builder = self.client.post(format!("{}/flush_cache", worker_url));
            tasks.push(request_builder.send());
        }

        // Wait for all responses
        let results = futures_util::future::join_all(tasks).await;

        // Check if all succeeded
        let all_success = results.iter().all(|r| {
            r.as_ref()
                .map(|res| res.status().is_success())
                .unwrap_or(false)
        });

        if all_success {
            (StatusCode::OK, "Cache flushed on all servers").into_response()
        } else {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Cache flush failed on one or more servers",
            )
                .into_response()
        }
    }

    async fn get_worker_loads(&self) -> Response {
        let urls = self.get_worker_urls();
        let mut loads = Vec::new();

        // Get loads from all workers
        for url in &urls {
            let load = self.get_worker_load(url).await.unwrap_or(-1);
            loads.push(serde_json::json!({
                "worker": url,
                "load": load
            }));
        }

        Json(serde_json::json!({
            "workers": loads
        }))
        .into_response()
    }

    fn router_type(&self) -> &'static str {
        "regular"
    }

    fn readiness(&self) -> Response {
        // Regular router is ready if it has at least one healthy worker
        let workers = self.worker_registry.get_all();
        let healthy_count = workers.iter().filter(|w| w.is_healthy()).count();
        let total_workers = workers.len();

        if healthy_count > 0 {
            Json(serde_json::json!({
                "status": "ready",
                "healthy_workers": healthy_count,
                "total_workers": total_workers
            }))
            .into_response()
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "status": "not_ready",
                    "reason": "no healthy workers available",
                    "total_workers": total_workers
                })),
            )
                .into_response()
        }
    }

    /// Route a transparent proxy request to a backend worker
    /// Forwards the request as-is to a selected worker
    async fn route_transparent(
        &self,
        headers: Option<&HeaderMap>,
        path: &str,
        method: &Method,
        body: serde_json::Value,
    ) -> Response {
        debug!("Transparent proxy: routing {} {} to backend", method, path);
        let start = Instant::now();

        // Select a worker (filter by availability like select_worker_for_model)
        let model_id = body.get("model").and_then(serde_json::Value::as_str);
        let routing_model = model_id.filter(|model| self.worker_registry.has_model(model));
        let workers = self.available_workers(routing_model);
        if workers.is_empty() {
            RouterMetrics::record_request_error(path, "no_available_workers");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "No available workers".to_string(),
            )
                .into_response();
        }

        let policy = self.policy_for_model(routing_model);
        let kv_ops = kv_aware_ops(policy.as_ref());
        // Serializing the whole body is expensive when it carries base64
        // media; only consumers that actually read the text pay for it.
        let request_text = if (policy.needs_request_text() && !policy.needs_cache_routing_key())
            || self.program_scheduler.is_some()
        {
            serde_json::to_string(&body).ok()
        } else {
            None
        };
        let is_stream = body
            .get("stream")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let lora_name = body
            .get("lora_name")
            .and_then(serde_json::Value::as_str)
            .filter(|name| !name.is_empty());
        let cache_salt = body
            .get("cache_salt")
            .and_then(serde_json::Value::as_str)
            .filter(|salt| !salt.is_empty());
        let affinity_key = self
            .kv_aware_session_affinity_ttl_secs
            .filter(|_| kv_ops.is_some())
            .and_then(|_| {
                resolve_affinity_identity(headers, IdentityFields::from_body(&body))
                    .ok()
                    .flatten()
                    .map(|identity| {
                        AffinityKey::from_identity(identity, model_id.unwrap_or("default"))
                    })
            });
        let http_inputs = if *method == Method::POST
            && kv_ops.is_some()
            && !affinity_key.as_ref().is_some_and(|key| {
                kv_ops.is_some_and(|ops| ops.has_affinity(&workers, lora_name, cache_salt, key))
            }) {
            prepare_http_kv_inputs(
                &self.client,
                &workers,
                headers,
                &body,
                path,
                self.api_key.as_deref(),
                Duration::from_secs(1),
            )
            .await
        } else {
            None
        };
        let request_tokens = http_inputs
            .as_ref()
            .map(KvRequestTokens::PerWorker)
            .unwrap_or(KvRequestTokens::Shared(None));
        let program_completion = if *method == Method::POST {
            match self
                .acquire_program_completion_from_payload(
                    headers,
                    Some(&body),
                    model_id,
                    path,
                    request_text.as_deref().unwrap_or_default(),
                )
                .await
            {
                Ok(completion) => completion,
                Err(error) => {
                    RouterMetrics::record_request_error(path, "program_scheduling_error");
                    return Self::schedule_error_response(error);
                }
            }
        } else {
            None
        };
        let selection = if let Some(completion) = &program_completion {
            self.worker_registry
                .get_by_url(&completion.dispatch().target_id)
                .filter(|worker| worker.is_available())
                .map(|worker| {
                    let kv_receipt = kv_ops.and_then(|ops| {
                        ops.select_worker_with_inputs(
                            std::slice::from_ref(&worker),
                            request_tokens,
                            lora_name,
                            cache_salt,
                            affinity_key.as_ref(),
                            &[],
                        )
                    });
                    SelectedWorker { worker, kv_receipt }
                })
        } else if let Some(ops) = kv_ops {
            ops.select_worker_with_inputs(
                &workers,
                request_tokens,
                lora_name,
                cache_salt,
                affinity_key.as_ref(),
                &[],
            )
            .map(|receipt| SelectedWorker {
                worker: workers[receipt.worker_index].clone(),
                kv_receipt: Some(receipt),
            })
        } else {
            let cache_key = (policy.needs_cache_routing_key())
                .then(|| crate::protocols::spec::cache_routing::for_route(&body, path));
            let request_headers = Self::headers_to_request_headers(headers);
            policy
                .select_worker_with_headers(
                    &workers,
                    cache_key.as_deref().or(request_text.as_deref()),
                    request_headers.as_ref(),
                )
                .and_then(|index| workers.get(index).cloned())
                .map(|worker| SelectedWorker {
                    worker,
                    kv_receipt: None,
                })
        };
        let Some(SelectedWorker { worker, kv_receipt }) = selection else {
            if let Some(completion) = &program_completion {
                completion.finish(false);
            }
            RouterMetrics::record_request_error(path, "no_available_workers");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "Failed to select the Program target".to_string(),
            )
                .into_response();
        };
        let url = worker.endpoint_url(path);

        let load_tracked = policy.tracks_request_load() || program_completion.is_some();
        let load = load_tracked
            .then(|| RequestLoadGuard::from_selection(worker.clone(), kv_receipt, true));

        let (body, mut transfers) = if let Some(epd) = &self.epd {
            if *method == Method::POST && path == "/v1/chat/completions" {
                match epd
                    .prepare(
                        &self.client,
                        body,
                        worker.url(),
                        headers,
                        self.api_key.as_deref(),
                    )
                    .await
                {
                    Ok(value) => value,
                    Err(error) => {
                        RouterMetrics::record_request_error(path, "encoder_error");
                        return error.into_response();
                    }
                }
            } else {
                (body, None)
            }
        } else {
            (body, None)
        };

        debug!("Transparent proxy: forwarding to {}", url);

        // Build the request
        let mut request_builder = match *method {
            Method::GET => self.client.get(&url),
            Method::POST => self.client.post(&url),
            Method::PUT => self.client.put(&url),
            Method::DELETE => self.client.delete(&url),
            Method::PATCH => self.client.patch(&url),
            Method::HEAD => self.client.head(&url),
            _ => {
                RouterMetrics::record_request_error(path, "unsupported_method");
                return (
                    StatusCode::METHOD_NOT_ALLOWED,
                    format!("Method {} not supported", method),
                )
                    .into_response();
            }
        };

        // Add X-data-parallel-rank header for DP-aware routing
        request_builder = dp_utils::add_dp_rank_header(request_builder, worker.dp_rank());

        // Add JSON body if not null/empty
        if !body.is_null() {
            request_builder = request_builder.json(&body);
        }

        // Add authorization if configured
        if self.epd.is_some() {
            if let Some(headers) = headers {
                // The same connection-aware filter the other forwarding paths
                // use, so a field named by `Connection` or any other hop-by-hop
                // field cannot reach the worker here either.
                let sanitized = header_utils::sanitize_request_headers(headers);
                for (name, value) in &sanitized {
                    // When a backend api_key is configured it must win:
                    // RequestBuilder::header appends, and vLLM reads the
                    // first Authorization value.
                    if self.api_key.is_some() && name == http::header::AUTHORIZATION {
                        continue;
                    }
                    // reqwest derives these from the body and URL it sends.
                    if name == http::header::CONTENT_LENGTH
                        || name == http::header::CONTENT_TYPE
                        || header_utils::TRACE_HEADER_NAMES.contains(&name.as_str())
                    {
                        continue;
                    }
                    request_builder = request_builder.header(name, value);
                }
            }
        }
        if let Some(ref key) = self.api_key {
            request_builder = request_builder.header("Authorization", format!("Bearer {}", key));
        }

        // Send request
        let response = match otel_http::send_client_request(
            request_builder,
            headers,
            ClientRequestOptions {
                method: method.as_str(),
                url: &url,
                route: Some(path),
                request_phase: Some("inference"),
            },
        )
        .await
        {
            Ok(response) => {
                let status = response.status();
                // Keep the circuit breaker informed, as the typed path does:
                // 2xx/4xx count as success, 5xx as worker failure.
                let was_available = worker.is_available();
                worker.record_outcome(status.is_success() || status.is_client_error());
                if was_available != worker.is_available() {
                    self.worker_registry.notify_worker_state_change();
                }
                if status.is_success() {
                    if let Some(transfers) = transfers.as_mut() {
                        transfers.disarm();
                    }
                }
                let headers = header_utils::preserve_response_headers(response.headers());
                let mut response_builder = Response::builder().status(status.as_u16());

                for (name, value) in headers.iter() {
                    if name != "transfer-encoding" && name != "content-length" {
                        response_builder = response_builder.header(name, value);
                    }
                }

                if !status.is_success() {
                    if let Some(completion) = &program_completion {
                        completion.finish(false);
                    }
                }

                if Self::should_proxy_transparent_directly(program_completion.is_some()) {
                    response_builder
                        .body(Body::from_stream(response.bytes_stream()))
                        .unwrap_or_else(|error| {
                            (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                format!("Failed to build response: {error}"),
                            )
                                .into_response()
                        })
                } else if Self::should_buffer_transparent_response(
                    is_stream,
                    program_completion.is_some(),
                ) {
                    match response.bytes().await {
                        Ok(bytes) => {
                            if status.is_success() {
                                if let Some(completion) = &program_completion {
                                    completion.observe_json(&bytes);
                                    completion.finish(true);
                                }
                            }
                            response_builder
                                .body(Body::from(bytes))
                                .unwrap_or_else(|error| {
                                    (
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        format!("Failed to build response: {error}"),
                                    )
                                        .into_response()
                                })
                        }
                        Err(error) => {
                            if status.is_success() {
                                if let Some(completion) = &program_completion {
                                    completion.finish(false);
                                }
                            }
                            (
                                StatusCode::BAD_GATEWAY,
                                format!("Failed to read backend response: {error}"),
                            )
                                .into_response()
                        }
                    }
                } else {
                    let stream = response.bytes_stream();
                    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
                    let completion = status.is_success().then_some(program_completion).flatten();
                    tokio::spawn(async move {
                        let mut stream = stream;
                        let mut stream_ok = true;
                        while let Some(chunk) = stream.next().await {
                            match chunk {
                                Ok(bytes) => {
                                    if let Some(completion) = &completion {
                                        completion.observe_sse_chunk(&bytes);
                                    }
                                    if tx.send(Ok(bytes)).is_err() {
                                        stream_ok = false;
                                        break;
                                    }
                                }
                                Err(error) => {
                                    stream_ok = false;
                                    let _ = tx.send(Err(format!("Stream error: {error}")));
                                    break;
                                }
                            }
                        }
                        if let Some(completion) = completion {
                            completion.finish(stream_ok);
                        }
                    });
                    response_builder
                        .body(Body::from_stream(UnboundedReceiverStream::new(rx)))
                        .unwrap_or_else(|error| {
                            (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                format!("Failed to build response: {error}"),
                            )
                                .into_response()
                        })
                }
            }
            Err(error) => {
                // The 502 response below still flows through
                // hold_load_until_body_done, which releases the load once
                // its body finishes or is dropped — do not decrement here.
                if let Some(completion) = &program_completion {
                    completion.finish(false);
                }
                // Transport failure: the worker never produced a response.
                let was_available = worker.is_available();
                worker.record_outcome(false);
                if was_available != worker.is_available() {
                    self.worker_registry.notify_worker_state_change();
                }
                (
                    StatusCode::BAD_GATEWAY,
                    format!("Backend request failed: {error}"),
                )
                    .into_response()
            }
        };
        if response.status().is_success() {
            RouterMetrics::record_request(path);
            RouterMetrics::record_generate_duration(start.elapsed());
        } else {
            RouterMetrics::record_request_error(path, "backend_error");
        }
        if load_tracked {
            hold_request_load_until_body_done(
                response,
                load.expect("tracked requests must own a load guard"),
            )
        } else {
            response
        }
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[tokio::test]
    async fn http_static_configuration_errors_precede_health_and_dp_expansion() {
        use crate::config::{normalize_static_kv_event_sources, RouterConfig};
        use serde_json::json;

        for dp_size in [1, 2] {
            let mut config = RouterConfig {
                policy: serde_json::from_value(json!({
                    "type": "kv_aware",
                    "static_kv_event_sources": {
                        "http://127.0.0.1:1": {
                            "endpoints": ["tcp://127.0.0.1:5557"],
                            "block_size": 8, "model": ""
                        }
                    }
                }))
                .unwrap(),
                intra_node_data_parallel_size: dp_size,
                worker_startup_timeout_secs: 5,
                ..Default::default()
            };
            // Typed explicit configuration is rejected by the real shared
            // boundary before any Router health check or DP expansion.
            let error =
                crate::server::AppContext::new(config.clone(), Client::new(), 10, None, vec![])
                    .err()
                    .expect("invalid static configuration");
            assert!(error.contains("model identity"), "{error}");

            config.policy = serde_json::from_value(json!({"type": "kv_aware"})).unwrap();
            assert!(
                normalize_static_kv_event_sources(&mut config.policy, Some("invalid JSON"))
                    .is_err()
            );
            normalize_static_kv_event_sources(
                &mut config.policy,
                Some(
                    r#"{
                "http://127.0.0.1:1": {
                    "endpoints": ["tcp://127.0.0.1:5557"],
                    "block_size": 8, "model": "m"
                }
            }"#,
                ),
            )
            .unwrap();
            let ctx = Arc::new(
                crate::server::AppContext::new(config, Client::new(), 10, None, vec![]).unwrap(),
            );
            if dp_size == 2 {
                let error = tokio::time::timeout(
                    Duration::from_secs(1),
                    Router::new(vec!["http://127.0.0.1:1".into()], &ctx),
                )
                .await
                .unwrap()
                .unwrap_err();
                assert!(error.contains("DP1 workers"), "{error}");
                let router = Router::new(vec![], &ctx).await.unwrap();
                let error = tokio::time::timeout(
                    Duration::from_secs(1),
                    router.add_worker("http://127.0.0.1:1"),
                )
                .await
                .unwrap()
                .unwrap_err();
                assert!(error.contains("DP1 workers"), "{error}");
                assert!(router.get_worker_urls().is_empty());
            }
        }
    }

    #[tokio::test]
    async fn empty_router_retains_static_mapping_for_future_http_add_worker() {
        use crate::config::{PolicyConfig, RouterConfig};
        use crate::kv_index::{MatchQuery, StorageTier};
        use axum::routing::get;
        use serde_json::json;

        let context = zmq::Context::new();
        let publisher = context.socket(zmq::XPUB).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let worker = format!("http://{}", listener.local_addr().unwrap());
        // Discovery endpoints deliberately do not exist: the saved mapping must
        // supply the source for this later, independently addressed worker.
        let app = axum::Router::new().route("/health", get(|| async { "ok" }));
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let mut config = RouterConfig {
            policy: PolicyConfig::KvAware {
                router_temperature: 0.0,
                kv_penalty_host: 0.8,
                kv_penalty_external: 0.6,
                kv_penalty_disk: 0.3,
                static_kv_event_sources: Default::default(),
            },
            worker_startup_timeout_secs: 5,
            ..Default::default()
        };
        let mapping = json!({&worker: {
            "endpoints": [publisher.get_last_endpoint().unwrap().unwrap()],
            "block_size": 8, "model": "future-model", "topic": "kv/future"
        }})
        .to_string();
        crate::config::normalize_static_kv_event_sources(&mut config.policy, Some(&mapping))
            .unwrap();
        let ctx = Arc::new(
            crate::server::AppContext::new(config, Client::new(), 10, None, vec![]).unwrap(),
        );
        let router = Router::new(vec![], &ctx).await.unwrap();
        router.add_worker(&worker).await.unwrap();
        let supervisor = router.kv_index_supervisor.as_ref().unwrap();
        let provider = supervisor.match_provider_for_worker(&worker).unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                match publisher.recv_bytes(zmq::DONTWAIT) {
                    Ok(frame) => {
                        assert_eq!(frame, b"\x01kv/future");
                        break;
                    }
                    Err(zmq::Error::EAGAIN) => tokio::task::yield_now().await,
                    Err(error) => panic!("future worker subscription: {error}"),
                }
            }
        })
        .await
        .unwrap();
        let payload = rmp_serde::to_vec(&(
            1.0,
            [json!({
                "type": "BlockStored", "block_hashes": [1], "parent_block_hash": null,
                "token_ids": vec![1; 8], "block_size": 8, "medium": "GPU",
                "lora_id": null, "lora_name": null
            })],
            0u32,
        ))
        .unwrap();
        publisher
            .send_multipart([b"kv/future".as_slice(), &0i64.to_be_bytes(), &payload], 0)
            .unwrap();
        let query = MatchQuery {
            group_idx: 0,
            local_hashes: crate::kv_index::local_block_hash_strings(&[1; 8], 8, None),
            tiers_of_interest: vec![StorageTier::Device],
        };
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let hits = provider.find_tiered_matches(&query);
                if !hits.is_empty() {
                    assert_eq!(hits.len(), 1);
                    assert_eq!(hits[0].target.instance_id.as_ref(), worker);
                    assert_eq!(hits[0].segments, [(0, 1)]);
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        supervisor.shutdown();
        task.abort();
    }

    #[tokio::test]
    async fn discovered_http_sources_use_dense_admission_through_router_startup_and_add() {
        use crate::config::{PolicyConfig, RouterConfig};
        use crate::kv_index::{MatchQuery, StorageTier};
        use axum::{routing::get, Json};
        use serde_json::json;

        assert!(std::env::var_os("VLLM_ROUTER_KV_EVENT_SOURCES").is_none());
        for (expected, actual, replayed, empty_start) in [
            (16u32, 8u32, false, false),
            (8, 16, true, true),
            (8, 8, false, false),
            (16, 16, true, false),
            (32, 32, false, true),
        ] {
            let context = zmq::Context::new();
            let publisher = context.socket(zmq::XPUB).unwrap();
            publisher.bind("tcp://127.0.0.1:*").unwrap();
            let replay = context.socket(zmq::ROUTER).unwrap();
            replay.bind("tcp://127.0.0.1:*").unwrap();
            let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
            let replay_endpoint = replay.get_last_endpoint().unwrap().unwrap();
            let sources = json!({"3": {
                "endpoint": endpoint, "replay_endpoint": replay_endpoint,
                "topic": "kv/router", "publisher": "zmq", "encoding": "msgpack",
                "schema_version": 1, "enable_kv_cache_events": true
            }});
            let app = axum::Router::new()
                .route("/health", get(|| async { "ok" }))
                .route("/get_server_info", get(move || async move { Json(json!({
                    "model_id": "router-dense", "instance_id": "router-source", "kv_block_size": expected
                })) }))
                .route("/kv_event_sources", get(move || {
                    let sources = sources.clone();
                    async move { Json(sources) }
                }));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let worker = format!("http://{}", listener.local_addr().unwrap());
            let http_task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let config = RouterConfig {
                policy: PolicyConfig::KvAware {
                    router_temperature: 0.0,
                    kv_penalty_host: 0.8,
                    kv_penalty_external: 0.6,
                    kv_penalty_disk: 0.3,
                    static_kv_event_sources: Default::default(),
                },
                worker_startup_timeout_secs: 5,
                worker_startup_check_interval_secs: 1,
                ..Default::default()
            };
            let ctx = Arc::new(
                crate::server::AppContext::new(config, Client::new(), 10, None, vec![]).unwrap(),
            );
            let router = Router::new(
                if empty_start {
                    vec![]
                } else {
                    vec![worker.clone()]
                },
                &ctx,
            )
            .await
            .unwrap();
            if empty_start {
                router.add_worker(&worker).await.unwrap();
            }
            let supervisor = router.kv_index_supervisor.as_ref().unwrap();
            let provider = supervisor.match_provider_for_worker(&worker).unwrap();
            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    match publisher.recv_bytes(zmq::DONTWAIT) {
                        Ok(frame) => {
                            assert_eq!(frame, b"\x01kv/router");
                            break;
                        }
                        Err(zmq::Error::EAGAIN) => tokio::task::yield_now().await,
                        Err(error) => panic!("Router XPUB subscription: {error}"),
                    }
                }
            })
            .await
            .unwrap();
            let payload =
                |hash: u32, token: u32, block_size: u32| {
                    rmp_serde::to_vec(&(1.0, [json!({
                "type": "BlockStored", "block_hashes": [hash], "parent_block_hash": null,
                "token_ids": vec![token; block_size as usize], "block_size": block_size,
                "medium": "CPU", "lora_id": null, "lora_name": null
            })], 3u32)).unwrap()
                };
            let query = |token| MatchQuery {
                group_idx: 0,
                local_hashes: crate::kv_index::local_block_hash_strings(
                    &vec![token; expected as usize],
                    expected,
                    None,
                ),
                tiers_of_interest: vec![StorageTier::HostPinned],
            };
            publisher
                .send_multipart(
                    [
                        b"kv/router".as_slice(),
                        &0i64.to_be_bytes(),
                        &payload(1, 1, expected),
                    ],
                    0,
                )
                .unwrap();
            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    let hits = provider.find_tiered_matches(&query(1));
                    if !hits.is_empty() {
                        assert_eq!(hits.len(), 1);
                        assert_eq!(hits[0].target.instance_id.as_ref(), "router-source");
                        assert_eq!(hits[0].target.dp_rank, 3);
                        assert_eq!(hits[0].tier, StorageTier::HostPinned);
                        assert_eq!(hits[0].segments, [(0, 1)]);
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            if replayed {
                let fresh = payload(3, 3, expected);
                publisher
                    .send_multipart([b"kv/router".as_slice(), &2i64.to_be_bytes(), &fresh], 0)
                    .unwrap();
                let request = tokio::time::timeout(Duration::from_secs(5), async {
                    loop {
                        match replay.recv_multipart(zmq::DONTWAIT) {
                            Ok(request) => break request,
                            Err(zmq::Error::EAGAIN) => tokio::task::yield_now().await,
                            Err(error) => panic!("Router replay: {error}"),
                        }
                    }
                })
                .await
                .unwrap();
                assert_eq!(request.last().unwrap(), &1i64.to_be_bytes());
                for (seq, body) in [(1i64, payload(2, 2, actual)), (2, fresh), (-1, vec![])] {
                    replay
                        .send_multipart(
                            [
                                request[0].clone(),
                                vec![],
                                if seq == -1 {
                                    vec![]
                                } else {
                                    b"kv/router".to_vec()
                                },
                                seq.to_be_bytes().to_vec(),
                                body,
                            ],
                            0,
                        )
                        .unwrap();
                }
            } else {
                publisher
                    .send_multipart(
                        [
                            b"kv/router".as_slice(),
                            &1i64.to_be_bytes(),
                            &payload(2, 2, actual),
                        ],
                        0,
                    )
                    .unwrap();
            }
            let valid = expected == actual;
            let result = tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    let old = provider.find_tiered_matches(&query(1));
                    let new = provider.find_tiered_matches(&query(2));
                    if valid && !new.is_empty() {
                        assert_eq!(new[0].segments, [(0, 1)]);
                        break true;
                    }
                    if !valid && old.is_empty() {
                        assert!(new.is_empty());
                        break true;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await;
            supervisor.shutdown();
            if let Some(task) = &router._kv_signal_handle {
                task.abort();
            }
            if let Some(task) = &router._load_monitor_handle {
                task.abort();
            }
            http_task.abort();
            assert!(result.is_ok(), "ordinary Router accepted source/event mismatch: source={expected},event={actual},replay={replayed},empty={empty_start}");
        }
    }

    struct CompletionCountingTokenizer {
        inner: vllm_tokenizer::HuggingFaceTokenizer,
        encodes: Arc<std::sync::atomic::AtomicUsize>,
    }

    struct CompletionBarrierTokenizer {
        started: tokio::sync::mpsc::UnboundedSender<()>,
        finished: tokio::sync::mpsc::UnboundedSender<()>,
        release: Mutex<std::sync::mpsc::Receiver<()>>,
    }

    impl vllm_tokenizer::Tokenizer for CompletionBarrierTokenizer {
        fn encode(&self, _: &str, _: bool) -> vllm_tokenizer::Result<Vec<u32>> {
            self.started.send(()).unwrap();
            self.release
                .lock()
                .recv_timeout(Duration::from_secs(10))
                .unwrap();
            self.finished.send(()).unwrap();
            Ok(vec![1, 3, 4])
        }
        fn encode_ordinary(&self, _: &str) -> vllm_tokenizer::Result<Vec<u32>> {
            unreachable!()
        }
        fn decode(&self, _: &[u32], _: bool) -> vllm_tokenizer::Result<String> {
            unreachable!()
        }
        fn token_to_id(&self, _: &str) -> Option<u32> {
            None
        }
        fn id_to_token(&self, _: u32) -> Option<String> {
            None
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn completion_cancelled_preparation_keeps_health_live_and_never_dispatches() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            format!("http://{}", listener.local_addr().unwrap()),
            WorkerType::Regular,
        ));
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let seen = calls.clone();
        let app = axum::Router::new().route(
            "/v1/completions",
            axum::routing::post(move || {
                seen.fetch_add(1, Ordering::Relaxed);
                async { Json(serde_json::json!({"choices":[]})) }
            }),
        );
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let mut router = create_test_regular_router();
        router.worker_registry = Arc::new(WorkerRegistry::new());
        router.worker_registry.register(worker.clone());
        router.policy_registry =
            Arc::new(PolicyRegistry::new(crate::config::PolicyConfig::KvAware {
                router_temperature: 0.0,
                kv_penalty_host: 0.8,
                kv_penalty_external: 0.6,
                kv_penalty_disk: 0.3,
                static_kv_event_sources: Default::default(),
            }));
        let (started, mut encoding) = tokio::sync::mpsc::unbounded_channel();
        let (finished, mut encoded) = tokio::sync::mpsc::unbounded_channel();
        let (release, released) = std::sync::mpsc::channel();
        router.completion_input = Some(crate::backend::completion_input::CompletionInput {
            model: "qwen3".into(),
            cache_model: "qwen3".into(),
            vocab_size: 16,
            tokenizer: Arc::new(CompletionBarrierTokenizer {
                started,
                finished,
                release: Mutex::new(released),
            }),
        });
        let preparation_capacity = router.completion_preparation.available_admission_permits();
        let router = Arc::new(router);
        let task = tokio::spawn({
            let router = router.clone();
            async move {
                let raw =
                    bytes::Bytes::from_static(b"{\"model\":\"qwen3\",\"prompt\":\"hello world\"}");
                router
                    .route_completion_raw(
                        None,
                        Arc::new(serde_json::from_slice(&raw).unwrap()),
                        &raw,
                        None,
                    )
                    .await
            }
        });
        tokio::time::timeout(Duration::from_secs(5), encoding.recv())
            .await
            .unwrap()
            .unwrap();
        let health = tokio::time::timeout(
            Duration::from_secs(5),
            router.health(Request::new(Body::empty())),
        )
        .await
        .unwrap();
        assert_eq!(health.status(), StatusCode::OK);
        assert_eq!(worker.load(), 0, "preparation must not reserve a Worker");
        let policy = router.policy_registry.get_default_policy();
        let kv = policy
            .as_any()
            .downcast_ref::<crate::policies::KvAwarePolicy>()
            .unwrap();
        let manager = kv.slot_manager().unwrap();
        let target = kv.target_for_worker(worker.as_ref());
        assert_eq!(
            manager.sample(&target),
            crate::policies::LoadSample::default()
        );
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert_eq!(
            router.completion_preparation.available_admission_permits(),
            preparation_capacity - 1
        );
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(5), encoded.recv())
            .await
            .unwrap()
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while router.completion_preparation.available_admission_permits()
                != preparation_capacity
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("cancelled preparation job actually exited");
        assert_eq!(
            calls.load(Ordering::Relaxed),
            0,
            "a cancelled preparation cannot dispatch after encoding"
        );
        assert_eq!(worker.load(), 0);
        assert_eq!(
            manager.sample(&target),
            crate::policies::LoadSample::default()
        );
        server.abort();
    }

    #[tokio::test]
    async fn completion_retry_reselects_another_worker_and_preserves_active_request() {
        let first_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let second_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let first: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            format!("http://{}", first_listener.local_addr().unwrap()),
            WorkerType::Regular,
        ));
        let second: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            format!("http://{}", second_listener.local_addr().unwrap()),
            WorkerType::Regular,
        ));
        second.set_healthy(false);
        let mut router = create_test_regular_router();
        router.worker_registry = Arc::new(WorkerRegistry::new());
        router.worker_registry.register(first.clone());
        router.worker_registry.register(second.clone());
        router.policy_registry =
            Arc::new(PolicyRegistry::new(crate::config::PolicyConfig::KvAware {
                router_temperature: 0.0,
                kv_penalty_host: 0.8,
                kv_penalty_external: 0.6,
                kv_penalty_disk: 0.3,
                static_kv_event_sources: Default::default(),
            }));
        router.retry_config = RetryConfig {
            max_retries: 2,
            initial_backoff_ms: 0,
            max_backoff_ms: 0,
            jitter_factor: 0.0,
            ..Default::default()
        };
        let encodes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        router.completion_input = Some(crate::backend::completion_input::CompletionInput {
            model: "qwen3".into(),
            cache_model: "qwen3".into(),
            vocab_size: 16,
            tokenizer: Arc::new(CompletionCountingTokenizer {
                inner: vllm_tokenizer::HuggingFaceTokenizer::new_hf(std::path::Path::new(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/rust/fixtures/tokenizer/completion_word_level.json"
                )))
                .unwrap(),
                encodes: encodes.clone(),
            }),
        });
        let policy = router.policy_registry.get_default_policy();
        let kv = policy
            .as_any()
            .downcast_ref::<crate::policies::KvAwarePolicy>()
            .unwrap();
        let manager = kv.slot_manager().unwrap();
        let first_target = kv.target_for_worker(first.as_ref());
        let second_target = kv.target_for_worker(second.as_ref());
        let captured = Arc::new(Mutex::new(Vec::new()));
        let (arrived, mut arrivals) = tokio::sync::mpsc::unbounded_channel();
        let (release, held) = tokio::sync::oneshot::channel();
        let held = Arc::new(Mutex::new(Some(held)));
        let app = axum::Router::new().route(
            "/v1/completions",
            axum::routing::post({
                let (first, second, manager, target, captured) = (
                    first.clone(),
                    second.clone(),
                    manager.clone(),
                    first_target.clone(),
                    captured.clone(),
                );
                move |body: bytes::Bytes| {
                    let (first, second, manager, target, captured, arrived, held) = (
                        first.clone(),
                        second.clone(),
                        manager.clone(),
                        target.clone(),
                        captured.clone(),
                        arrived.clone(),
                        held.clone(),
                    );
                    async move {
                        let request: serde_json::Value = serde_json::from_slice(&body).unwrap();
                        if request["prompt"].is_array() {
                            let held = held.lock().take().unwrap();
                            assert_eq!(
                                manager.sample(&target),
                                crate::policies::LoadSample {
                                    active_prefill_tokens: 2,
                                    active_requests: 1
                                }
                            );
                            arrived.send(()).unwrap();
                            held.await.unwrap();
                            Json(serde_json::json!({"choices":[]})).into_response()
                        } else {
                            assert_eq!(
                                manager.sample(&target),
                                crate::policies::LoadSample {
                                    active_prefill_tokens: 5,
                                    active_requests: 2
                                }
                            );
                            assert_eq!(first.load(), 2);
                            captured.lock().push(("A", body));
                            first.set_healthy(false);
                            second.set_healthy(true);
                            StatusCode::INTERNAL_SERVER_ERROR.into_response()
                        }
                    }
                }
            }),
        );
        let first_server =
            tokio::spawn(async move { axum::serve(first_listener, app).await.unwrap() });
        let app = axum::Router::new().route(
            "/v1/completions",
            axum::routing::post({
                let (first, second, manager, first_target, second_target, captured) = (
                    first.clone(),
                    second.clone(),
                    manager.clone(),
                    first_target.clone(),
                    second_target.clone(),
                    captured.clone(),
                );
                move |body: bytes::Bytes| {
                    let (first, second, manager, first_target, second_target, captured) = (
                        first.clone(),
                        second.clone(),
                        manager.clone(),
                        first_target.clone(),
                        second_target.clone(),
                        captured.clone(),
                    );
                    async move {
                        assert_eq!(
                            manager.sample(&first_target),
                            crate::policies::LoadSample {
                                active_prefill_tokens: 2,
                                active_requests: 1
                            }
                        );
                        assert_eq!(
                            manager.sample(&second_target),
                            crate::policies::LoadSample {
                                active_prefill_tokens: 3,
                                active_requests: 1
                            }
                        );
                        assert_eq!((first.load(), second.load()), (1, 1));
                        captured.lock().push(("B", body));
                        Json(serde_json::json!({"choices":[{"text":"ok"}]}))
                    }
                }
            }),
        );
        let second_server =
            tokio::spawn(async move { axum::serve(second_listener, app).await.unwrap() });
        let router = Arc::new(router);
        let background = tokio::spawn({
            let router = router.clone();
            async move {
                let raw = bytes::Bytes::from_static(b"{\"model\":\"qwen3\",\"prompt\":[1,2]}");
                let response = router
                    .route_completion_raw(
                        None,
                        Arc::new(serde_json::from_slice(&raw).unwrap()),
                        &raw,
                        None,
                    )
                    .await;
                assert_eq!(response.status(), StatusCode::OK);
                to_bytes(response.into_body(), usize::MAX).await.unwrap();
            }
        });
        tokio::time::timeout(Duration::from_secs(5), arrivals.recv())
            .await
            .unwrap()
            .unwrap();
        let raw =
            bytes::Bytes::from_static(b" { \"model\":\"qwen3\", \"prompt\":\"hello world\" }\n");
        let response = tokio::time::timeout(
            Duration::from_secs(5),
            router.route_completion_raw(
                None,
                Arc::new(serde_json::from_slice(&raw).unwrap()),
                &raw,
                None,
            ),
        )
        .await
        .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(*captured.lock(), vec![("A", raw.clone()), ("B", raw)]);
        assert_eq!(encodes.load(Ordering::Relaxed), 1);
        assert_eq!((first.load(), second.load()), (1, 0));
        assert_eq!(
            manager.sample(&first_target),
            crate::policies::LoadSample {
                active_prefill_tokens: 2,
                active_requests: 1
            }
        );
        assert_eq!(
            manager.sample(&second_target),
            crate::policies::LoadSample::default()
        );
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(5), background)
            .await
            .unwrap()
            .unwrap();
        assert_eq!((first.load(), second.load()), (0, 0));
        assert_eq!(
            manager.sample(&first_target),
            crate::policies::LoadSample::default()
        );
        first_server.abort();
        second_server.abort();
    }

    impl vllm_tokenizer::Tokenizer for CompletionCountingTokenizer {
        fn encode(&self, text: &str, special: bool) -> vllm_tokenizer::Result<Vec<u32>> {
            self.encodes.fetch_add(1, Ordering::Relaxed);
            self.inner.encode(text, special)
        }
        fn encode_ordinary(&self, _text: &str) -> vllm_tokenizer::Result<Vec<u32>> {
            panic!("Completion preparation must call encode directly")
        }
        fn decode(&self, ids: &[u32], special: bool) -> vllm_tokenizer::Result<String> {
            self.inner.decode(ids, special)
        }
        fn token_to_id(&self, text: &str) -> Option<u32> {
            self.inner.token_to_id(text)
        }
        fn id_to_token(&self, id: u32) -> Option<String> {
            self.inner.id_to_token(id)
        }
    }

    #[tokio::test]
    async fn completion_classification_controls_actual_dispatch() {
        assert_completion_classification_dispatch(None).await;
    }

    #[derive(Clone, Copy, Debug)]
    enum HeldCompletionCapacity {
        Cpu,
        Admission,
    }

    impl HeldCompletionCapacity {
        fn hold(self, router: &mut Router) -> tokio::sync::OwnedSemaphorePermit {
            let cpu = Arc::new(tokio::sync::Semaphore::new(1));
            let admission = Arc::new(tokio::sync::Semaphore::new(1));
            let held = match self {
                Self::Cpu => cpu.clone().try_acquire_owned().unwrap(),
                Self::Admission => admission.clone().try_acquire_owned().unwrap(),
            };
            assert_eq!(
                cpu.available_permits(),
                usize::from(matches!(self, Self::Admission))
            );
            assert_eq!(
                admission.available_permits(),
                usize::from(matches!(self, Self::Cpu))
            );
            router.completion_preparation =
                crate::backend::completion_input::CompletionPreparation::with_test_capacity(
                    cpu,
                    admission,
                    Duration::from_secs(2),
                    Duration::from_millis(20),
                );
            held
        }
    }

    #[tokio::test]
    async fn completion_preflight_bypasses_held_cpu_capacity() {
        assert_completion_classification_dispatch(Some(HeldCompletionCapacity::Cpu)).await;
        assert_unsupported_completion_stream(Some(HeldCompletionCapacity::Cpu)).await;
    }

    #[tokio::test]
    async fn completion_preflight_bypasses_held_admission_capacity() {
        assert_completion_classification_dispatch(Some(HeldCompletionCapacity::Admission)).await;
        assert_unsupported_completion_stream(Some(HeldCompletionCapacity::Admission)).await;
    }

    async fn assert_completion_classification_dispatch(capacity: Option<HeldCompletionCapacity>) {
        for (raw, vocab_size, fixture, status, encodes_expected, dispatched, error) in [
            (
                bytes::Bytes::from_static(b" { \"model\":\"qwen3\", \"prompt\":\"hello world\", \"cache_salt\":\"tenant\" }\n"),
                16,
                "completion_word_level.json",
                StatusCode::OK,
                0,
                true,
                "",
            ),
            (
                bytes::Bytes::from_static(br#"{"model":"qwen3","prompt":["hello","world"]}"#),
                16,
                "completion_word_level.json",
                StatusCode::OK,
                0,
                true,
                "",
            ),
            (
                bytes::Bytes::from_static(br#"{"model":"qwen3","prompt":[[3],[15]]}"#),
                16,
                "completion_word_level.json",
                StatusCode::OK,
                0,
                true,
                "",
            ),
            (
                bytes::Bytes::from_static(br#"{"model":"qwen3","prompt":"hello world","suffix":"tail"}"#),
                16,
                "completion_word_level.json",
                StatusCode::OK,
                0,
                true,
                "",
            ),
            (
                bytes::Bytes::from_static(br#"{"model":"qwen3","prompt":[0,15,3],"add_special_tokens":true}"#),
                16,
                "completion_word_level.json",
                StatusCode::OK,
                0,
                true,
                "",
            ),
            (
                bytes::Bytes::from_static(b"{\"model\":\"qwen3\",\"prompt\":[-1],\"cache_salt\":\"tenant\"}"),
                16,
                "completion_word_level.json",
                StatusCode::BAD_REQUEST,
                0,
                false,
                "completion prompt contains negative token ID -1",
            ),
            (
                bytes::Bytes::from_static(b"{\"model\":\"qwen3\",\"prompt\":[16],\"cache_salt\":\"tenant\"}"),
                16,
                "completion_word_level.json",
                StatusCode::BAD_REQUEST,
                0,
                false,
                "completion token ID 16 is outside model vocabulary size 16",
            ),
            (
                bytes::Bytes::from_static(br#"{"model":"qwen3","prompt":[-1],"suffix":"tail"}"#),
                16,
                "completion_word_level.json",
                StatusCode::BAD_REQUEST,
                0,
                false,
                "completion prompt contains negative token ID -1",
            ),
            (
                bytes::Bytes::from_static(br#"{"model":"qwen3","prompt":[16],"lora_path":"adapter"}"#),
                16,
                "completion_word_level.json",
                StatusCode::BAD_REQUEST,
                0,
                false,
                "completion token ID 16 is outside model vocabulary size 16",
            ),
            (
                bytes::Bytes::from_static(br#"{"model":"qwen3","prompt":[],"cache_salt":"empty-ids"}"#),
                16,
                "completion_word_level.json",
                StatusCode::BAD_REQUEST,
                0,
                false,
                "completion preparation requires nonempty token input",
            ),
            (
                bytes::Bytes::from_static(br#"{"model":"qwen3","prompt":[[3],[-1]],"suffix":"tail"}"#),
                16,
                "completion_word_level.json",
                StatusCode::BAD_REQUEST,
                0,
                false,
                "completion prompt contains negative token ID -1",
            ),
            (
                bytes::Bytes::from_static(br#"{"model":"qwen3","prompt":[[3],[16]],"cache_salt":"tenant"}"#),
                16,
                "completion_word_level.json",
                StatusCode::BAD_REQUEST,
                0,
                false,
                "completion token ID 16 is outside model vocabulary size 16",
            ),
            (
                bytes::Bytes::from_static(br#"{"model":"qwen3","prompt":[[3],[]],"lora_path":"adapter"}"#),
                16,
                "completion_word_level.json",
                StatusCode::BAD_REQUEST,
                0,
                false,
                "completion preparation requires nonempty token input",
            ),
            (
                bytes::Bytes::from_static(br#"{"model":"qwen3","prompt":"hello","suffix":"tail","cache_salt":"tenant","lora_path":"adapter","add_special_tokens":1}"#),
                16,
                "completion_word_level.json",
                StatusCode::BAD_REQUEST,
                0,
                false,
                "completion add_special_tokens must be a boolean",
            ),
            (
                bytes::Bytes::from_static(br#"{"model":"qwen3","prompt":[-1],"suffix":"tail","add_special_tokens":1}"#),
                0,
                "completion_word_level.json",
                StatusCode::SERVICE_UNAVAILABLE,
                0,
                false,
                "completion model vocabulary must be nonzero",
            ),
            (
                bytes::Bytes::from_static(b"{\"model\":\"qwen3\",\"prompt\":\"hello world\"}"),
                0,
                "completion_word_level.json",
                StatusCode::SERVICE_UNAVAILABLE,
                0,
                false,
                "completion model vocabulary must be nonzero",
            ),
            (
                bytes::Bytes::from_static(b"{\"model\":\"qwen3\",\"prompt\":\"unknown\"}"),
                16,
                "completion_missing_unk.json",
                StatusCode::SERVICE_UNAVAILABLE,
                1,
                false,
                "completion tokenization failed:",
            ),
            (
                bytes::Bytes::from_static(br#"{"model":"qwen3","prompt":"hello world"}"#),
                16,
                "completion_word_level.json",
                StatusCode::OK,
                1,
                true,
                "",
            ),
        ] {
            // Only supported text is subject to either encoding-capacity gate.
            let (status, encodes_expected, dispatched, error) = match capacity {
                Some(HeldCompletionCapacity::Cpu) if encodes_expected > 0 => (
                    StatusCode::REQUEST_TIMEOUT,
                    0,
                    false,
                    "Completion preparation queue timed out",
                ),
                Some(HeldCompletionCapacity::Admission) if encodes_expected > 0 => (
                    StatusCode::TOO_MANY_REQUESTS,
                    0,
                    false,
                    "Completion preparation capacity exhausted",
                ),
                _ => (status, encodes_expected, dispatched, error),
            };
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
                format!("http://{}", listener.local_addr().unwrap()),
                WorkerType::Regular,
            ));
            let mut router = create_test_regular_router();
            let held_capacity = capacity.map(|capacity| capacity.hold(&mut router));
            router.worker_registry = Arc::new(WorkerRegistry::new());
            router.worker_registry.register(worker.clone());
            router.policy_registry =
                Arc::new(PolicyRegistry::new(crate::config::PolicyConfig::KvAware {
                    router_temperature: 0.0,
                    kv_penalty_host: 0.8,
                    kv_penalty_external: 0.6,
                    kv_penalty_disk: 0.3,
                    static_kv_event_sources: Default::default(),
                }));
            router.retry_config = RetryConfig {
                max_retries: 2,
                initial_backoff_ms: 0,
                max_backoff_ms: 0,
                jitter_factor: 0.0,
                ..Default::default()
            };
            let encodes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            router.completion_input = Some(crate::backend::completion_input::CompletionInput {
                model: "qwen3".into(),
                cache_model: "qwen3".into(),
                vocab_size,
                tokenizer: Arc::new(CompletionCountingTokenizer {
                    inner: vllm_tokenizer::HuggingFaceTokenizer::new_hf(
                        &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                            .join("tests/rust/fixtures/tokenizer")
                            .join(fixture),
                    )
                    .unwrap(),
                    encodes: encodes.clone(),
                }),
            });
            let policy = router.policy_registry.get_default_policy();
            assert_eq!(policy.name(), "kv_aware");
            let kv = policy
                .as_any()
                .downcast_ref::<crate::policies::KvAwarePolicy>()
                .unwrap();
            let manager = kv.slot_manager().unwrap();
            let target = kv.target_for_worker(worker.as_ref());
            let captured = Arc::new(Mutex::new(Vec::new()));
            let (bodies, inflight_manager, inflight_target, inflight_worker) =
                (captured.clone(), manager.clone(), target.clone(), worker.clone());
            let mut request: CompletionRequest = serde_json::from_slice(&raw).unwrap();
            // Wire [] is an empty IntBatch under the untagged enum; exercise
            // the existing typed empty-IntArray contract without changing it.
            if request.other.get("cache_salt").and_then(serde_json::Value::as_str) == Some("empty-ids") {
                request.prompt = crate::protocols::spec::PromptInput::IntArray(vec![]);
            }
            let input = router.completion_input.as_ref().unwrap();
            assert_eq!(request.model.as_deref(), Some(input.model.as_str()));
            assert_eq!(input.model, "qwen3");
            assert_eq!(input.vocab_size, vocab_size);
            let explicit_ids = matches!(&request.prompt, crate::protocols::spec::PromptInput::IntArray(ids) if ids == &[0, 15, 3]);
            if explicit_ids {
                assert!(input.tokenizer.id_to_token(15).is_none());
            }
            let expected_prefill = if explicit_ids || encodes_expected > 0 { 3 } else { 0 };
            let app = axum::Router::new().route(
                "/v1/completions",
                axum::routing::post(move |body: axum::body::Bytes| {
                    let (bodies, manager, target, worker) = (
                        bodies.clone(),
                        inflight_manager.clone(),
                        inflight_target.clone(),
                        inflight_worker.clone(),
                    );
                    async move {
                        assert_eq!(
                            manager.sample(&target),
                            crate::policies::LoadSample {
                                active_prefill_tokens: expected_prefill,
                                active_requests: 1,
                            },
                            "Only prepared IDs/text contribute input load; no source has cache credit"
                        );
                        assert_eq!(worker.load(), 1);
                        bodies.lock().push(body);
                        Json(serde_json::json!({"choices":[{"text":"ok"}]}))
                    }
                }),
            );
            let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let original_request = serde_json::to_value(&request).unwrap();
            let response = tokio::time::timeout(
                Duration::from_secs(5),
                router.route_completion_raw(None, Arc::new(request.clone()), &raw, None),
            )
            .await
            .unwrap();
            let actual_status = response.status();
            let response_body = tokio::time::timeout(
                Duration::from_secs(5),
                to_bytes(response.into_body(), usize::MAX),
            )
            .await
            .unwrap()
            .unwrap();
            server.abort();
            assert_eq!(actual_status, status, "{raw:?}: {response_body:?}");
            if !dispatched {
                let actual_error = std::str::from_utf8(&response_body).unwrap();
                if error == "completion tokenization failed:" {
                    assert!(actual_error.starts_with(error), "{actual_error}");
                } else {
                    assert_eq!(actual_error, error);
                }
            }
            assert_eq!(serde_json::to_value(&request).unwrap(), original_request);
            assert_eq!(encodes.load(Ordering::Relaxed), encodes_expected);
            assert_eq!(captured.lock().len(), usize::from(dispatched));
            if dispatched {
                assert_eq!(captured.lock()[0], raw);
                if encodes_expected == 0 {
                    // Inspect the same real HTTP preparation consumer as the
                    // Router dispatch, without a synthetic classification stub.
                    let completion = prepare_completion_route(
                        Some(input),
                        &router.completion_preparation,
                        policy.as_ref(),
                        Arc::new(request.clone()),
                        Some(&raw),
                    )
                    .await
                    .unwrap();
                    assert_eq!(
                        completion.tokens,
                        match request.prompt {
                            crate::protocols::spec::PromptInput::IntArray(_) => Some(vec![0, 15, 3]),
                            _ => None,
                        },
                        "Unsupported has no tokens; explicit IDs retain exact order without BOS/EOS"
                    );
                    assert_eq!(completion.raw_body, Some(&raw));
                    assert_eq!(encodes.load(Ordering::Relaxed), 0);
                }
            }
            assert_eq!(worker.load(), 0);
            assert_eq!(manager.sample(&target), crate::policies::LoadSample::default());
            if let Some(held) = held_capacity {
                assert_eq!(held.semaphore().available_permits(), 0);
                assert_eq!(
                    router.completion_preparation.available_admission_permits(),
                    usize::from(matches!(capacity, Some(HeldCompletionCapacity::Cpu)))
                );
            }
        }
    }

    #[tokio::test]
    async fn completion_placement_qualifies_each_exact_model_and_dp1_target() {
        let context = zmq::Context::new();
        let mut publishers = Vec::new();
        let mut servers = Vec::new();
        let mut workers: Vec<Arc<dyn Worker>> = Vec::new();
        for (model, rank) in [
            ("engine-cache-identity", Some(0)),
            ("engine-cache-identity", None),
            ("wrong-cache-identity", Some(0)),
            ("engine-cache-identity", Some(1)),
        ] {
            let publisher = context.socket(zmq::XPUB).unwrap();
            publisher.bind("tcp://127.0.0.1:*").unwrap();
            let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
            publishers.push(publisher);
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            workers.push(Arc::new(BasicWorker::new(url, WorkerType::Regular)));
            let app = axum::Router::new()
                .route(
                    "/get_server_info",
                    axum::routing::get(move || async move {
                        Json(serde_json::json!({"model_id":model,"kv_block_size":4}))
                    }),
                )
                .route(
                    "/kv_event_sources",
                    axum::routing::get(move || {
                        let endpoint = endpoint.clone();
                        async move {
                            match rank {
                                Some(rank) => Json(serde_json::json!({rank.to_string(): {
                                    "endpoint":endpoint,"publisher":"zmq","encoding":"msgpack",
                                    "schema_version":1,"enable_kv_cache_events":true
                                }}))
                                .into_response(),
                                None => StatusCode::NOT_FOUND.into_response(),
                            }
                        }
                    }),
                );
            servers.push(tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap()
            }));
        }
        let (supervisor, _signals) = crate::kv_index::KvIndexSupervisor::new_http_dense(
            Client::new(),
            crate::kv_index::HashMode::Sha256,
        );
        for worker in &workers {
            supervisor.on_worker_added(worker.url()).await.unwrap();
        }
        let mut router = create_test_regular_router();
        router.completion_input = Some(crate::backend::completion_input::CompletionInput {
            model: "public-alias".into(),
            cache_model: "engine-cache-identity".into(),
            vocab_size: 16,
            tokenizer: Arc::new(
                vllm_tokenizer::HuggingFaceTokenizer::new_hf(std::path::Path::new(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/rust/fixtures/tokenizer/completion_word_level.json"
                )))
                .unwrap(),
            ),
        });
        router.kv_index_supervisor = Some(Arc::new(supervisor));
        assert_eq!(
            placement_eligibility(
                router.completion_input.as_ref(),
                router.kv_index_supervisor.as_deref(),
                &workers,
            ),
            Some(vec![true, false, false, false])
        );
        let wrong_rank: Arc<dyn Worker> = Arc::new(crate::core::DPAwareWorker::new(
            workers[0].base_url().to_owned(),
            1,
            2,
            WorkerType::Regular,
        ));
        assert_eq!(
            placement_eligibility(
                router.completion_input.as_ref(),
                router.kv_index_supervisor.as_deref(),
                &[wrong_rank],
            ),
            Some(vec![false])
        );
        router.kv_index_supervisor.as_ref().unwrap().shutdown();
        for server in servers {
            server.abort();
        }
    }

    #[tokio::test]
    async fn completion_retry_prepares_once_preserves_raw_and_releases_only_own_load() {
        use crate::kv_index::wire::ExternalBlockHash;
        use crate::kv_index::{BlockStored, IngestionSignal, KVEvent};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            format!("http://{}", listener.local_addr().unwrap()),
            WorkerType::Regular,
        ));
        let mut router = create_test_regular_router();
        router.worker_registry = Arc::new(WorkerRegistry::new());
        router.worker_registry.register(worker.clone());
        router.policy_registry =
            Arc::new(PolicyRegistry::new(crate::config::PolicyConfig::KvAware {
                router_temperature: 0.0,
                kv_penalty_host: 0.8,
                kv_penalty_external: 0.6,
                kv_penalty_disk: 0.3,
                static_kv_event_sources: Default::default(),
            }));
        router.retry_config = RetryConfig {
            max_retries: 2,
            initial_backoff_ms: 0,
            max_backoff_ms: 0,
            jitter_factor: 0.0,
            ..Default::default()
        };
        let encodes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        router.completion_input = Some(crate::backend::completion_input::CompletionInput {
            model: "qwen3".into(),
            cache_model: "qwen3".into(),
            vocab_size: 16,
            tokenizer: Arc::new(CompletionCountingTokenizer {
                inner: vllm_tokenizer::HuggingFaceTokenizer::new_hf(std::path::Path::new(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/rust/fixtures/tokenizer/completion_word_level.json"
                )))
                .unwrap(),
                encodes: encodes.clone(),
            }),
        });
        let policy = router.policy_registry.get_default_policy();
        let kv = policy
            .as_any()
            .downcast_ref::<crate::policies::KvAwarePolicy>()
            .unwrap();
        let manager = kv.slot_manager().unwrap();
        let target = kv.target_for_worker(worker.as_ref());
        let context = zmq::Context::new();
        let publisher = context.socket(zmq::XPUB).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let captured = Arc::new(Mutex::new(Vec::new()));
        let (seen, bodies) = (attempts.clone(), captured.clone());
        let (inflight_manager, inflight_target, inflight_worker) =
            (manager.clone(), target.clone(), worker.clone());
        let source_url = worker.url().to_owned();
        let app = axum::Router::new()
            .route("/get_server_info", axum::routing::get(move || {
                let source_url = source_url.clone();
                async move { Json(serde_json::json!({"model_id":"qwen3","instance_id":source_url,"kv_block_size":4})) }
            }))
            .route("/kv_event_sources", axum::routing::get(move || {
                let endpoint = endpoint.clone();
                async move { Json(serde_json::json!({"0":{"endpoint":endpoint,"publisher":"zmq","encoding":"msgpack","schema_version":1,"enable_kv_cache_events":true}})) }
            }))
            .route("/v1/completions", axum::routing::post(move |body: axum::body::Bytes| {
                let (seen, bodies, manager, target, worker) = (seen.clone(), bodies.clone(), inflight_manager.clone(), inflight_target.clone(), inflight_worker.clone());
                async move {
                    let request: serde_json::Value = serde_json::from_slice(&body).unwrap();
                    let prepared = request["model"] == "qwen3";
                    assert_eq!(manager.sample(&target), crate::policies::LoadSample {active_prefill_tokens:7 + u64::from(prepared),active_requests:2}, "only the configured alias has prepared input and placement credit");
                    assert_eq!(worker.load(), 2);
                    bodies.lock().push(body);
                    if seen.fetch_add(1, Ordering::Relaxed) % 2 == 0 {
                        StatusCode::INTERNAL_SERVER_ERROR.into_response()
                    } else {
                        Json(serde_json::json!({"choices":[{"text":"ok","finish_reason":"stop"}]})).into_response()
                    }
                }
            }));
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let (supervisor, mut signals) = crate::kv_index::KvIndexSupervisor::new_http_dense(
            Client::new(),
            crate::kv_index::HashMode::Sha256,
        );
        let supervisor = Arc::new(supervisor);
        let info = supervisor.on_worker_added(worker.url()).await.unwrap();
        kv_aware_ops(policy.as_ref()).unwrap().register_worker(
            worker.as_ref(),
            Some(&info),
            supervisor.match_provider_for_worker(worker.url()),
        );
        router.kv_index_supervisor = Some(supervisor.clone());
        assert_eq!(
            supervisor.dense_model_eligibility(&[worker.url()], "qwen3"),
            [true]
        );
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                match publisher.recv_bytes(zmq::DONTWAIT) {
                    Ok(subscription) => {
                        assert_eq!(subscription, [1]);
                        break;
                    }
                    Err(zmq::Error::EAGAIN) => tokio::task::yield_now().await,
                    Err(error) => panic!("subscription handshake: {error}"),
                }
            }
        })
        .await
        .expect("subscriber handshake");
        let event = KVEvent::BlockStored(BlockStored {
            extra_keys: None,
            block_hashes: vec![ExternalBlockHash::Int(1), ExternalBlockHash::Int(2)],
            parent_block_hash: None,
            token_ids: vec![1, 3, 4, 3, 4, 3, 4, 3],
            block_size: 4,
            lora_id: None,
            medium: Some("GPU".into()),
            lora_name: None,
            group_idx: Some(0),
            kv_cache_spec_kind: Some("full_attention".into()),
            kv_cache_spec_sliding_window: None,
            locality: Some("LOCAL".into()),
            ownership: None,
            session_id: None,
        });
        let payload = rmp_serde::to_vec_named(&(0.0, vec![event], Some(0u32))).unwrap();
        publisher
            .send_multipart(
                [
                    b"".as_slice(),
                    0i64.to_be_bytes().as_slice(),
                    payload.as_slice(),
                ],
                0,
            )
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                match signals.recv().await {
                    Some(IngestionSignal::Advance { last_seq: 0, .. }) => break,
                    Some(_) => {}
                    None => panic!("source signal channel closed"),
                }
            }
        })
        .await
        .expect("stored prefix applied");
        let receipt = kv
            .select_worker_with_receipt(
                std::slice::from_ref(&worker),
                Some(&[1, 3, 4, 3, 4, 3, 4, 3, 4]),
                None,
                None,
                None,
                &[],
            )
            .unwrap();
        assert_eq!(receipt.new_prefill_tokens, 1);
        drop(RequestLoadGuard::from_selection(
            worker.clone(),
            Some(receipt),
            true,
        ));
        manager.begin_prefill(&target, 7);
        worker.increment_load();
        for raw in [
            bytes::Bytes::from_static(b" { \"prompt\": \"hello world hello world hello world hello world\", \"model\":\"qwen3\", \"priority\":0 } "),
            bytes::Bytes::from_static(b"{\"model\":\"qwen3\", \"prompt\":[1,3,4,3,4,3,4,3,15]}\n"),
            bytes::Bytes::from_static(b"{\"model\":\"other-alias\",\"prompt\":\"hello world\"}"),
        ] {
            let request: CompletionRequest = serde_json::from_slice(&raw).unwrap();
            let response = tokio::time::timeout(Duration::from_secs(5), router.route_completion_raw(None, Arc::new(request), &raw, None)).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            to_bytes(response.into_body(), usize::MAX).await.unwrap();
            let bodies = captured.lock();
            assert_eq!(bodies[bodies.len()-2], raw); assert_eq!(bodies[bodies.len()-1], raw);
            drop(bodies);
            assert_eq!(encodes.load(Ordering::Relaxed), 1, "text prepared once across retries; IDs never encoded");
            assert_eq!(worker.load(), 1);
            assert_eq!(manager.sample(&target), crate::policies::LoadSample {active_prefill_tokens:7,active_requests:1});
        }
        assert_eq!(attempts.load(Ordering::Relaxed), 6);
        supervisor.shutdown();
        server.abort();
    }
    #[tokio::test]
    async fn completion_raw_forwarding_uses_effective_model_policy() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let bodies = captured.clone();
        let app = axum::Router::new().route(
            "/v1/completions",
            axum::routing::post(move |headers: HeaderMap, body: axum::body::Bytes| {
                let bodies = bodies.clone();
                async move {
                    bodies.lock().push((headers, body));
                    Json(serde_json::json!({"choices":[]}))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorker::new(
                format!("http://{}", listener.local_addr().unwrap()),
                WorkerType::Regular,
            )
            .with_labels(HashMap::from([("model_id".into(), "qwen3".into())])),
        );
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let raw = bytes::Bytes::from_static(
            b" { \"model\":\"qwen3\", \"prompt\":[3], \"custom\":null } ",
        );
        let request: CompletionRequest = serde_json::from_slice(&raw).unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/problem+json; charset=utf-8"),
        );
        for model_kv in [false, true] {
            let mut router = create_test_regular_router();
            router.worker_registry = Arc::new(WorkerRegistry::new());
            router.worker_registry.register(worker.clone());
            router.policy_registry = Arc::new(PolicyRegistry::new(if model_kv {
                crate::config::PolicyConfig::RoundRobin
            } else {
                crate::config::PolicyConfig::KvAware {
                    router_temperature: 0.0,
                    kv_penalty_host: 0.8,
                    kv_penalty_external: 0.6,
                    kv_penalty_disk: 0.3,
                    static_kv_event_sources: Default::default(),
                }
            }));
            router.policy_registry.on_worker_added(
                "qwen3",
                Some(if model_kv { "kv_aware" } else { "round_robin" }),
            );
            let response = router
                .route_completion_raw(
                    Some(&headers),
                    Arc::new(request.clone()),
                    &raw,
                    Some("qwen3"),
                )
                .await;
            assert_eq!(response.status(), StatusCode::OK);
            to_bytes(response.into_body(), usize::MAX).await.unwrap();
            let bodies = captured.lock();
            let (sent_headers, sent) = bodies.last().unwrap();
            let expected = if model_kv {
                raw.clone()
            } else {
                bytes::Bytes::from(serde_json::to_vec(&request).unwrap())
            };
            assert_eq!(sent, &expected);
            assert_eq!(
                sent_headers[CONTENT_TYPE],
                if model_kv {
                    "application/problem+json; charset=utf-8"
                } else {
                    "application/json"
                }
            );
        }
        server.abort();
    }

    #[tokio::test]
    async fn test_transparent_response_preserves_encoding_and_filters_connection() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = axum::Router::new().fallback(|| async {
            Response::builder()
                .header("content-encoding", "gzip")
                .header("connection", "X-Internal")
                .header("x-internal", "private")
                .header("set-cookie", "a=1")
                .header("set-cookie", "b=2")
                .body(Body::from(vec![0x1f, 0x8b, 0x08, 0x00, 0xff]))
                .unwrap()
        });
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let mut router = create_test_regular_router();
        router.worker_registry = Arc::new(WorkerRegistry::new());
        router.worker_registry.register(Arc::new(BasicWorker::new(
            format!("http://{address}"),
            WorkerType::Regular,
        )));
        let response = router
            .route_transparent(None, "/custom", &Method::GET, serde_json::Value::Null)
            .await;
        assert_eq!(response.headers().get("content-encoding").unwrap(), "gzip");
        assert!(!response.headers().contains_key("connection"));
        assert!(!response.headers().contains_key("x-internal"));
        assert_eq!(response.headers().get_all("set-cookie").iter().count(), 2);
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(bytes.as_ref(), &[0x1f, 0x8b, 0x08, 0x00, 0xff]);
        server.abort();
    }

    #[tokio::test]
    async fn test_inference_and_get_paths_strip_connection_headers() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let worker_url = format!("http://{address}");
        let app = axum::Router::new().fallback(|headers: HeaderMap| async move {
            let echoed: HashMap<String, String> = headers
                .iter()
                .map(|(name, value)| (name.to_string(), value.to_str().unwrap().to_string()))
                .collect();
            Json(echoed)
        });
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let mut router = create_test_regular_router();
        router.worker_registry = Arc::new(WorkerRegistry::new());
        router.worker_registry.register(Arc::new(BasicWorker::new(
            worker_url.clone(),
            WorkerType::Regular,
        )));
        let mut headers = HeaderMap::new();
        headers.append("connection", "X-Internal, traceparent".parse().unwrap());
        headers.append("connection", " X-Second".parse().unwrap());
        headers.insert("host", "client-controlled.example".parse().unwrap());
        headers.insert("x-internal", "private".parse().unwrap());
        headers.insert("x-second", "private".parse().unwrap());
        headers.insert("trailer", "x-checksum".parse().unwrap());
        headers.insert(
            "traceparent",
            "00-11111111111111111111111111111111-2222222222222222-01"
                .parse()
                .unwrap(),
        );
        headers.insert("x-public", "keep".parse().unwrap());
        headers.insert("authorization", "Bearer token".parse().unwrap());
        let request = Request::builder().body(Body::empty()).unwrap();
        let (mut parts, body) = request.into_parts();
        parts.headers = headers.clone();
        let get_response = router
            .proxy_get_request(Request::from_parts(parts, body), "v1/models")
            .await;
        let post_response = router
            .send_typed_request(
                &serde_json::json!({"model": "test", "messages": []}),
                TypedDispatch {
                    headers: Some(&headers),
                    route: "/v1/chat/completions",
                    worker_url: &worker_url,
                    is_stream: false,
                    completion: None,
                    // No worker is tracked, so this dispatch does not count against
                    // a worker's load — the same state the old `load_incremented:
                    // false` expressed before the field became a guard.
                    load: RequestLoadGuard::new(None, None),
                    prepared: None,
                },
                None,
            )
            .await;
        for response in [get_response, post_response] {
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
            let echoed: HashMap<String, String> = serde_json::from_slice(&bytes).unwrap();
            for name in [
                "connection",
                "x-internal",
                "x-second",
                "trailer",
                "traceparent",
            ] {
                assert!(
                    !echoed.contains_key(name),
                    "{name} leaked to the worker: {echoed:?}"
                );
            }
            assert_eq!(echoed["host"], address.to_string());
            assert_eq!(echoed["x-public"], "keep");
            assert_eq!(echoed["authorization"], "Bearer token");
        }
        server.abort();
    }

    struct HttpKvFixture {
        router: Router,
        workers: Vec<Arc<dyn Worker>>,
        servers: Vec<tokio::task::JoinHandle<()>>,
        calls: Arc<Mutex<Vec<(usize, String, serde_json::Value)>>>,
    }

    impl Drop for HttpKvFixture {
        fn drop(&mut self) {
            for server in &self.servers {
                server.abort();
            }
        }
    }

    async fn http_kv_fixture(
        inputs: [Vec<Vec<u32>>; 2],
        warm: usize,
        render_available: bool,
    ) -> HttpKvFixture {
        use crate::kv_index::{
            CacheKey, HashMode, KvBlockIndexer, ResidencyOwner, RoutableTarget, StorageTier,
        };
        use crate::policies::KvAwarePolicy;
        use axum::{extract::OriginalUri, routing::post};
        let mut router = create_test_regular_router();
        router.worker_registry = Arc::new(WorkerRegistry::new());
        router.policy_registry =
            Arc::new(PolicyRegistry::new(crate::config::PolicyConfig::KvAware {
                router_temperature: 0.0,
                kv_penalty_host: 0.8,
                kv_penalty_external: 0.6,
                kv_penalty_disk: 0.3,
                static_kv_event_sources: Default::default(),
            }));
        let index = Arc::new(KvBlockIndexer::new());
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut servers = Vec::new();
        let mut workers: Vec<Arc<dyn Worker>> = Vec::new();
        for (worker_index, prompts) in inputs.iter().cloned().enumerate() {
            let captured = calls.clone();
            let app = axum::Router::new().fallback(post(move |OriginalUri(uri): OriginalUri, Json(body): Json<serde_json::Value>| {
                let prompts = prompts.clone(); let captured = captured.clone();
                async move {
                    let path = uri.path().to_string();
                    captured.lock().push((worker_index, path.clone(), body));
                    if path.ends_with("/render") {
                        if prompts.is_empty() {
                            tokio::time::sleep(Duration::from_secs(3)).await;
                            return StatusCode::SERVICE_UNAVAILABLE.into_response();
                        }
                        if !render_available { return StatusCode::NOT_FOUND.into_response(); }
                        let values: Vec<_> = prompts.iter().map(|ids| serde_json::json!({"token_ids": ids, "features": null})).collect();
                        return Json(if path == "/v1/completions/render" { serde_json::json!(values) } else { values[0].clone() }).into_response();
                    }
                    if path == "/tokenize" { return Json(serde_json::json!({"tokens": prompts[0], "count": prompts[0].len()})).into_response(); }
                    Json(serde_json::json!({"worker": worker_index, "choices": []})).into_response()
                }
            }));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            servers.push(tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap()
            }));
            let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(url, WorkerType::Regular));
            router.worker_registry.register(worker.clone());
            let policy = router.policy_registry.get_default_policy();
            let policy = policy.as_any().downcast_ref::<KvAwarePolicy>().unwrap();
            let target = RoutableTarget {
                instance_id: Arc::from(format!("worker-{worker_index}")),
                dp_rank: 0,
            };
            policy.register_target_with_cache_identity(
                worker.url(),
                target.clone(),
                CacheKey {
                    model: Arc::from("model"),
                    hash_mode: HashMode::Sha256,
                    block_size: 2,
                },
                index.clone(),
            );
            if worker_index == warm {
                worker.increment_load();
                for (prompt_index, ids) in inputs[warm].iter().enumerate() {
                    let blocks: Vec<_> = KvAwarePolicy::local_hashes(&ids[..4], 2)
                        .into_iter()
                        .enumerate()
                        .map(|(block, hash)| (Arc::from(format!("{prompt_index}-{block}")), hash))
                        .collect();
                    index.store(
                        0,
                        ResidencyOwner::Worker {
                            source: target.instance_id.clone(),
                            dp_rank: 0,
                            incarnation: 0,
                        },
                        StorageTier::Device,
                        None,
                        &blocks,
                    );
                }
            }
            workers.push(worker);
        }
        HttpKvFixture {
            router,
            workers,
            servers,
            calls,
        }
    }

    async fn selected_http_worker(response: Response) -> usize {
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()["worker"]
            .as_u64()
            .unwrap() as usize
    }

    #[tokio::test]
    async fn http_completion_ids_use_real_prefix_placement() {
        let ids = vec![1, 2, 3, 4, 5, 6];
        let fixture = http_kv_fixture([vec![ids.clone()], vec![ids]], 0, true).await;
        for (prompt, expected) in [
            (serde_json::json!([1, 2, 3, 4, 5, 6]), 0),
            (serde_json::json!([11, 12, 13, 14, 15, 16]), 1),
            (serde_json::json!([1, 2, 3, 4, 99, 100]), 0),
            (
                serde_json::json!([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 7, 8]]),
                0,
            ),
        ] {
            let request: CompletionRequest =
                serde_json::from_value(serde_json::json!({"prompt":prompt})).unwrap();
            assert_eq!(
                selected_http_worker(
                    fixture
                        .router
                        .route_typed_request(None, &request, "/v1/completions", None)
                        .await
                )
                .await,
                expected
            );
        }
        assert!(fixture
            .calls
            .lock()
            .iter()
            .all(|(_, path, _)| !path.ends_with("/render") && path != "/tokenize"));
        assert_eq!(fixture.workers[0].load(), 1);
        assert_eq!(fixture.workers[1].load(), 0);
    }

    #[tokio::test]
    async fn completion_local_contract_retains_preflight_and_never_regains_render_credit() {
        let ids = vec![1, 2, 3, 4, 5, 6];
        let mut fixture = http_kv_fixture([vec![ids.clone()], vec![ids.clone()]], 0, true).await;
        fixture.router.kv_aware_session_affinity_ttl_secs = Some(1800);
        let encodes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        fixture.router.completion_input = Some(crate::backend::completion_input::CompletionInput {
            model: "local".into(),
            cache_model: "model".into(),
            vocab_size: 16,
            tokenizer: Arc::new(CompletionCountingTokenizer {
                inner: vllm_tokenizer::HuggingFaceTokenizer::new_hf(std::path::Path::new(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/rust/fixtures/tokenizer/completion_word_level.json"
                )))
                .unwrap(),
                encodes: encodes.clone(),
            }),
        });
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-session-id",
            HeaderValue::from_static("completion-session"),
        );
        let raw = bytes::Bytes::from_static(br#"{"model":"local","prompt":[-1]}"#);
        let invalid: CompletionRequest = serde_json::from_slice(&raw).unwrap();
        let key = AffinityKey::from_identity(
            resolve_affinity_identity(Some(&headers), invalid.affinity_identity())
                .unwrap()
                .unwrap(),
            "local",
        );
        let policy = fixture.router.policy_registry.get_default_policy();
        let kv = kv_aware_policy(policy.as_ref()).unwrap();
        let receipt = kv
            .select_worker_with_receipt(&fixture.workers, Some(&ids), None, None, Some(&key), &[])
            .unwrap();
        assert_eq!(receipt.worker_index, 0);
        drop(RequestLoadGuard::from_selection(
            fixture.workers[0].clone(),
            Some(receipt),
            true,
        ));
        assert!(kv_aware_ops(policy.as_ref()).unwrap().has_affinity(
            &fixture.workers,
            None,
            None,
            &key
        ));

        let response = fixture
            .router
            .route_completion_raw(Some(&headers), Arc::new(invalid), &raw, None)
            .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert!(std::str::from_utf8(&body)
            .unwrap()
            .contains("negative token ID -1"));
        assert!(
            fixture.calls.lock().is_empty(),
            "affinity cannot bypass preflight"
        );

        let raw = bytes::Bytes::from_static(
            br#" {"model":"local","prompt":[1,2,3,4,5,6],"suffix":"tail"} "#,
        );
        let response = fixture
            .router
            .route_completion_raw(
                None,
                Arc::new(serde_json::from_slice(&raw).unwrap()),
                &raw,
                None,
            )
            .await;
        assert_eq!(selected_http_worker(response).await, 1);
        let calls = fixture.calls.lock();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, 1);
        assert_eq!(calls[0].1, "/v1/completions");
        assert_eq!(
            calls[0].2,
            serde_json::from_slice::<serde_json::Value>(&raw).unwrap()
        );
        assert_eq!(encodes.load(Ordering::Relaxed), 0);
        assert_eq!(
            (fixture.workers[0].load(), fixture.workers[1].load()),
            (1, 0)
        );
        for worker in &fixture.workers {
            assert_eq!(
                crate::policies::LoadProvider::sample(
                    kv.slot_manager().unwrap().as_ref(),
                    &kv.target_for_worker(worker.as_ref()),
                ),
                crate::policies::LoadSample::default(),
            );
        }
    }

    #[tokio::test]
    async fn http_affinity_hits_skip_render_on_typed_and_transparent_paths() {
        for transparent in [false, true] {
            let ids = vec![1, 2, 3, 4, 5, 6];
            let mut fixture = http_kv_fixture([vec![ids.clone()], vec![ids]], 0, true).await;
            fixture.router.kv_aware_session_affinity_ttl_secs = Some(1800);
            let body = serde_json::json!({"messages":[{"role":"user","content":"hello"}], "session_params":{"session_id":"session"}});
            let request: ChatCompletionRequest = serde_json::from_value(body.clone()).unwrap();
            for turn in 0..2 {
                let response = if transparent {
                    fixture
                        .router
                        .route_transparent(
                            None,
                            "/v1/chat/completions",
                            &Method::POST,
                            body.clone(),
                        )
                        .await
                } else {
                    fixture
                        .router
                        .route_typed_request(None, &request, "/v1/chat/completions", None)
                        .await
                };
                assert_eq!(selected_http_worker(response).await, 0);
                let calls = fixture.calls.lock();
                assert_eq!(
                    calls
                        .iter()
                        .filter(|(_, path, _)| path.ends_with("/render"))
                        .count(),
                    if turn == 0 { 2 } else { 0 }
                );
                drop(calls);
                fixture.calls.lock().clear();
                if turn == 0 {
                    for _ in 0..8 {
                        fixture.workers[0].increment_load();
                    }
                }
            }
            // An unavailable binding must return to preparation and fresh selection.
            fixture.workers[0].set_healthy(false);
            let response = fixture
                .router
                .route_typed_request(None, &request, "/v1/chat/completions", None)
                .await;
            assert_eq!(selected_http_worker(response).await, 1);
            assert_eq!(
                fixture
                    .calls
                    .lock()
                    .iter()
                    .filter(|(_, path, _)| path.ends_with("/render"))
                    .count(),
                1
            );
        }
    }

    #[tokio::test]
    async fn http_chat_shapes_preserve_full_render_and_dispatch_payloads() {
        let ids = vec![1, 2, 3, 4, 5, 6];
        let fixture = http_kv_fixture([vec![ids.clone()], vec![ids]], 0, true).await;
        let requests = [
            serde_json::json!({"messages":[{"role":"user","content":"hello"}]}),
            serde_json::json!({"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}],"stream":true}),
            serde_json::json!({"messages":[{"role":"user","content":"call f"},{"role":"assistant","content":null,"tool_calls":[{"id":"call-1","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call-1","content":"done"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"tool_choice":"auto","chat_template_kwargs":{"enable_thinking":false},"response_format":{"type":"json_object"},"reasoning_effort":"low"}),
        ];
        for body in requests {
            let request: ChatCompletionRequest = serde_json::from_value(body).unwrap();
            let forwarded = serde_json::to_value(&request).unwrap();
            assert_eq!(
                selected_http_worker(
                    fixture
                        .router
                        .route_typed_request(None, &request, "/v1/chat/completions", None)
                        .await
                )
                .await,
                0
            );
            let calls = fixture.calls.lock();
            for (_, _, payload) in calls.iter().rev().take(3) {
                assert_eq!(*payload, forwarded);
            }
        }
        assert_eq!(fixture.workers[0].load(), 1);
    }

    #[tokio::test]
    async fn http_render_uses_each_workers_own_tokens_and_batch_prefixes() {
        let a = vec![1, 2, 3, 4, 5, 6];
        let b = vec![11, 12, 13, 14, 15, 16];
        let fixture = http_kv_fixture([vec![a.clone(), a], vec![b.clone(), b]], 1, true).await;
        let request: CompletionRequest =
            serde_json::from_value(serde_json::json!({"prompt":["one","two"]})).unwrap();
        assert_eq!(
            selected_http_worker(
                fixture
                    .router
                    .route_typed_request(None, &request, "/v1/completions", None)
                    .await
            )
            .await,
            1
        );
        assert_eq!(fixture.workers[1].load(), 1);
    }

    #[tokio::test]
    async fn http_preparation_fallback_and_cache_salt_do_not_claim_hits() {
        let ids = vec![1, 2, 3, 4, 5, 6];
        let fixture = http_kv_fixture([vec![ids.clone()], vec![ids]], 0, false).await;
        let request: CompletionRequest =
            serde_json::from_value(serde_json::json!({"prompt":"hello"})).unwrap();
        assert_eq!(
            selected_http_worker(
                fixture
                    .router
                    .route_typed_request(None, &request, "/v1/completions", None)
                    .await
            )
            .await,
            0
        );
        let request: CompletionRequest = serde_json::from_value(
            serde_json::json!({"prompt":[1,2,3,4,5,6],"cache_salt":"private"}),
        )
        .unwrap();
        assert_eq!(
            selected_http_worker(
                fixture
                    .router
                    .route_typed_request(None, &request, "/v1/completions", None)
                    .await
            )
            .await,
            1
        );
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({"messages":[{"role":"user","content":"hello"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"tool_choice":"auto"})).unwrap();
        assert_eq!(
            selected_http_worker(
                fixture
                    .router
                    .route_typed_request(None, &request, "/v1/chat/completions", None)
                    .await
            )
            .await,
            1
        );
    }

    #[tokio::test]
    async fn failed_or_slow_render_keeps_other_workers_placement() {
        for slow in [false, true] {
            let ids = vec![1, 2, 3, 4, 5, 6];
            let cold = if slow { vec![] } else { vec![ids.clone()] };
            let fixture = http_kv_fixture([vec![ids], cold], 0, true).await;
            if !slow {
                fixture.servers[1].abort();
            }
            let request: CompletionRequest =
                serde_json::from_value(serde_json::json!({"prompt":"hello"})).unwrap();
            let response = tokio::time::timeout(
                Duration::from_secs(2),
                fixture
                    .router
                    .route_typed_request(None, &request, "/v1/completions", None),
            )
            .await
            .expect("optional rendering must not wait for the slow worker");
            assert_eq!(selected_http_worker(response).await, 0);
        }
    }

    #[tokio::test]
    async fn http_preparation_only_contacts_the_selected_model_pool() {
        for transparent in [false, true] {
            let ids = vec![1, 2, 3, 4, 5, 6];
            let fixture = http_kv_fixture([vec![ids.clone()], vec![ids]], 0, true).await;
            fixture
                .router
                .worker_registry
                .remove_by_url(fixture.workers[1].url());
            fixture.router.worker_registry.register(Arc::new(
                BasicWorker::new(fixture.workers[1].url().into(), WorkerType::Regular)
                    .with_labels(HashMap::from([("model_id".into(), "other-model".into())])),
            ));
            let response = if transparent {
                fixture
                    .router
                    .route_transparent(
                        None,
                        "/v1/responses",
                        &Method::POST,
                        serde_json::json!({"model":fixture.workers[0].model_id(),"input":"hello"}),
                    )
                    .await
            } else {
                let request: CompletionRequest = serde_json::from_value(
                    serde_json::json!({"model":fixture.workers[0].model_id(), "prompt":"hello"}),
                )
                .unwrap();
                fixture
                    .router
                    .route_typed_request(None, &request, "/v1/completions", None)
                    .await
            };
            assert_eq!(selected_http_worker(response).await, 0);
            assert!(fixture
                .calls
                .lock()
                .iter()
                .all(|(worker, _, _)| *worker == 0));
        }
    }

    #[tokio::test]
    async fn transparent_responses_and_native_tokens_use_prefix_placement() {
        let ids = vec![1, 2, 3, 4, 5, 6];
        let fixture = http_kv_fixture([vec![ids.clone()], vec![ids]], 0, true).await;
        for (path, body, expected) in [
            (
                "/v1/responses",
                serde_json::json!({"model":"served-model","input":"hello"}),
                0,
            ),
            (
                "/inference/v1/generate",
                serde_json::json!({"token_ids":[1,2,3,4,5,6]}),
                0,
            ),
            (
                "/generate",
                serde_json::json!({"input_ids":[1,2,3,4,5,6]}),
                0,
            ),
            (
                "/v1/responses",
                serde_json::json!({"input":"hello","previous_response_id":"previous"}),
                1,
            ),
        ] {
            assert_eq!(
                selected_http_worker(
                    fixture
                        .router
                        .route_transparent(None, path, &Method::POST, body)
                        .await
                )
                .await,
                expected
            );
        }
        assert_eq!(fixture.workers[0].load(), 1);
    }

    #[tokio::test]
    async fn transparent_completion_uses_prefix_receipt_and_releases_load() {
        let ids = vec![1, 2, 3, 4, 5, 6];
        let fixture = http_kv_fixture([vec![ids.clone()], vec![ids]], 0, true).await;
        assert_eq!(
            selected_http_worker(
                fixture
                    .router
                    .route_transparent(
                        None,
                        "/v1/completions",
                        &Method::POST,
                        serde_json::json!({"prompt":[1,2,3,4,5,6]})
                    )
                    .await
            )
            .await,
            0
        );
        assert_eq!(fixture.workers[0].load(), 1);
    }

    fn create_test_regular_router() -> Router {
        // Create registries
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(
            crate::config::types::PolicyConfig::RoundRobin,
        ));

        // Register test workers
        let worker1 = BasicWorker::new("http://worker1:8080".to_string(), WorkerType::Regular);
        let worker2 = BasicWorker::new("http://worker2:8080".to_string(), WorkerType::Regular);
        worker_registry.register(Arc::new(worker1));
        worker_registry.register(Arc::new(worker2));

        let (_, rx) = tokio::sync::watch::channel(HashMap::new());
        Router {
            epd: None,
            worker_registry,
            policy_registry,
            worker_startup_timeout_secs: 5,
            worker_startup_check_interval_secs: 1,
            intra_node_data_parallel_size: 1,
            api_key: None,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            health_config: HealthConfig::default(),
            frontend: crate::backend::EngineFrontend::new(),
            completion_input: None,
            completion_preparation: crate::backend::completion_input::CompletionPreparation::new(
                64,
                Duration::from_secs(30),
                Duration::from_secs(5),
            ),
            _worker_loads: Arc::new(rx),
            _load_monitor_handle: None,
            kv_index_supervisor: None,
            _kv_signal_handle: None,
            kv_aware_session_affinity_ttl_secs: None,
            program_scheduler: None,
            _program_observation_handle: None,
            program_targets_cache: Mutex::new(HashMap::new()),
            program_token_estimator: Arc::new(MomentumTokenEstimator::default()),
        }
    }

    fn create_test_grpc_router() -> Router {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(
            crate::config::types::PolicyConfig::RoundRobin,
        ));

        let worker = BasicWorker::new("grpc://127.0.0.1:15002".to_string(), WorkerType::Regular);
        worker_registry.register(Arc::new(worker));

        let (_, rx) = tokio::sync::watch::channel(HashMap::new());
        Router {
            worker_registry,
            policy_registry,
            worker_startup_timeout_secs: 5,
            worker_startup_check_interval_secs: 1,
            intra_node_data_parallel_size: 1,
            api_key: None,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            health_config: HealthConfig::default(),
            frontend: crate::backend::EngineFrontend::new(),
            completion_input: None,
            completion_preparation: crate::backend::completion_input::CompletionPreparation::new(
                64,
                Duration::from_secs(30),
                Duration::from_secs(5),
            ),
            epd: None,
            _worker_loads: Arc::new(rx),
            _load_monitor_handle: None,
            kv_index_supervisor: None,
            _kv_signal_handle: None,
            kv_aware_session_affinity_ttl_secs: None,
            program_scheduler: None,
            _program_observation_handle: None,
            program_targets_cache: Mutex::new(HashMap::new()),
            program_token_estimator: Arc::new(MomentumTokenEstimator::default()),
        }
    }

    #[test]
    fn program_targets_cache_tracks_worker_registry_revision() {
        let router = create_test_regular_router();
        let all_pool = router.resolved_program_model_pool(None);
        let first = router.program_targets_for_model(&all_pool);
        let second = router.program_targets_for_model(&all_pool);
        assert!(Arc::ptr_eq(&first, &second));

        let first_missing_pool = router.resolved_program_model_pool(Some("missing-a"));
        let second_missing_pool = router.resolved_program_model_pool(Some("missing-b"));
        assert_eq!(
            first_missing_pool.scheduler_key,
            PROGRAM_MODEL_POOL_FALLBACK
        );
        assert_eq!(first_missing_pool, second_missing_pool);
        let first_miss = router.program_targets_for_model(&first_missing_pool);
        let second_miss = router.program_targets_for_model(&second_missing_pool);
        assert!(Arc::ptr_eq(&first_miss, &second_miss));
        assert_eq!(first_miss.len(), 2);

        let literal_unknown = router.resolved_program_model_pool(Some("unknown"));
        assert_eq!(
            literal_unknown.scheduler_key,
            format!("{PROGRAM_MODEL_POOL_NAMED_PREFIX}unknown")
        );
        assert_ne!(
            literal_unknown.scheduler_key,
            first_missing_pool.scheduler_key
        );
        assert_eq!(router.program_targets_for_model(&literal_unknown).len(), 2);

        router.worker_registry.register(Arc::new(BasicWorker::new(
            "http://worker3:8080".to_string(),
            WorkerType::Regular,
        )));
        let third = router.program_targets_for_model(&all_pool);
        assert!(!Arc::ptr_eq(&first, &third));
        assert_eq!(third.len(), 3);
    }

    #[test]
    fn observation_refreshes_tracked_pools_without_request_arrival() {
        let registry = WorkerRegistry::new();
        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), "model-a".to_string());
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorker::new("http://worker-a:8080".to_string(), WorkerType::Regular)
                .with_labels(labels),
        );
        registry.register(worker.clone());

        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig::default());
        let named_model_a = format!("{PROGRAM_MODEL_POOL_NAMED_PREFIX}model-a");
        scheduler.sync_targets(PROGRAM_MODEL_POOL_ALL, &[]);
        scheduler.sync_targets(&named_model_a, &[]);
        scheduler.sync_targets(PROGRAM_MODEL_POOL_FALLBACK, &[]);
        Router::sync_program_targets_from_registry(&scheduler, &registry);

        assert_eq!(scheduler.targets(PROGRAM_MODEL_POOL_ALL).len(), 1);
        assert_eq!(scheduler.targets(&named_model_a).len(), 1);
        assert!(scheduler.targets(PROGRAM_MODEL_POOL_FALLBACK).is_empty());

        worker.set_healthy(false);
        registry.notify_worker_state_change();
        Router::sync_program_targets_from_registry(&scheduler, &registry);
        assert!(scheduler.targets(PROGRAM_MODEL_POOL_ALL).is_empty());
        assert!(scheduler.targets(&named_model_a).is_empty());

        registry.remove_by_url("http://worker-a:8080");
        Router::sync_program_targets_from_registry(&scheduler, &registry);
        assert!(scheduler.all_targets().is_empty());
    }

    #[test]
    fn observation_refresh_retries_registry_change_across_install_boundary() {
        let registry = WorkerRegistry::new();
        let worker_url = "http://worker-a:8080";
        registry.register(Arc::new(BasicWorker::new(
            worker_url.to_string(),
            WorkerType::Regular,
        )));

        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig::default());
        scheduler.sync_targets(PROGRAM_MODEL_POOL_ALL, &[]);

        let mut removed = false;
        Router::sync_program_targets_from_registry_with_hook(&scheduler, &registry, || {
            if !removed {
                registry.remove_by_url(worker_url);
                removed = true;
            }
        });

        assert!(removed);
        assert!(scheduler.targets(PROGRAM_MODEL_POOL_ALL).is_empty());
        assert!(scheduler.all_targets().is_empty());
    }

    #[test]
    fn test_router_get_worker_urls_regular() {
        let router = create_test_regular_router();
        let urls = router.get_worker_urls();

        assert_eq!(urls.len(), 2);
        assert!(urls.contains(&"http://worker1:8080".to_string()));
        assert!(urls.contains(&"http://worker2:8080".to_string()));
    }

    #[test]
    fn test_select_first_worker_regular() {
        let router = create_test_regular_router();
        let result = router.select_first_worker();

        assert!(result.is_ok());
        let url = result.unwrap();
        // DashMap doesn't guarantee order, so just check we get one of the workers
        assert!(url == "http://worker1:8080" || url == "http://worker2:8080");
    }

    #[test]
    fn test_select_first_worker_skips_unhealthy() {
        let router = create_test_regular_router();
        // Mark worker1 unhealthy; select_first_worker must not return it.
        for w in router.worker_registry.get_all() {
            if w.url() == "http://worker1:8080" {
                w.set_healthy(false);
            }
        }
        let result = router.select_first_worker();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "http://worker2:8080");
    }

    #[test]
    fn test_select_first_worker_skips_circuit_open() {
        let router = create_test_regular_router();
        // Worker1 stays healthy but its circuit is forced open.
        for w in router.worker_registry.get_all() {
            if w.url() == "http://worker1:8080" {
                for _ in 0..5 {
                    w.record_outcome(false);
                }
                assert!(w.is_healthy()); // Still healthy
                assert!(!w.is_available()); // But circuit is open
            }
        }
        let result = router.select_first_worker();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "http://worker2:8080");
    }

    #[test]
    fn test_select_first_worker_all_unhealthy() {
        let router = create_test_regular_router();
        for w in router.worker_registry.get_all() {
            w.set_healthy(false);
        }
        let result = router.select_first_worker();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_grpc_health_generate_is_explicitly_not_implemented() {
        let router = create_test_grpc_router();
        let response = router.health_generate(Request::new(Body::empty())).await;
        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body = std::str::from_utf8(&body).unwrap();
        assert!(body.contains("not implemented for gRPC workers"));
    }

    #[tokio::test]
    async fn test_add_worker_rejects_mixed_scheme() {
        let router = create_test_regular_router();
        let err = router
            .add_worker("grpc://127.0.0.1:50051")
            .await
            .unwrap_err();
        assert!(err.contains("mixed"), "{err}");
    }

    /// An axum backend that echoes the header names and values it received, so a
    /// test can assert on what actually reached the worker.
    async fn spawn_header_echo_backend() -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
        use axum::Json;
        let app = axum::Router::new().fallback(|headers: HeaderMap| async move {
            let mut echoed: HashMap<String, Vec<String>> = HashMap::new();
            for (name, value) in headers.iter() {
                if let Ok(value) = value.to_str() {
                    echoed
                        .entry(name.to_string())
                        .or_default()
                        .push(value.to_string());
                }
            }
            Json(echoed)
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (address, server)
    }

    fn router_pointing_at(address: std::net::SocketAddr) -> Router {
        let mut router = create_test_regular_router();
        router.worker_registry = Arc::new(WorkerRegistry::new());
        router.worker_registry.register(Arc::new(BasicWorker::new(
            format!("http://{address}"),
            WorkerType::Regular,
        )));
        router
    }

    /// The transparent path with EPD enabled has to apply the same header policy
    /// as every other forwarding path. Its own skip list only covered
    /// host/content-length/content-type/connection/transfer-encoding, so
    /// `keep-alive`, `te`, `upgrade`, `proxy-authorization` and whatever the
    /// client's `Connection` named still went through to the worker.
    #[tokio::test]
    async fn transparent_epd_path_sanitizes_client_headers() {
        let (address, server) = spawn_header_echo_backend().await;
        let mut router = router_pointing_at(address);
        router.epd = Some(super::super::epd::EncoderStage::new(
            crate::config::EpdConfig {
                encoder_urls: vec!["http://unused".into()],
                consumer_zmq_addrs: Default::default(),
            },
        ));
        let mut headers = HeaderMap::new();
        headers.insert("keep-alive", "timeout=5".parse().unwrap());
        headers.insert("te", "trailers".parse().unwrap());
        headers.insert("upgrade", "websocket".parse().unwrap());
        headers.insert("proxy-authorization", "Basic c2VjcmV0".parse().unwrap());
        headers.insert("connection", "X-Internal".parse().unwrap());
        headers.insert("x-internal", "must-not-leak".parse().unwrap());
        headers.insert("x-session-id", "sticky-1".parse().unwrap());

        // /v1/responses is not handed to the encoder, so this exercises the
        // forwarding loop itself.
        let response = router
            .route_transparent(
                Some(&headers),
                "/v1/responses",
                &Method::POST,
                serde_json::json!({"model": "m"}),
            )
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let echoed: HashMap<String, Vec<String>> = serde_json::from_slice(&bytes).unwrap();
        for name in [
            "keep-alive",
            "te",
            "upgrade",
            "proxy-authorization",
            "connection",
            "x-internal",
        ] {
            assert!(
                !echoed.contains_key(name),
                "{name} reached the worker: {echoed:?}"
            );
        }
        assert_eq!(echoed["x-session-id"], vec!["sticky-1"]);
        server.abort();
    }

    #[tokio::test]
    async fn transparent_epd_records_request_metrics() {
        use axum::{routing::post, Json};
        use serde_json::{json, Value};

        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _metrics = metrics::set_default_local_recorder(&recorder);
        let app = axum::Router::new().route(
            "/v1/chat/completions",
            post(|Json(body): Json<Value>| async move {
                if body["fail"] == true {
                    (StatusCode::BAD_REQUEST, "invalid request").into_response()
                } else if body["stream"] == true {
                    ([("content-type", "text/event-stream")], "data: [DONE]\n\n").into_response()
                } else {
                    Json(json!({"choices": []})).into_response()
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let mut router = create_test_regular_router();
        router.worker_registry = Arc::new(WorkerRegistry::new());
        router
            .worker_registry
            .register(Arc::new(BasicWorker::new(url, WorkerType::Regular)));
        router.epd = Some(super::super::epd::EncoderStage::new(
            crate::config::EpdConfig {
                encoder_urls: vec!["http://unused".into()],
                consumer_zmq_addrs: Default::default(),
            },
        ));
        for (body, status) in [
            (json!({"messages": []}), StatusCode::OK),
            (json!({"messages": [], "stream": true}), StatusCode::OK),
            (
                json!({"messages": [], "fail": true}),
                StatusCode::BAD_REQUEST,
            ),
            (json!({"messages": null}), StatusCode::BAD_REQUEST),
        ] {
            let response = router
                .route_transparent(None, "/v1/chat/completions", &Method::POST, body)
                .await;
            assert_eq!(response.status(), status);
            axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap();
        }
        let exported = handle.render();
        assert!(exported.contains("vllm_router_requests_total{route=\"/v1/chat/completions\"} 2"));
        assert!(exported.contains("vllm_router_generate_duration_seconds_count 2"));
        for error in ["encoder_error", "backend_error"] {
            assert!(
                exported
                    .lines()
                    .any(|line| line.starts_with("vllm_router_request_errors_total{")
                        && line.contains(&format!("error_type=\"{error}\""))
                        && line.ends_with(" 1")),
                "{exported}"
            );
        }
        server.abort();
    }

    #[tokio::test]
    async fn grpc_completion_does_not_use_http_input_contract() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            format!("grpc://{}", listener.local_addr().unwrap()),
            WorkerType::Regular,
        ));
        drop(listener);
        let mut router = create_test_regular_router();
        router.worker_registry = Arc::new(WorkerRegistry::new());
        router.worker_registry.register(worker.clone());
        router.policy_registry =
            Arc::new(PolicyRegistry::new(crate::config::PolicyConfig::KvAware {
                router_temperature: 0.0,
                kv_penalty_host: 0.8,
                kv_penalty_external: 0.6,
                kv_penalty_disk: 0.3,
                static_kv_event_sources: Default::default(),
            }));
        router.retry_config.max_retries = 0;
        router.pin_test_token_ids(vec![9]);
        let encodes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        router.completion_input = Some(crate::backend::completion_input::CompletionInput {
            model: "qwen3".into(),
            cache_model: "qwen3".into(),
            vocab_size: 0,
            tokenizer: Arc::new(CompletionCountingTokenizer {
                inner: vllm_tokenizer::HuggingFaceTokenizer::new_hf(
                    &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("tests/rust/fixtures/tokenizer/completion_word_level.json"),
                )
                .unwrap(),
                encodes: encodes.clone(),
            }),
        });
        let raw = bytes::Bytes::from_static(b"{\"model\":\"qwen3\",\"prompt\":[1,2,3]}");
        let request = serde_json::from_slice(&raw).unwrap();
        let response = tokio::time::timeout(
            Duration::from_secs(5),
            router.route_completion_raw(None, Arc::new(request), &raw, None),
        )
        .await
        .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        drop(response);
        assert_eq!(encodes.load(Ordering::Relaxed), 0);
        assert_eq!(worker.load(), 0);
    }

    #[tokio::test]
    async fn unsupported_grpc_route_does_not_leak_tracked_load() {
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "grpc://127.0.0.1:1".into(),
            WorkerType::Regular,
        ));
        let mut router = create_test_regular_router();
        router.worker_registry = Arc::new(WorkerRegistry::new());
        router.worker_registry.register(worker.clone());
        router.policy_registry =
            Arc::new(PolicyRegistry::new(crate::config::PolicyConfig::KvAware {
                router_temperature: 0.0,
                kv_penalty_host: 0.8,
                kv_penalty_external: 0.6,
                kv_penalty_disk: 0.3,
                static_kv_event_sources: Default::default(),
            }));
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
        for _ in 0..3 {
            let response = router
                .route_typed_request(None, &request, "/v1/completions", None)
                .await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert_eq!(worker.load(), 0);
        }
    }

    #[tokio::test]
    async fn cancelled_transparent_backend_header_wait_releases_worker_load() {
        use axum::routing::post;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            format!("http://{}", listener.local_addr().unwrap()),
            WorkerType::Regular,
        ));
        let request_started = Arc::new(tokio::sync::Notify::new());
        let handler_started = request_started.clone();
        let app = axum::Router::new().route(
            "/v1/chat/completions",
            post(move || {
                let request_started = handler_started.clone();
                async move {
                    request_started.notify_one();
                    std::future::pending::<Response>().await
                }
            }),
        );
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let mut router = create_test_regular_router();
        router.worker_registry = Arc::new(WorkerRegistry::new());
        router.worker_registry.register(worker.clone());
        router.policy_registry =
            Arc::new(PolicyRegistry::new(crate::config::PolicyConfig::KvAware {
                router_temperature: 0.0,
                kv_penalty_host: 0.8,
                kv_penalty_external: 0.6,
                kv_penalty_disk: 0.3,
                static_kv_event_sources: Default::default(),
            }));

        let request = serde_json::json!({"messages": []});
        let dispatch = tokio::spawn(async move {
            router
                .route_transparent(None, "/v1/chat/completions", &Method::POST, request)
                .await
        });
        request_started.notified().await;
        assert_eq!(
            worker.load(),
            1,
            "admitted request must count while awaiting headers"
        );

        dispatch.abort();
        assert!(dispatch.await.unwrap_err().is_cancelled());
        assert_eq!(
            worker.load(),
            0,
            "cancellation must release the admission load"
        );
        server.abort();
    }

    #[tokio::test]
    async fn streaming_body_holds_load_until_consumed_or_dropped() {
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "grpc://worker:50051".to_string(),
            WorkerType::Regular,
        ));

        worker.increment_load();
        let response =
            hold_load_until_body_done(Response::new(Body::from("complete")), worker.clone(), None);
        assert_eq!(worker.load(), 1);
        let _ = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(worker.load(), 0);

        worker.increment_load();
        let response =
            hold_load_until_body_done(Response::new(Body::from("cancelled")), worker.clone(), None);
        assert_eq!(worker.load(), 1);
        drop(response);
        assert_eq!(worker.load(), 0);
    }

    fn admitted_kv_load(
        worker: Arc<dyn Worker>,
        manager: Arc<crate::policies::InMemorySlotManager>,
        target: crate::kv_index::RoutableTarget,
    ) -> RequestLoadGuard {
        crate::policies::SlotManager::begin_prefill(manager.as_ref(), &target, 64);
        worker.increment_load();
        RequestLoadGuard::new(
            Some(worker),
            Some(KvRequestSlot::new(manager, target, 64, None)),
        )
    }

    #[tokio::test]
    async fn unsupported_completion_stream_keeps_raw_body_and_output_lifecycle() {
        assert_unsupported_completion_stream(None).await;
    }

    async fn assert_unsupported_completion_stream(capacity: Option<HeldCompletionCapacity>) {
        for echo in [false, true] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
                format!("http://{}", listener.local_addr().unwrap()),
                WorkerType::Regular,
            ));
            let mut router = create_test_regular_router();
            let held_capacity = capacity.map(|capacity| capacity.hold(&mut router));
            router.worker_registry = Arc::new(WorkerRegistry::new());
            router.worker_registry.register(worker.clone());
            router.policy_registry =
                Arc::new(PolicyRegistry::new(crate::config::PolicyConfig::KvAware {
                    router_temperature: 0.0,
                    kv_penalty_host: 0.8,
                    kv_penalty_external: 0.6,
                    kv_penalty_disk: 0.3,
                    static_kv_event_sources: Default::default(),
                }));
            let encodes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            router.completion_input = Some(crate::backend::completion_input::CompletionInput {
                model: "qwen3".into(),
                cache_model: "qwen3".into(),
                vocab_size: 16,
                tokenizer: Arc::new(CompletionCountingTokenizer {
                    inner: vllm_tokenizer::HuggingFaceTokenizer::new_hf(
                        &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                            .join("tests/rust/fixtures/tokenizer/completion_word_level.json"),
                    )
                    .unwrap(),
                    encodes: encodes.clone(),
                }),
            });
            let policy = router.policy_registry.get_default_policy();
            assert_eq!(policy.name(), "kv_aware");
            let kv = policy
                .as_any()
                .downcast_ref::<crate::policies::KvAwarePolicy>()
                .unwrap();
            let manager = kv.slot_manager().unwrap();
            let target = kv.target_for_worker(worker.as_ref());
            let raw = bytes::Bytes::from(format!(
                " {{ \"model\": \"qwen3\", \"prompt\": \"hello world\", \"cache_salt\": \"tenant\", \"stream\": true, \"echo\": {echo} }}\n"
            ));
            let request: CompletionRequest = serde_json::from_slice(&raw).unwrap();
            let input = router.completion_input.as_ref().unwrap();
            assert_eq!(request.model.as_deref(), Some(input.model.as_str()));
            assert_eq!((input.model.as_str(), input.vocab_size), ("qwen3", 16));
            let (tx, rx) =
                tokio::sync::mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();
            let receiver = Arc::new(Mutex::new(Some(rx)));
            let expected = raw.clone();
            let app = axum::Router::new().route(
                "/v1/completions",
                axum::routing::post(move |body: axum::body::Bytes| {
                    assert_eq!(body, expected);
                    let rx = receiver.lock().take().unwrap();
                    async move { Response::new(Body::from_stream(UnboundedReceiverStream::new(rx))) }
                }),
            );
            let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let response = tokio::time::timeout(
                Duration::from_secs(5),
                router.route_completion_raw(None, Arc::new(request), &raw, None),
            )
            .await
            .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let mut body = response.into_body().into_data_stream();
            for chunk in [
                b": comment containing data: [DONE]\n\n".as_slice(),
                b"data: {\"choices\":[{\"text\":\"output or echo\"}]}\n\n".as_slice(),
            ] {
                tx.send(Ok(bytes::Bytes::from_static(chunk))).unwrap();
                let received = tokio::time::timeout(Duration::from_secs(5), body.next())
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap();
                assert!(received.ends_with(chunk));
                assert_eq!(
                    worker.load(),
                    1,
                    "fallback must retain Completion observation"
                );
                assert_eq!(
                    manager.sample(&target),
                    crate::policies::LoadSample {
                        active_prefill_tokens: 0,
                        active_requests: 1,
                    },
                    "Unsupported has no token credit, but its request remains active"
                );
            }
            drop(body);
            assert_eq!(worker.load(), 0);
            assert_eq!(
                manager.sample(&target),
                crate::policies::LoadSample::default()
            );
            assert_eq!(encodes.load(Ordering::Relaxed), 0);
            if let Some(held) = held_capacity {
                assert_eq!(held.semaphore().available_permits(), 0);
                assert_eq!(
                    router.completion_preparation.available_admission_permits(),
                    usize::from(matches!(capacity, Some(HeldCompletionCapacity::Cpu)))
                );
            }
            server.abort();
        }
    }

    #[tokio::test]
    async fn http_completion_stream_requires_complete_model_output() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            format!("http://{}", listener.local_addr().unwrap()),
            WorkerType::Regular,
        ));
        let (tx, rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();
        let receiver = Arc::new(Mutex::new(Some(rx)));
        let app = axum::Router::new().route(
            "/v1/completions",
            axum::routing::post(move || {
                let rx = receiver.lock().take().unwrap();
                async move { Response::new(Body::from_stream(UnboundedReceiverStream::new(rx))) }
            }),
        );
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let manager = Arc::new(crate::policies::InMemorySlotManager::new());
        let target = crate::kv_index::RoutableTarget {
            instance_id: Arc::from(worker.url()),
            dp_rank: 0,
        };
        crate::policies::SlotManager::begin_prefill(manager.as_ref(), &target, 7);
        worker.increment_load();
        let unrelated = RequestLoadGuard::new(
            Some(worker.clone()),
            Some(KvRequestSlot::new(manager.clone(), target.clone(), 7, None)),
        );
        let load = admitted_kv_load(worker.clone(), manager.clone(), target.clone());
        let request: CompletionRequest = serde_json::from_value(serde_json::json!({
            "prompt": [1], "stream": true, "echo": false,
        }))
        .unwrap();
        let completion = CompletionRoute {
            tokens: None,
            input_active: false,
            raw_body: None,
            echo: request.echo,
        };
        let router = create_test_regular_router();
        let response = router
            .send_typed_request(
                &request,
                TypedDispatch {
                    headers: None,
                    route: "/v1/completions",
                    worker_url: worker.url(),
                    is_stream: true,
                    completion: Some(&completion),
                    load,
                    prepared: None,
                },
                None,
            )
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let mut body = response.into_body().into_data_stream();
        for chunk in [
            ": heartbeat\r\n\r\n",
            "\n\n",
            "data: {\"choices\":[{\"text\":\"\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"completion_tokens\":0}}\n\n",
            "data: {\"choices\":[{\"text\":\"generated",
        ] {
            tx.send(Ok(bytes::Bytes::from_static(chunk.as_bytes())))
                .unwrap();
            let bytes = tokio::time::timeout(Duration::from_secs(5), body.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            assert!(
                bytes.ends_with(chunk.as_bytes()),
                "raw backend bytes must be forwarded"
            );
            assert_eq!(
                crate::policies::LoadProvider::sample(manager.as_ref(), &target),
                crate::policies::LoadSample {
                    active_prefill_tokens: 71,
                    active_requests: 2
                },
                "non-output chunk must retain prefill: {chunk:?}",
            );
            assert_eq!(worker.load(), 2);
        }
        tx.send(Ok(bytes::Bytes::from_static(b"\"}]}\r\n\r\n")))
            .unwrap();
        body.next().await.unwrap().unwrap();
        assert_eq!(
            crate::policies::LoadProvider::sample(manager.as_ref(), &target),
            crate::policies::LoadSample {
                active_prefill_tokens: 7,
                active_requests: 2
            },
        );
        for chunk in [
            "data: {\"choices\":[{\"text\":\"again\"}]}\n\n",
            ": keepalive containing data: [DONE]\n\n",
            "data: [DONE]\n\n",
        ] {
            tx.send(Ok(bytes::Bytes::from_static(chunk.as_bytes())))
                .unwrap();
            body.next().await.unwrap().unwrap();
            assert_eq!(worker.load(), 2, "only EOF/drop releases the request");
            assert_eq!(
                crate::policies::LoadProvider::sample(manager.as_ref(), &target),
                crate::policies::LoadSample {
                    active_prefill_tokens: 7,
                    active_requests: 2
                },
            );
        }
        drop(tx);
        assert!(tokio::time::timeout(Duration::from_secs(5), body.next())
            .await
            .unwrap()
            .is_none());
        assert_eq!(worker.load(), 1);
        assert_eq!(
            crate::policies::LoadProvider::sample(manager.as_ref(), &target),
            crate::policies::LoadSample {
                active_prefill_tokens: 7,
                active_requests: 1
            },
        );
        drop(unrelated);
        assert_eq!(worker.load(), 0);
        assert_eq!(
            crate::policies::LoadProvider::sample(manager.as_ref(), &target),
            crate::policies::LoadSample::default()
        );
        server.abort();
    }

    #[tokio::test]
    async fn http_completion_output_frame_marker_prevents_first_byte_decode() {
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "http://completion-marker".into(),
            WorkerType::Regular,
        ));
        let manager = Arc::new(crate::policies::InMemorySlotManager::new());
        let target = crate::kv_index::RoutableTarget {
            instance_id: Arc::from(worker.url()),
            dp_rank: 0,
        };
        let load = admitted_kv_load(worker.clone(), manager.clone(), target.clone());
        let (mut producer, consumer) = crate::backend::stream_output::output_frame_tracking();
        let (tx, rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();
        assert!(emit_http_first_sse(
            &tx,
            &serde_json::json!({}),
            Some(&mut producer)
        ));
        assert!(producer.send(
            &tx,
            bytes::Bytes::from_static(b"data: {\"choices\":[{\"text\":\"output\"}]}\n\n"),
            true
        ));
        let response = hold_request_load_with_output_frame(
            Response::new(Body::from_stream(UnboundedReceiverStream::new(rx))),
            load,
            None,
            Some(consumer),
        );
        let mut body = response.into_body().into_data_stream();
        body.next().await.unwrap().unwrap();
        assert_eq!(
            crate::policies::LoadProvider::sample(manager.as_ref(), &target),
            crate::policies::LoadSample {
                active_prefill_tokens: 64,
                active_requests: 1
            }
        );
        body.next().await.unwrap().unwrap();
        assert_eq!(
            crate::policies::LoadProvider::sample(manager.as_ref(), &target),
            crate::policies::LoadSample {
                active_prefill_tokens: 0,
                active_requests: 1
            }
        );
        assert_eq!(worker.load(), 1);
        drop(body);
        assert_eq!(worker.load(), 0);
    }

    #[tokio::test]
    async fn http_completion_nonoutput_cancel_and_error_isolate_admitted_load() {
        for (echo, fail_body, status) in [
            (false, false, StatusCode::OK),
            (false, true, StatusCode::OK),
            (true, false, StatusCode::OK),
            (false, false, StatusCode::BAD_REQUEST),
        ] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
                format!("http://{}", listener.local_addr().unwrap()),
                WorkerType::Regular,
            ));
            let (tx, rx) =
                tokio::sync::mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();
            let receiver = Arc::new(Mutex::new(Some(rx)));
            let app = axum::Router::new().route(
                "/v1/completions",
                axum::routing::post(move || {
                    let rx = receiver.lock().take().unwrap();
                    async move {
                        let mut response =
                            Response::new(Body::from_stream(UnboundedReceiverStream::new(rx)));
                        *response.status_mut() = status;
                        response
                    }
                }),
            );
            let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let manager = Arc::new(crate::policies::InMemorySlotManager::new());
            let target = crate::kv_index::RoutableTarget {
                instance_id: Arc::from(worker.url()),
                dp_rank: 0,
            };
            crate::policies::SlotManager::begin_prefill(manager.as_ref(), &target, 7);
            worker.increment_load();
            let unrelated = RequestLoadGuard::new(
                Some(worker.clone()),
                Some(KvRequestSlot::new(manager.clone(), target.clone(), 7, None)),
            );
            let load = admitted_kv_load(worker.clone(), manager.clone(), target.clone());
            let request: CompletionRequest = serde_json::from_value(
                serde_json::json!({ "prompt": [1], "stream": true, "echo": echo }),
            )
            .unwrap();
            let completion = CompletionRoute {
                tokens: None,
                input_active: false,
                raw_body: None,
                echo,
            };
            let response = create_test_regular_router()
                .send_typed_request(
                    &request,
                    TypedDispatch {
                        headers: None,
                        route: "/v1/completions",
                        worker_url: worker.url(),
                        is_stream: true,
                        completion: Some(&completion),
                        load,
                        prepared: None,
                    },
                    None,
                )
                .await;
            let mut body = response.into_body().into_data_stream();
            let chunk = if echo || !status.is_success() {
                b"data: {\"choices\":[{\"text\":\"not proven generated output\"}]}\n\n".as_slice()
            } else {
                b": comment containing data: [DONE]\n\n".as_slice()
            };
            tx.send(Ok(bytes::Bytes::from_static(chunk))).unwrap();
            tokio::time::timeout(Duration::from_secs(5), body.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            assert_eq!(worker.load(), 2);
            assert_eq!(
                crate::policies::LoadProvider::sample(manager.as_ref(), &target),
                crate::policies::LoadSample {
                    active_prefill_tokens: 71,
                    active_requests: 2
                }
            );
            if fail_body {
                tx.send(Err(std::io::Error::other("broken backend body")))
                    .unwrap();
                assert!(tokio::time::timeout(Duration::from_secs(5), body.next())
                    .await
                    .unwrap()
                    .unwrap()
                    .is_err());
            }
            drop(body);
            assert_eq!(worker.load(), 1);
            assert_eq!(
                crate::policies::LoadProvider::sample(manager.as_ref(), &target),
                crate::policies::LoadSample {
                    active_prefill_tokens: 7,
                    active_requests: 1
                }
            );
            drop(unrelated);
            assert_eq!(worker.load(), 0);
            server.abort();
        }
    }

    async fn check_health_preserves_guarded_load(registry_checker: bool) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            format!("http://{}", listener.local_addr().unwrap()),
            WorkerType::Regular,
        ));
        let checks = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let checked = Arc::new(tokio::sync::Notify::new());
        let handler_checks = checks.clone();
        let handler_checked = checked.clone();
        let app = axum::Router::new().route(
            "/health",
            axum::routing::get(move || {
                let checks = handler_checks.clone();
                let checked = handler_checked.clone();
                async move {
                    checks.fetch_add(1, Ordering::Release);
                    checked.notify_one();
                    StatusCode::OK
                }
            }),
        );
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let manager = Arc::new(crate::policies::InMemorySlotManager::new());
        let target = crate::kv_index::RoutableTarget {
            instance_id: Arc::from(worker.url()),
            dp_rank: 0,
        };
        let old_request = admitted_kv_load(worker.clone(), manager.clone(), target.clone());
        let health_checker = if registry_checker {
            let registry = WorkerRegistry::new();
            registry.register(worker.clone());
            registry.start_health_checker(1)
        } else {
            crate::core::start_health_checker(
                Arc::new(std::sync::RwLock::new(vec![worker.clone()])),
                1,
            )
        };

        // The next completed health check is after the old tenth-cycle reset,
        // without invoking reset_load directly or relying on a guessed sleep.
        let completed = tokio::time::timeout(Duration::from_secs(14), async {
            while checks.load(Ordering::Acquire) < 11 {
                checked.notified().await;
            }
        })
        .await;
        let load_after_checks = worker.load();
        let new_request = admitted_kv_load(worker.clone(), manager.clone(), target.clone());
        drop(old_request);
        let load_after_old_release = worker.load();
        let remaining = crate::policies::LoadProvider::sample(manager.as_ref(), &target);
        drop(new_request);
        health_checker.shutdown().await;
        server.abort();

        completed.expect("health checker must execute more than ten cycles");
        assert_eq!(
            load_after_checks, 1,
            "health checks cannot erase active load"
        );
        assert_eq!(
            load_after_old_release, 1,
            "an older request cannot release a newer request's load"
        );
        assert_eq!(remaining.active_requests, 1);
        assert_eq!(remaining.active_prefill_tokens, 64);
        assert_eq!(worker.load(), 0);
        assert_eq!(
            crate::policies::LoadProvider::sample(manager.as_ref(), &target).active_requests,
            0
        );
    }

    #[tokio::test]
    async fn registry_health_checker_preserves_guarded_load() {
        check_health_preserves_guarded_load(true).await;
    }

    #[tokio::test]
    async fn collection_health_checker_preserves_guarded_load() {
        check_health_preserves_guarded_load(false).await;
    }

    #[tokio::test]
    async fn cancelled_dispatch_releases_only_its_prefill_and_worker_load() {
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "grpc://cancel:50051".into(),
            WorkerType::Regular,
        ));
        let manager = Arc::new(crate::policies::InMemorySlotManager::new());
        let target = crate::kv_index::RoutableTarget {
            instance_id: Arc::from("cancel"),
            dp_rank: 0,
        };
        // An unrelated in-flight request makes duplicate decrement visible.
        crate::policies::SlotManager::begin_prefill(manager.as_ref(), &target, 7);
        worker.increment_load();
        let load = admitted_kv_load(worker.clone(), manager.clone(), target.clone());
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let dispatch = tokio::spawn(async move {
            let _load = load;
            started_tx.send(()).unwrap();
            std::future::pending::<Response>().await
        });
        started_rx.await.unwrap();
        assert_eq!(worker.load(), 2);
        dispatch.abort();
        assert!(dispatch.await.unwrap_err().is_cancelled());
        assert_eq!(worker.load(), 1);
        assert_eq!(
            crate::policies::LoadProvider::sample(manager.as_ref(), &target),
            crate::policies::LoadSample {
                active_prefill_tokens: 7,
                active_requests: 1,
            }
        );
    }

    #[tokio::test]
    async fn cancelled_nonstream_body_read_releases_prefill_and_worker_load() {
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "grpc://body:50051".into(),
            WorkerType::Regular,
        ));
        let manager = Arc::new(crate::policies::InMemorySlotManager::new());
        let target = crate::kv_index::RoutableTarget {
            instance_id: Arc::from("body"),
            dp_rank: 0,
        };
        let load = admitted_kv_load(worker.clone(), manager.clone(), target.clone());
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let dispatch = tokio::spawn(async move {
            let _load = load;
            let body = Body::from_stream(futures_util::stream::once(async move {
                started_tx.send(()).unwrap();
                std::future::pending::<Result<bytes::Bytes, std::io::Error>>().await
            }));
            to_bytes(body, usize::MAX).await
        });
        started_rx.await.unwrap();
        dispatch.abort();
        assert!(dispatch.await.unwrap_err().is_cancelled());
        assert_eq!(worker.load(), 0);
        assert_eq!(
            crate::policies::LoadProvider::sample(manager.as_ref(), &target),
            crate::policies::LoadSample::default()
        );
    }

    #[tokio::test]
    async fn response_transfer_keeps_load_through_decode_and_releases_on_drop() {
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "grpc://transfer:50051".into(),
            WorkerType::Regular,
        ));
        let manager = Arc::new(crate::policies::InMemorySlotManager::new());
        let target = crate::kv_index::RoutableTarget {
            instance_id: Arc::from("transfer"),
            dp_rank: 0,
        };
        let load = admitted_kv_load(worker.clone(), manager.clone(), target.clone());
        let stream = futures_util::stream::once(async {
            Ok::<_, std::io::Error>(bytes::Bytes::from_static(b"first token"))
        })
        .chain(futures_util::stream::pending());
        let response =
            hold_request_load_until_body_done(Response::new(Body::from_stream(stream)), load);
        assert_eq!(worker.load(), 1);
        let mut body = response.into_body().into_data_stream();
        body.next().await.unwrap().unwrap();
        assert_eq!(
            crate::policies::LoadProvider::sample(manager.as_ref(), &target),
            crate::policies::LoadSample {
                active_prefill_tokens: 0,
                active_requests: 1,
            }
        );
        assert_eq!(worker.load(), 1);
        drop(body);
        assert_eq!(worker.load(), 0);
        assert_eq!(
            crate::policies::LoadProvider::sample(manager.as_ref(), &target),
            crate::policies::LoadSample::default()
        );
    }

    #[tokio::test]
    async fn grpc_prompt_info_chunk_keeps_kv_load_in_prefill_until_model_output() {
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "grpc://prompt-info:50051".into(),
            WorkerType::Regular,
        ));
        let manager = Arc::new(crate::policies::InMemorySlotManager::new());
        let target = crate::kv_index::RoutableTarget {
            instance_id: Arc::from("prompt-info"),
            dp_rank: 0,
        };
        let load = admitted_kv_load(worker.clone(), manager.clone(), target.clone());
        let (mut producer_output_frame, output_frame) =
            crate::backend::stream_output::output_frame_tracking();
        let (body_tx, body_rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();
        let (queued_tx, queued_rx) = tokio::sync::oneshot::channel();
        let (finish_tx, finish_rx) = tokio::sync::oneshot::channel();
        let producer = tokio::spawn(async move {
            assert!(producer_output_frame.send(
                &body_tx,
                bytes::Bytes::from_static(b"data: prompt_info_only\n\n"),
                false,
            ));
            // Model the gRPC producer getting ahead of the client: by the
            // time the first body frame is polled, it has already queued the
            // generated-output frame too.
            assert!(producer_output_frame.send(
                &body_tx,
                bytes::Bytes::from_static(b"data: generated_token\n\n"),
                true,
            ));
            queued_tx.send(()).unwrap();
            let _ = finish_rx.await;
        });
        queued_rx.await.unwrap();
        let mut response = Response::new(Body::from_stream(UnboundedReceiverStream::new(body_rx)));
        response
            .extensions_mut()
            .insert(crate::backend::grpc::GrpcStreamTask::with_output_frame(
                producer,
                output_frame,
            ));
        let response = hold_request_load_until_body_done(response, load);
        let mut body = response.into_body().into_data_stream();

        assert_eq!(
            body.next().await.unwrap().unwrap(),
            bytes::Bytes::from_static(b"data: prompt_info_only\n\n")
        );
        assert_eq!(
            crate::policies::LoadProvider::sample(manager.as_ref(), &target),
            crate::policies::LoadSample {
                active_prefill_tokens: 64,
                active_requests: 1,
            }
        );

        assert_eq!(
            body.next().await.unwrap().unwrap(),
            bytes::Bytes::from_static(b"data: generated_token\n\n")
        );
        assert_eq!(
            crate::policies::LoadProvider::sample(manager.as_ref(), &target),
            crate::policies::LoadSample {
                active_prefill_tokens: 0,
                active_requests: 1,
            }
        );

        finish_tx.send(()).unwrap();
        assert!(body.next().await.is_none());
        assert_eq!(worker.load(), 0);
        assert_eq!(
            crate::policies::LoadProvider::sample(manager.as_ref(), &target),
            crate::policies::LoadSample::default()
        );
    }

    #[tokio::test]
    async fn producer_completion_and_body_drop_release_shared_load_once() {
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "grpc://producer:50051".into(),
            WorkerType::Regular,
        ));
        let manager = Arc::new(crate::policies::InMemorySlotManager::new());
        let target = crate::kv_index::RoutableTarget {
            instance_id: Arc::from("producer"),
            dp_rank: 0,
        };
        crate::policies::SlotManager::begin_prefill(manager.as_ref(), &target, 7);
        worker.increment_load();
        for cancel in [false, true] {
            let load = admitted_kv_load(worker.clone(), manager.clone(), target.clone());
            let (finish_tx, finish_rx) = tokio::sync::oneshot::channel();
            let producer = tokio::spawn(async move {
                let _ = finish_rx.await;
            });
            let response = hold_request_load_with_output_frame(
                Response::new(Body::from("buffered")),
                load,
                Some(producer),
                None,
            );
            if cancel {
                drop(response);
                assert_eq!(worker.load(), 1, "body cancellation releases synchronously");
                drop(finish_tx);
            } else {
                finish_tx.send(()).unwrap();
                tokio::time::timeout(Duration::from_secs(1), async {
                    while worker.load() != 1 {
                        tokio::task::yield_now().await;
                    }
                })
                .await
                .unwrap();
                drop(response);
            }
            assert_eq!(worker.load(), 1);
            assert_eq!(
                crate::policies::LoadProvider::sample(manager.as_ref(), &target),
                crate::policies::LoadSample {
                    active_prefill_tokens: 7,
                    active_requests: 1,
                }
            );
        }
    }

    #[tokio::test]
    async fn typed_request_cancellation_before_headers_or_during_body_releases_load() {
        use axum::routing::post;
        for (headers_ready, streaming) in [(false, false), (false, true), (true, false)] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
                format!("http://{}", listener.local_addr().unwrap()),
                WorkerType::Regular,
            ));
            let started = Arc::new(tokio::sync::Notify::new());
            let handler_started = started.clone();
            let app = axum::Router::new().route(
                "/v1/chat/completions",
                post(move || {
                    let started = handler_started.clone();
                    async move {
                        if headers_ready {
                            let stream = futures_util::stream::once(async move {
                                started.notify_one();
                                std::future::pending::<Result<bytes::Bytes, std::io::Error>>().await
                            });
                            Response::new(Body::from_stream(stream))
                        } else {
                            started.notify_one();
                            std::future::pending::<Response>().await
                        }
                    }
                }),
            );
            let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let mut router = create_test_regular_router();
            router.worker_registry = Arc::new(WorkerRegistry::new());
            router.worker_registry.register(worker.clone());
            router.policy_registry =
                Arc::new(PolicyRegistry::new(crate::config::PolicyConfig::KvAware {
                    router_temperature: 0.0,
                    kv_penalty_host: 0.8,
                    kv_penalty_external: 0.6,
                    kv_penalty_disk: 0.3,
                    static_kv_event_sources: Default::default(),
                }));
            let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
                "messages": [{"role": "user", "content": "hello"}], "stream": streaming,
            }))
            .unwrap();
            let mut future =
                Box::pin(router.route_typed_request(None, &request, "/v1/chat/completions", None));
            tokio::time::timeout(Duration::from_secs(5), async {
                tokio::select! {
                    _ = started.notified() => {},
                    _ = &mut future => panic!("request completed before cancellation"),
                }
            })
            .await
            .unwrap();
            assert_eq!(worker.load(), 1);
            drop(future);
            assert_eq!(
                worker.load(),
                0,
                "headers_ready={headers_ready}, streaming={streaming}"
            );
            server.abort();
        }
    }

    #[tokio::test]
    async fn typed_request_success_and_errors_release_only_the_current_attempt() {
        use axum::routing::post;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            format!("http://{}", listener.local_addr().unwrap()),
            WorkerType::Regular,
        ));
        let mode = Arc::new(Mutex::new((StatusCode::OK, false)));
        let handler_mode = mode.clone();
        let app = axum::Router::new().route(
            "/v1/chat/completions",
            post(move || {
                let (status, fail_body) = *handler_mode.lock();
                async move {
                    let body = if fail_body {
                        Body::from_stream(futures_util::stream::once(async {
                            Err::<bytes::Bytes, _>(std::io::Error::other("broken backend body"))
                        }))
                    } else {
                        Body::from("{}")
                    };
                    let mut response = Response::new(body);
                    *response.status_mut() = status;
                    response
                }
            }),
        );
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let mut router = create_test_regular_router();
        router.worker_registry = Arc::new(WorkerRegistry::new());
        router.worker_registry.register(worker.clone());
        router.policy_registry =
            Arc::new(PolicyRegistry::new(crate::config::PolicyConfig::KvAware {
                router_temperature: 0.0,
                kv_penalty_host: 0.8,
                kv_penalty_external: 0.6,
                kv_penalty_disk: 0.3,
                static_kv_event_sources: Default::default(),
            }));
        router.retry_config.max_retries = 0;
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "hello"}],
        }))
        .unwrap();
        worker.increment_load(); // An unrelated request must remain counted.
        for (status, fail_body) in [
            (StatusCode::OK, false),
            (StatusCode::BAD_REQUEST, false),
            (StatusCode::INTERNAL_SERVER_ERROR, false),
            (StatusCode::OK, true),
        ] {
            *mode.lock() = (status, fail_body);
            let response = router
                .route_typed_request(None, &request, "/v1/chat/completions", None)
                .await;
            let expected_status = if fail_body {
                StatusCode::INTERNAL_SERVER_ERROR
            } else {
                status
            };
            assert_eq!(response.status(), expected_status);
            assert_eq!(worker.load(), 1, "status={status}, fail_body={fail_body}");
            to_bytes(response.into_body(), usize::MAX).await.unwrap();
            assert_eq!(worker.load(), 1);
        }
        worker.decrement_load();
        server.abort();
    }

    #[tokio::test]
    async fn streaming_body_releases_kv_slot_lifecycle() {
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "grpc://worker:50051".to_string(),
            WorkerType::Regular,
        ));
        let manager = Arc::new(crate::policies::InMemorySlotManager::new());
        let target = crate::kv_index::RoutableTarget {
            instance_id: Arc::from("worker"),
            dp_rank: 0,
        };
        crate::policies::SlotManager::begin_prefill(manager.as_ref(), &target, 64);
        worker.increment_load();
        let response = hold_load_until_body_done(
            Response::new(Body::from("first token")),
            worker.clone(),
            Some(KvRequestSlot::new(
                manager.clone(),
                target.clone(),
                64,
                None,
            )),
        );
        assert_eq!(
            crate::policies::LoadProvider::sample(manager.as_ref(), &target),
            crate::policies::LoadSample {
                active_prefill_tokens: 64,
                active_requests: 1,
            }
        );
        let _ = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(
            crate::policies::LoadProvider::sample(manager.as_ref(), &target),
            crate::policies::LoadSample::default()
        );
        assert_eq!(worker.load(), 0);
    }

    #[tokio::test]
    async fn grpc_stream_load_follows_producer_and_drop_cancels_it() {
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "grpc://worker:50051".to_string(),
            WorkerType::Regular,
        ));

        worker.increment_load();
        let (finish_tx, finish_rx) = tokio::sync::oneshot::channel();
        let producer = tokio::spawn(async move {
            let _ = finish_rx.await;
        });
        let mut response = Response::new(Body::from("buffered"));
        response
            .extensions_mut()
            .insert(crate::backend::grpc::GrpcStreamTask::new(producer));
        let response = hold_load_until_body_done(response, worker.clone(), None);
        finish_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while worker.load() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        // The body is still buffered, but backend generation has finished.
        assert_eq!(worker.load(), 0);
        drop(response);

        worker.increment_load();
        let producer = tokio::spawn(std::future::pending::<()>());
        let mut response = Response::new(Body::from("buffered"));
        response
            .extensions_mut()
            .insert(crate::backend::grpc::GrpcStreamTask::new(producer));
        let response = hold_load_until_body_done(response, worker.clone(), None);
        drop(response);
        tokio::time::timeout(Duration::from_secs(1), async {
            while worker.load() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_wait_for_healthy_workers_empty_list() {
        // Empty list will return error immediately
        let result = Router::wait_for_healthy_workers(&[], 1, 1).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no workers provided"));
    }

    #[tokio::test]
    async fn test_wait_for_healthy_workers_invalid_urls() {
        // This test will timeout quickly since the URLs are invalid
        let result =
            Router::wait_for_healthy_workers(&["http://nonexistent:8080".to_string()], 1, 1).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Timeout"));
    }

    // =============================
    // Tests for transparent proxy header/availability fixes
    // =============================

    /// Create a test router with ConsistentHash policy instead of RoundRobin
    fn create_test_consistent_hash_router() -> Router {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(
            crate::config::types::PolicyConfig::ConsistentHash { virtual_nodes: 100 },
        ));

        let worker1 = BasicWorker::new("http://worker1:8080".to_string(), WorkerType::Regular);
        let worker2 = BasicWorker::new("http://worker2:8080".to_string(), WorkerType::Regular);
        let worker3 = BasicWorker::new("http://worker3:8080".to_string(), WorkerType::Regular);
        worker_registry.register(Arc::new(worker1));
        worker_registry.register(Arc::new(worker2));
        worker_registry.register(Arc::new(worker3));

        let (_, rx) = tokio::sync::watch::channel(HashMap::new());
        Router {
            worker_registry,
            policy_registry,
            epd: None,
            worker_startup_timeout_secs: 5,
            worker_startup_check_interval_secs: 1,
            intra_node_data_parallel_size: 1,
            api_key: None,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            health_config: HealthConfig::default(),
            frontend: crate::backend::EngineFrontend::new(),
            completion_input: None,
            completion_preparation: crate::backend::completion_input::CompletionPreparation::new(
                64,
                Duration::from_secs(30),
                Duration::from_secs(5),
            ),
            _worker_loads: Arc::new(rx),
            _load_monitor_handle: None,
            kv_index_supervisor: None,
            _kv_signal_handle: None,
            kv_aware_session_affinity_ttl_secs: None,
            program_scheduler: None,
            _program_observation_handle: None,
            program_targets_cache: Mutex::new(HashMap::new()),
            program_token_estimator: Arc::new(MomentumTokenEstimator::default()),
        }
    }

    #[test]
    fn test_headers_to_request_headers_basic() {
        // Test that headers_to_request_headers correctly converts HeaderMap to HashMap
        let mut header_map = HeaderMap::new();
        header_map.insert("x-session-id", HeaderValue::from_static("session-123"));
        header_map.insert("content-type", HeaderValue::from_static("application/json"));
        header_map.insert("X-Custom-Header", HeaderValue::from_static("custom-value"));

        let result = Router::headers_to_request_headers(Some(&header_map));
        assert!(result.is_some());
        let headers = result.unwrap();

        // All keys should be lowercased
        assert_eq!(headers.get("x-session-id").unwrap(), "session-123");
        assert_eq!(headers.get("content-type").unwrap(), "application/json");
        assert_eq!(headers.get("x-custom-header").unwrap(), "custom-value");
    }

    #[test]
    fn test_headers_to_request_headers_none() {
        // Test that None headers produce None output
        let result = Router::headers_to_request_headers(None);
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn program_target_unavailable_has_distinct_retryable_response() {
        let response = Router::program_target_unavailable_response("/v1/chat/completions");
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(body, "Program-bound worker is unavailable");
    }

    #[test]
    fn test_headers_to_request_headers_empty() {
        // Test that empty HeaderMap produces empty HashMap
        let header_map = HeaderMap::new();
        let result = Router::headers_to_request_headers(Some(&header_map));
        assert!(result.is_some());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_select_worker_for_model_with_consistent_hash_uses_headers() {
        // Verify that select_worker_for_model passes headers through to the policy,
        // producing consistent routing for the same session ID
        let router = create_test_consistent_hash_router();

        let mut header_map = HeaderMap::new();
        header_map.insert("x-session-id", HeaderValue::from_static("sticky-session-1"));

        // Make multiple selections with the same headers - should all pick the same worker
        let mut selected_urls: Vec<String> = Vec::new();
        for _ in 0..10 {
            let worker = router
                .select_worker_for_model(
                    None,
                    &router.available_workers(None),
                    Some(r#"{"prompt": "test"}"#),
                    Some(&header_map),
                    KvRequestTokens::Shared(None),
                    None,
                    None,
                    None,
                    None,
                    &[],
                )
                .expect("Should select a worker");
            selected_urls.push(worker.worker.url().to_string());
        }

        // All selections should go to the same worker (sticky routing)
        let first = &selected_urls[0];
        for (i, url) in selected_urls.iter().enumerate() {
            assert_eq!(
                url, first,
                "Request {} routed to {}, expected {} (session stickiness broken)",
                i, url, first
            );
        }
    }

    #[test]
    fn test_select_worker_for_model_filters_unavailable_workers() {
        // Verify that select_worker_for_model skips unhealthy workers
        let router = create_test_consistent_hash_router();

        // Mark worker1 and worker2 as unhealthy, leaving only worker3
        let all_workers = router.worker_registry.get_all();
        for w in &all_workers {
            if w.url() == "http://worker1:8080" || w.url() == "http://worker2:8080" {
                w.set_healthy(false);
            }
        }

        let worker = router
            .select_worker_for_model(
                None,
                &router.available_workers(None),
                Some(r#"{"prompt": "test"}"#),
                None,
                KvRequestTokens::Shared(None),
                None,
                None,
                None,
                None,
                &[],
            )
            .expect("Should select the remaining healthy worker");

        assert_eq!(
            worker.worker.url(),
            "http://worker3:8080",
            "Should only select the healthy worker"
        );
    }

    #[test]
    fn test_select_worker_for_model_returns_none_when_all_unavailable() {
        // Verify that when all workers are unhealthy, None is returned
        let router = create_test_consistent_hash_router();

        // Mark all workers as unhealthy
        let all_workers = router.worker_registry.get_all();
        for w in &all_workers {
            w.set_healthy(false);
        }

        let result = router.select_worker_for_model(
            None,
            &router.available_workers(None),
            Some(r#"{"prompt": "test"}"#),
            None,
            KvRequestTokens::Shared(None),
            None,
            None,
            None,
            None,
            &[],
        );
        assert!(
            result.is_none(),
            "Should return None when all workers are unavailable"
        );
    }

    #[test]
    fn test_retry_exclusion_does_not_change_other_policies() {
        let router = create_test_consistent_hash_router();
        let select = |excluded: &[String]| {
            router
                .select_worker_for_model(
                    None,
                    &router.available_workers(None),
                    Some(r#"{"prompt": "test"}"#),
                    None,
                    KvRequestTokens::Shared(None),
                    None,
                    None,
                    None,
                    None,
                    excluded,
                )
                .expect("Should select a worker")
                .worker
                .url()
                .to_string()
        };
        let first = select(&[]);
        assert_eq!(select(std::slice::from_ref(&first)), first);
    }

    #[test]
    fn test_consistent_hash_different_sessions_can_route_differently() {
        // Verify that different session IDs can route to different workers
        let router = create_test_consistent_hash_router();

        let mut worker_urls_seen = std::collections::HashSet::new();
        for i in 0..50 {
            let mut header_map = HeaderMap::new();
            let session_id = format!("session-{}", i);
            header_map.insert("x-session-id", HeaderValue::from_str(&session_id).unwrap());

            if let Some(worker) = router.select_worker_for_model(
                None,
                &router.available_workers(None),
                Some(r#"{"prompt": "test"}"#),
                Some(&header_map),
                KvRequestTokens::Shared(None),
                None,
                None,
                None,
                None,
                &[],
            ) {
                worker_urls_seen.insert(worker.worker.url().to_string());
            }
        }

        // With 50 different sessions and 3 workers, we should see at least 2 workers used
        assert!(
            worker_urls_seen.len() >= 2,
            "Expected distribution across workers, only used: {:?}",
            worker_urls_seen
        );
    }

    #[test]
    fn test_inline_header_conversion_matches_headers_to_request_headers() {
        // Verify that the inline header conversion pattern used in vllm_pd_router
        // produces the same result as Router::headers_to_request_headers.
        let mut header_map = HeaderMap::new();
        header_map.insert("X-Session-Id", HeaderValue::from_static("session-abc"));
        header_map.insert("Content-Type", HeaderValue::from_static("application/json"));
        header_map.insert("x-user-id", HeaderValue::from_static("user-42"));

        // Method 1: Router::headers_to_request_headers (used in router.rs)
        let method1 = Router::headers_to_request_headers(Some(&header_map)).unwrap();

        // Method 2: Inline conversion (used in vllm_pd_router.rs)
        let method2: HashMap<String, String> = header_map
            .iter()
            .filter_map(|(name, value)| {
                value
                    .to_str()
                    .ok()
                    .map(|v| (name.as_str().to_lowercase(), v.to_string()))
            })
            .collect();

        assert_eq!(
            method1, method2,
            "Both header conversion methods should produce identical results"
        );
    }

    /// Helper: start a minimal mock server that responds 200 on /health.
    async fn start_healthy_mock_server() -> (String, tokio::task::JoinHandle<()>) {
        use axum::{routing::get, Router as AxumRouter};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = AxumRouter::new().route("/health", get(|| async { "ok" }));
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        (format!("http://{}", addr), handle)
    }

    #[tokio::test]
    async fn test_wait_for_healthy_workers_all_healthy() {
        let (url, _handle) = start_healthy_mock_server().await;
        let result = Router::wait_for_healthy_workers(&[url], 5, 1).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_wait_for_healthy_workers_partial_health() {
        // One healthy server + one unreachable URL.
        // The new behaviour succeeds when at least one host is healthy.
        let (healthy_url, _handle) = start_healthy_mock_server().await;
        let unreachable_url = "http://127.0.0.1:1".to_string(); // port 1 is unreachable

        let result = Router::wait_for_healthy_workers(&[healthy_url, unreachable_url], 5, 1).await;
        assert!(result.is_ok());
    }

    /// Helper: start a mock server that returns 503 on /health for a given
    /// duration, then switches to 200. Simulates a worker with a slow startup.
    async fn start_delayed_healthy_mock_server(
        delay: std::time::Duration,
    ) -> (String, tokio::task::JoinHandle<()>) {
        use axum::{extract::State, http::StatusCode, routing::get, Router as AxumRouter};
        use std::sync::Arc;
        use tokio::net::TcpListener;

        // Start the delay on the first health request, not when the task is
        // spawned. Under a loaded test runner, spawn-to-request scheduling can
        // otherwise exceed the delay and make the first assertion flaky.
        let first_request = Arc::new(std::sync::OnceLock::<std::time::Instant>::new());
        let ready_after = Arc::new(delay);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let app = AxumRouter::new()
            .route(
                "/health",
                get(
                    move |State((first_request, ready_after)): State<(
                        Arc<std::sync::OnceLock<std::time::Instant>>,
                        Arc<std::time::Duration>,
                    )>| async move {
                        let start = first_request.get_or_init(std::time::Instant::now);
                        if start.elapsed() >= *ready_after {
                            StatusCode::OK
                        } else {
                            StatusCode::SERVICE_UNAVAILABLE
                        }
                    },
                ),
            )
            .with_state((first_request, ready_after));

        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        (format!("http://{}", addr), handle)
    }

    #[tokio::test]
    async fn test_wait_for_healthy_workers_dp_aware_dedup() {
        // DP-aware URLs like http://host:port@0, @1, @2 should be deduplicated
        // to a single /health check on http://host:port.
        let (base_url, _handle) = start_healthy_mock_server().await;
        let dp_urls: Vec<String> = (0..4)
            .map(|rank| format!("{}@{}", base_url, rank))
            .collect();

        let result = Router::wait_for_healthy_workers(&dp_urls, 5, 1).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_delayed_worker_becomes_routable_via_health_checker() {
        use crate::core::{BasicWorker, HealthConfig, WorkerRegistry, WorkerType};
        use std::sync::Arc;

        // Two workers: one immediately healthy, one delayed (503 for 2s, then 200).
        let (healthy_url, _h1) = start_healthy_mock_server().await;
        let (delayed_url, _h2) =
            start_delayed_healthy_mock_server(std::time::Duration::from_secs(2)).await;

        // Verify the delayed worker is genuinely unhealthy right now.
        let client = reqwest::Client::new();
        let resp = client
            .get(format!("{}/health", delayed_url))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 503);

        // ── Step 1: wait_for_healthy_workers (mirrors PdRouterBase::new startup) ──
        // This succeeds because the healthy worker responds immediately,
        // even though the delayed worker is still returning 503.
        let result =
            Router::wait_for_healthy_workers(&[healthy_url.clone(), delayed_url.clone()], 10, 1)
                .await;
        assert!(
            result.is_ok(),
            "Startup should succeed with at least one healthy worker"
        );

        // ── Step 2: register workers in the registry (mirrors PdRouterBase::new) ──
        let registry = Arc::new(WorkerRegistry::new());

        let healthy_worker = Arc::new(
            BasicWorker::new(healthy_url, WorkerType::Decode).with_health_config(HealthConfig {
                timeout_secs: 2,
                check_interval_secs: 1,
                endpoint: "/health".to_string(),
                failure_threshold: 3,
                success_threshold: 1,
            }),
        );
        registry.register(healthy_worker);

        let delayed_worker = Arc::new(
            BasicWorker::new(delayed_url, WorkerType::Decode).with_health_config(HealthConfig {
                timeout_secs: 2,
                check_interval_secs: 1,
                endpoint: "/health".to_string(),
                failure_threshold: 3,
                success_threshold: 1,
            }),
        );
        delayed_worker.set_healthy(false); // starts unhealthy
        registry.register(delayed_worker.clone());

        // Only the immediately-healthy worker should be available for routing.
        let healthy = registry.get_workers_filtered(None, None, None, true);
        assert_eq!(
            healthy.len(),
            1,
            "Only 1 worker should be healthy initially, got {}",
            healthy.len()
        );

        // ── Step 3: start background health checker (mirrors PdRouterBase::new) ──
        let revision_before_recovery = registry.revision();
        let health_checker = registry.start_health_checker(1);

        // ── Step 4: wait for delayed worker to recover ──
        // The mock server switches to 200 at t≈2s. With a 1s check interval
        // and success_threshold=1, the worker should be healthy by t≈3-4s.
        tokio::time::sleep(std::time::Duration::from_secs(4)).await;

        // Both workers should now be available for routing.
        let healthy = registry.get_workers_filtered(None, None, None, true);
        assert_eq!(
            healthy.len(),
            2,
            "Both workers should be healthy after recovery, got {}",
            healthy.len()
        );
        assert!(
            delayed_worker.is_healthy(),
            "Delayed worker should have transitioned to healthy via health checker"
        );
        assert!(registry.revision() > revision_before_recovery);

        health_checker.shutdown().await;
    }
}
