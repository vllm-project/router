//! Optional one-token generation checks. Never dispatch through inference routes.

use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use futures_util::{stream, StreamExt};
use metrics::{counter, gauge};
use parking_lot::Mutex;
use serde_json::{json, Value};
use tokio::{
    task::JoinHandle,
    time::{self, Instant},
};
use tokio_util::sync::CancellationToken;

use super::{
    worker::UNLABELED_WORKER_MODEL_ID, ConnectionMode, Worker, WorkerId, WorkerRegistry, WorkerType,
};
use crate::metrics::RouterMetrics;

const CONCURRENCY: usize = 4;
const RESPONSE_LIMIT_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, Default)]
#[doc(hidden)]
pub struct CanaryState {
    inner: Arc<CanaryInner>,
}

#[derive(Debug, Default)]
struct CanaryInner {
    enabled: AtomicBool,
    generation: AtomicU64,
    blocked: AtomicBool,
    traffic: Mutex<Option<Traffic>>,
}

#[derive(Debug)]
struct Traffic {
    users: usize,
    generation: u64,
    cancel: CancellationToken,
    idle_since: Instant,
    last_probe: Option<Instant>,
    revision: u64,
    probing: bool,
    verified: bool,
    unsupported: bool,
    failures: u32,
    successes: u32,
}

impl CanaryState {
    pub(crate) fn healthy(&self) -> bool {
        !self.inner.blocked.load(Ordering::Acquire)
    }

    /// Serialize combined-health metric updates with canary transitions.
    pub(crate) fn update_passive_health(&self, update: impl FnOnce()) {
        if self.inner.enabled.load(Ordering::Acquire) {
            let _state = self.inner.traffic.lock();
            update();
        } else {
            update();
        }
    }

    pub(crate) fn enable(&self) {
        let mut state = self.inner.traffic.lock();
        if state.is_none() {
            *state = Some(Traffic {
                users: 0,
                generation: self.inner.generation.fetch_add(1, Ordering::Relaxed),
                cancel: CancellationToken::new(),
                idle_since: Instant::now(),
                last_probe: None,
                revision: 0,
                probing: false,
                verified: false,
                unsupported: false,
                failures: 0,
                successes: 0,
            });
            self.inner.enabled.store(true, Ordering::Release);
        }
    }

    pub(crate) fn retire(&self) {
        let mut state = self.inner.traffic.lock();
        if let Some(state) = state.take() {
            state.cancel.cancel();
        }
        self.inner.enabled.store(false, Ordering::Release);
        self.inner.blocked.store(false, Ordering::Release);
    }

    fn idle_seconds(&self, now: Instant) -> Option<f64> {
        let state = self.inner.traffic.lock();
        let state = state.as_ref().filter(|s| !s.unsupported)?;
        Some(if state.users == 0 {
            now.saturating_duration_since(state.idle_since)
                .as_secs_f64()
        } else {
            0.
        })
    }

    fn reserve(&self, now: Instant, interval: Duration) -> Option<Probe> {
        let mut state = self.inner.traffic.lock();
        let state = state.as_mut()?;
        if state.unsupported
            || state.probing
            || state.users != 0
            || now.saturating_duration_since(state.idle_since) < interval
            || state
                .last_probe
                .is_some_and(|last| now.saturating_duration_since(last) < interval)
        {
            return None;
        }
        state.probing = true;
        state.last_probe = Some(now);
        Some(Probe {
            state: self.clone(),
            revision: state.revision,
            generation: state.generation,
            cancel: state.cancel.clone(),
        })
    }

    fn enter(&self) -> Option<UserTraffic> {
        if !self.inner.enabled.load(Ordering::Acquire) {
            return None;
        }
        let mut state = self.inner.traffic.lock();
        let state = state.as_mut()?;
        state.users += 1;
        state.revision = state.revision.wrapping_add(1);
        Some(UserTraffic(self.clone(), state.generation))
    }
}

pub(crate) struct UserTraffic(CanaryState, u64);

impl UserTraffic {
    pub(crate) fn begin(worker: &dyn Worker) -> Option<Self> {
        worker.canary_state()?.enter()
    }

    pub(crate) fn track<S: futures_util::Stream>(
        guard: Option<Self>,
        stream: S,
    ) -> impl futures_util::Stream<Item = S::Item> {
        stream::unfold(
            (Box::pin(stream), guard),
            |(mut stream, guard)| async move { stream.next().await.map(|item| (item, (stream, guard))) },
        )
    }
}

impl Drop for UserTraffic {
    fn drop(&mut self) {
        let mut state = self.0.inner.traffic.lock();
        if let Some(state) = state.as_mut().filter(|state| state.generation == self.1) {
            state.users -= 1;
            if state.users == 0 {
                state.idle_since = Instant::now();
            }
        }
    }
}

struct Probe {
    state: CanaryState,
    revision: u64,
    generation: u64,
    cancel: CancellationToken,
}

impl Probe {
    fn apply(&self, worker: &dyn Worker, result: ResultKind) -> ResultKind {
        let mut state = self.state.inner.traffic.lock();
        let Some(state) = state.as_mut() else {
            return ResultKind::Superseded;
        };
        if state.generation != self.generation
            || state.revision != self.revision
            || state.users != 0
        {
            return ResultKind::Superseded;
        }
        let health = &worker.metadata().health_config;
        let was_healthy = self.state.healthy();
        let mut healthy = was_healthy;
        match result {
            ResultKind::Ok => {
                state.verified = true;
                state.failures = 0;
                state.successes = state.successes.saturating_add(1);
                if state.successes >= health.success_threshold.max(1) {
                    healthy = true;
                }
            }
            ResultKind::Unsupported if !state.verified => {
                // No successful standalone generation has ever been seen.
                state.unsupported = true;
            }
            ResultKind::Superseded => {}
            _ if state.verified => {
                state.successes = 0;
                state.failures = state.failures.saturating_add(1);
                if state.failures >= health.failure_threshold.max(1) {
                    healthy = false;
                }
            }
            _ => {}
        }
        if healthy != was_healthy {
            self.state.inner.blocked.store(!healthy, Ordering::Release);
            RouterMetrics::set_worker_health(worker.url(), worker.is_healthy());
            tracing::info!(
                worker = worker.url(),
                healthy,
                "Worker generation health changed"
            );
        }
        result
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        if let Some(state) = self
            .state
            .inner
            .traffic
            .lock()
            .as_mut()
            .filter(|state| state.generation == self.generation)
        {
            state.probing = false;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResultKind {
    Ok,
    Failed,
    Timeout,
    Unsupported,
    Superseded,
}

impl ResultKind {
    fn record(self) {
        let result = match self {
            Self::Ok => "ok",
            Self::Failed => "failed",
            Self::Timeout => "timeout",
            Self::Unsupported => "unsupported",
            Self::Superseded => "superseded",
        };
        counter!("vllm_router_worker_canary_total", "result" => result).increment(1);
    }

    fn transport(error: &reqwest::Error) -> Self {
        if error.is_timeout() {
            Self::Timeout
        } else {
            Self::Failed
        }
    }
}

pub(crate) fn enable(worker: &dyn Worker) {
    // Dedicated P/D roles are not evidence of independent generation support.
    if worker.worker_type() == WorkerType::Regular
        && worker.connection_mode() == ConnectionMode::Http
        && (worker.dp_rank().is_some() || worker.dp_size().unwrap_or(1) <= 1)
    {
        if let Some(state) = worker.canary_state() {
            state.enable();
        }
    }
}

async fn generate(client: &reqwest::Client, worker: &dyn Worker, timeout: Duration) -> ResultKind {
    let mut payload = json!({
        "prompt": "hi",
        "max_tokens": 1,
        "temperature": 0,
        "stream": false,
        "ignore_eos": true,
        "return_token_ids": true
    });
    if worker.model_id() != UNLABELED_WORKER_MODEL_ID {
        payload["model"] = json!(worker.model_id());
    }
    let mut request = client
        .post(worker.endpoint_url("/v1/completions"))
        .timeout(timeout)
        .json(&payload);
    if let Some(rank) = worker.dp_rank() {
        request = request.header("X-data-parallel-rank", rank.to_string());
    }
    let response = match worker.authorize_http_probe(request).send().await {
        Ok(response) => response,
        Err(error) => return ResultKind::transport(&error),
    };
    match response.status().as_u16() {
        404 | 405 | 501 => return ResultKind::Unsupported,
        408 | 504 => return ResultKind::Timeout,
        200..=299 => {}
        _ => return ResultKind::Failed,
    }
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(bytes) if bytes.len() <= RESPONSE_LIMIT_BYTES - body.len() => {
                body.extend_from_slice(&bytes)
            }
            Ok(_) => return ResultKind::Failed,
            Err(error) => return ResultKind::transport(&error),
        }
    }
    let Ok(value) = serde_json::from_slice::<Value>(&body) else {
        return ResultKind::Failed;
    };
    if value.get("error").is_none_or(Value::is_null)
        && value
            .pointer("/usage/completion_tokens")
            .and_then(Value::as_u64)
            == Some(1)
        && value
            .get("choices")
            .and_then(Value::as_array)
            .is_some_and(|choices| {
                choices.len() == 1
                    && matches!(
                        choices[0].get("finish_reason").and_then(Value::as_str),
                        Some("length" | "stop")
                    )
                    && (choices[0]
                        .get("text")
                        .and_then(Value::as_str)
                        .is_some_and(|s| !s.is_empty())
                        || choices[0]
                            .get("token_ids")
                            .and_then(Value::as_array)
                            .is_some_and(|s| s.len() == 1 && s[0].is_u64()))
            })
    {
        ResultKind::Ok
    } else {
        ResultKind::Failed
    }
}

pub(crate) struct WorkerCanary {
    stop: CancellationToken,
    task: Option<JoinHandle<()>>,
}

impl WorkerCanary {
    pub(crate) fn start(
        registry: Arc<WorkerRegistry>,
        client: reqwest::Client,
        interval: Duration,
        timeout: Duration,
    ) -> Option<Self> {
        if interval.is_zero() || !registry.enable_canary() {
            return None;
        }
        let stop = CancellationToken::new();
        let cancelled = stop.clone();
        let task = tokio::spawn(async move {
            let mut ticks = time::interval_at(Instant::now() + interval, interval);
            ticks.set_missed_tick_behavior(time::MissedTickBehavior::Delay);
            let metric = gauge!("vllm_router_worker_idle_seconds");
            loop {
                tokio::select! {
                    biased;
                    _ = cancelled.cancelled() => break,
                    _ = ticks.tick() => {}
                }
                tokio::select! {
                    biased;
                    _ = cancelled.cancelled() => break,
                    _ = sweep(&registry, &client, interval, timeout, &metric) => {}
                }
            }
            registry.disable_canary();
            metric.set(0.);
        });
        Some(Self {
            stop,
            task: Some(task),
        })
    }

    pub(crate) async fn shutdown(mut self) {
        self.stop.cancel();
        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
    }
}

impl Drop for WorkerCanary {
    fn drop(&mut self) {
        self.stop.cancel();
    }
}

async fn sweep(
    registry: &WorkerRegistry,
    client: &reqwest::Client,
    interval: Duration,
    timeout: Duration,
    metric: &metrics::Gauge,
) {
    let workers = registry.get_all_with_ids();
    let now = Instant::now();
    let oldest = workers
        .iter()
        .filter_map(|(_, worker)| worker.canary_state()?.idle_seconds(now))
        .fold(0., f64::max);
    metric.set(oldest);
    stream::iter(workers)
        .for_each_concurrent(CONCURRENCY, |(id, worker)| async move {
            run_one(registry, client, &id, &worker, interval, timeout).await;
        })
        .await;
}

async fn run_one(
    registry: &WorkerRegistry,
    client: &reqwest::Client,
    id: &WorkerId,
    worker: &Arc<dyn Worker>,
    interval: Duration,
    timeout: Duration,
) {
    let Some(probe) = worker
        .canary_state()
        .and_then(|state| state.reserve(Instant::now(), interval))
    else {
        return;
    };
    if registry.with_registered_worker(id, worker, || {}).is_none() {
        return;
    }
    let result = tokio::select! {
        biased;
        _ = probe.cancel.cancelled() => return,
        result = generate(client, worker.as_ref(), timeout) => result,
    };
    registry.with_registered_worker(id, worker, || {
        let was_healthy = worker.is_healthy();
        let result = probe.apply(worker.as_ref(), result);
        if worker.is_healthy() != was_healthy {
            registry.notify_worker_state_change();
        }
        result.record();
    });
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, DPAwareWorker, HealthConfig};
    use axum::{
        extract::State,
        http::{HeaderMap, StatusCode},
        response::{IntoResponse, Response},
        routing::{get, post},
        Json, Router,
    };
    use tokio::{
        net::TcpListener,
        sync::{mpsc, oneshot},
    };

    struct Call {
        headers: HeaderMap,
        body: Value,
        reply: oneshot::Sender<Response>,
    }

    struct Mock {
        calls: mpsc::Receiver<Call>,
        url: String,
        task: JoinHandle<()>,
    }

    impl Mock {
        async fn start() -> Self {
            let (send, calls) = mpsc::channel(16);
            let app = Router::new()
                .route("/health", get(|| async { StatusCode::OK }))
                .route(
                    "/v1/completions",
                    post(
                        |State(send): State<mpsc::Sender<Call>>,
                         headers: HeaderMap,
                         Json(body): Json<Value>| async move {
                            let (reply, response) = oneshot::channel();
                            send.send(Call {
                                headers,
                                body,
                                reply,
                            })
                            .await
                            .unwrap();
                            response.await.unwrap_or_else(|_| {
                                StatusCode::INTERNAL_SERVER_ERROR.into_response()
                            })
                        },
                    ),
                )
                .with_state(send);
            let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            Self { calls, url, task }
        }

        fn regular(&self) -> Arc<dyn Worker> {
            Arc::new(
                BasicWorker::new(self.url.clone(), WorkerType::Regular)
                    .with_health_config(test_config()),
            )
        }
    }

    impl Drop for Mock {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    fn test_config() -> HealthConfig {
        HealthConfig {
            failure_threshold: 2,
            success_threshold: 2,
            ..Default::default()
        }
    }

    fn served_one_token() -> Response {
        Json(json!({
            "choices": [{"text":" x","token_ids":[42],"finish_reason":"length"}],
            "usage": {"completion_tokens":1}
        }))
        .into_response()
    }

    async fn probe_and_reply(
        registry: &WorkerRegistry,
        client: &reqwest::Client,
        id: &WorkerId,
        worker: &Arc<dyn Worker>,
        mock: &mut Mock,
        response: Response,
    ) {
        tokio::join!(
            run_one(
                registry,
                client,
                id,
                worker,
                Duration::ZERO,
                Duration::from_secs(5),
            ),
            async {
                let call = mock.calls.recv().await.unwrap();
                assert!(!call.headers.contains_key("authorization"));
                assert_eq!(call.body["max_tokens"], 1);
                assert_eq!(call.body["stream"], false);
                assert!(call.body.get("kv_transfer_params").is_none());
                call.reply.send(response).unwrap();
            }
        );
    }

    #[tokio::test]
    async fn validates_generation_evidence_and_requires_separate_recovery() {
        tokio::time::timeout(Duration::from_secs(15), async {
            let mut mock = Mock::start().await;
            let registry = Arc::new(WorkerRegistry::new());
            let worker = mock.regular();
            let id = registry.register(worker.clone());
            let client = reqwest::Client::new();

            assert!(WorkerCanary::start(
                registry.clone(),
                client.clone(),
                Duration::ZERO,
                Duration::from_secs(1),
            )
            .is_none());
            run_one(
                &registry,
                &client,
                &id,
                &worker,
                Duration::ZERO,
                Duration::from_secs(1),
            )
            .await;
            assert!(mock.calls.try_recv().is_err());

            assert!(registry.enable_canary());
            let revision = registry.revision();
            for _ in 0..3 {
                probe_and_reply(
                    &registry,
                    &client,
                    &id,
                    &worker,
                    &mut mock,
                    Json(json!({"choices":[]})).into_response(),
                )
                .await;
                assert!(
                    worker.is_healthy(),
                    "an unverified 2xx has no generation evidence"
                );
            }
            probe_and_reply(
                &registry,
                &client,
                &id,
                &worker,
                &mut mock,
                served_one_token(),
            )
            .await;
            for attempt in 0..2 {
                probe_and_reply(
                    &registry,
                    &client,
                    &id,
                    &worker,
                    &mut mock,
                    StatusCode::INTERNAL_SERVER_ERROR.into_response(),
                )
                .await;
                worker.check_health_async().await.unwrap();
                assert_eq!(worker.is_healthy(), attempt == 0);
            }
            assert_ne!(registry.revision(), revision);
            // More passive successes, including its recovery threshold, do not
            // recover the generation health gate.
            for _ in 0..3 {
                worker.check_health_async().await.unwrap();
                assert!(!worker.is_healthy());
            }
            for attempt in 0..2 {
                probe_and_reply(
                    &registry,
                    &client,
                    &id,
                    &worker,
                    &mut mock,
                    served_one_token(),
                )
                .await;
                assert_eq!(worker.is_healthy(), attempt == 1);
            }
            assert_eq!(worker.load(), 0);
            assert_eq!(worker.processed_requests(), 0);
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn user_traffic_and_removal_fence_in_flight_probes() {
        tokio::time::timeout(Duration::from_secs(15), async {
            let mut mock = Mock::start().await;
            let registry = Arc::new(WorkerRegistry::new());
            assert!(registry.enable_canary());
            let worker = mock.regular();
            let id = registry.register(worker.clone());
            let client = reqwest::Client::new();
            probe_and_reply(
                &registry,
                &client,
                &id,
                &worker,
                &mut mock,
                served_one_token(),
            )
            .await;
            probe_and_reply(
                &registry,
                &client,
                &id,
                &worker,
                &mut mock,
                StatusCode::INTERNAL_SERVER_ERROR.into_response(),
            )
            .await;

            let traffic = UserTraffic::begin(worker.as_ref()).unwrap();
            run_one(
                &registry,
                &client,
                &id,
                &worker,
                Duration::ZERO,
                Duration::from_secs(5),
            )
            .await;
            assert!(mock.calls.try_recv().is_err());
            drop(traffic);

            let probe = |registry: Arc<WorkerRegistry>, id: WorkerId, worker: Arc<dyn Worker>| {
                let client = client.clone();
                tokio::spawn(async move {
                    run_one(
                        &registry,
                        &client,
                        &id,
                        &worker,
                        Duration::ZERO,
                        Duration::from_secs(10),
                    )
                    .await;
                })
            };
            let running = probe(registry.clone(), id.clone(), worker.clone());
            let call = mock.calls.recv().await.unwrap();
            run_one(
                &registry,
                &client,
                &id,
                &worker,
                Duration::ZERO,
                Duration::from_secs(5),
            )
            .await;
            assert!(mock.calls.try_recv().is_err());
            // Also reject a stale result if a whole request starts and finishes
            // during the probe, leaving the worker idle again.
            drop(UserTraffic::begin(worker.as_ref()).unwrap());
            call.reply
                .send(StatusCode::INTERNAL_SERVER_ERROR.into_response())
                .unwrap();
            running.await.unwrap();
            assert!(worker.is_healthy());

            let running = probe(registry.clone(), id.clone(), worker.clone());
            let _pending_reply = mock.calls.recv().await.unwrap();
            registry.remove(&id).unwrap();
            running.await.unwrap();
            assert!(worker
                .canary_state()
                .unwrap()
                .idle_seconds(Instant::now())
                .is_none());
            assert!(worker.is_healthy());

            let second = mock.regular();
            let id = registry.register(second.clone());
            probe_and_reply(
                &registry,
                &client,
                &id,
                &second,
                &mut mock,
                StatusCode::INTERNAL_SERVER_ERROR.into_response(),
            )
            .await;
            assert!(second.is_healthy());
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn router_canary_respects_configured_and_unbound_registrations() {
        use crate::{
            config::{PolicyConfig, RouterConfig, RoutingMode},
            routers::{router_manager::RouterManager, RouterFactory},
            server::{build_app, AppContext, AppState},
        };

        time::timeout(Duration::from_secs(15), async {
            let mut configured = Mock::start().await;
            let mut unbound = Mock::start().await;
            let client = reqwest::Client::new();
            let config = RouterConfig {
                mode: RoutingMode::Regular {
                    worker_urls: vec![configured.url.clone()],
                },
                api_key: Some("configured-worker-test-key".into()),
                policy: PolicyConfig::RoundRobin,
                ..Default::default()
            };
            let context =
                Arc::new(AppContext::new(config, client.clone(), 32, None, vec![]).unwrap());
            // The actual regular Router sets a trusted credential on its
            // configured worker. IGW registrations share the registry but
            // do not inherit that binding.
            let router = Arc::from(RouterFactory::create_router(&context).await.unwrap());
            let registry = context.worker_registry.clone();
            let manager = Arc::new(RouterManager::new(
                context.router_config.clone(),
                client.clone(),
                registry.clone(),
                context.policy_registry.clone(),
            ));
            let app = build_app(
                Arc::new(AppState {
                    router,
                    context,
                    concurrency_queue_tx: None,
                    router_manager: Some(manager),
                }),
                1024 * 1024,
                vec![],
                vec![],
                false,
            );
            let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
            let api = format!("http://{}", listener.local_addr().unwrap());
            let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

            let canary = WorkerCanary::start(
                registry,
                client.clone(),
                Duration::from_millis(50),
                Duration::from_secs(3),
            )
            .unwrap();
            client
                .post(format!("{api}/workers"))
                .json(&json!({"url":unbound.url,"model_id":"unbound","worker_type":"regular"}))
                .send()
                .await
                .unwrap()
                .error_for_status()
                .unwrap();

            let (configured_call, unbound_call) =
                tokio::join!(configured.calls.recv(), unbound.calls.recv(),);
            let configured_call = configured_call.unwrap();
            let unbound_call = unbound_call.unwrap();
            assert_eq!(
                configured_call.headers["authorization"],
                "Bearer configured-worker-test-key"
            );
            assert!(!unbound_call.headers.contains_key("authorization"));
            for call in [configured_call, unbound_call] {
                assert_eq!(call.body["max_tokens"], 1);
                assert!(call.body.get("kv_transfer_params").is_none());
                call.reply.send(served_one_token()).unwrap();
            }
            canary.shutdown().await;
            server.abort();
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn live_router_stream_holds_idle_gate_through_disconnect_and_shutdown() {
        use crate::{
            config::{PolicyConfig, RouterConfig, RoutingMode},
            routers::RouterFactory,
            server::{build_app, AppContext, AppState},
        };
        use axum::body::{Body, Bytes};

        time::timeout(Duration::from_secs(15), async {
            let mut mock = Mock::start().await;
            let client = reqwest::Client::new();
            let config = RouterConfig {
                mode: RoutingMode::Regular {
                    worker_urls: vec![mock.url.clone()],
                },
                policy: PolicyConfig::RoundRobin,
                ..Default::default()
            };
            let context =
                Arc::new(AppContext::new(config, client.clone(), 32, None, vec![]).unwrap());
            let router = Arc::from(RouterFactory::create_router(&context).await.unwrap());
            let registry = context.worker_registry.clone();
            assert!(registry.enable_canary());
            let (id, worker) = registry.get_all_with_ids().pop().unwrap();
            let app = build_app(
                Arc::new(AppState {
                    router,
                    context,
                    concurrency_queue_tx: None,
                    router_manager: None,
                }),
                1024 * 1024,
                vec![],
                vec![],
                false,
            );
            let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
            let url = format!("http://{}/v1/completions", listener.local_addr().unwrap());
            let serving = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let user = tokio::spawn({
                let client = client.clone();
                async move {
                    client
                        .post(url)
                        .json(&json!({"prompt":"user","max_tokens":8,"stream":true}))
                        .send()
                        .await
                        .unwrap()
                        .error_for_status()
                        .unwrap()
                }
            });
            let call = mock.calls.recv().await.unwrap();
            assert_eq!(call.body["max_tokens"], 8);
            let first = stream::once(async {
                Ok::<_, std::io::Error>(Bytes::from_static(
                    b"data: {\"choices\":[{\"text\":\"x\"}]}\n\n",
                ))
            });
            call.reply
                .send(
                    (
                        [("content-type", "text/event-stream")],
                        Body::from_stream(first.chain(stream::pending())),
                    )
                        .into_response(),
                )
                .unwrap();
            let mut user = user.await.unwrap();
            assert!(user.chunk().await.unwrap().is_some());
            run_one(
                &registry,
                &client,
                &id,
                &worker,
                Duration::ZERO,
                Duration::from_secs(1),
            )
            .await;
            assert!(
                mock.calls.try_recv().is_err(),
                "a live untracked-policy stream is busy"
            );
            drop(user);
            let state = worker.canary_state().unwrap();
            time::timeout(Duration::from_secs(5), async {
                loop {
                    if state
                        .inner
                        .traffic
                        .lock()
                        .as_ref()
                        .is_some_and(|s| s.users == 0)
                    {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();

            registry.disable_canary();
            let canary = WorkerCanary::start(
                registry.clone(),
                client,
                Duration::from_millis(25),
                Duration::from_secs(10),
            )
            .unwrap();
            let pending = mock.calls.recv().await.unwrap();
            assert_eq!(pending.body["max_tokens"], 1);
            time::timeout(Duration::from_secs(2), canary.shutdown())
                .await
                .unwrap();
            assert!(state.idle_seconds(Instant::now()).is_none());
            assert!(worker.is_healthy());
            serving.abort();
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn idle_sweep_caps_parallelism_and_probes_time_out() {
        time::timeout(Duration::from_secs(15), async {
            let registry = Arc::new(WorkerRegistry::new());
            assert!(registry.enable_canary());
            let mut mocks = Vec::new();
            let mut workers = Vec::new();
            for _ in 0..=CONCURRENCY {
                let mock = Mock::start().await;
                let worker = mock.regular();
                registry.register(worker.clone());
                workers.push(worker);
                mocks.push(mock);
            }
            let client = reqwest::Client::new();
            let sweep_task = tokio::spawn({
                let registry = registry.clone();
                let client = client.clone();
                async move {
                    sweep(
                        &registry,
                        &client,
                        Duration::ZERO,
                        Duration::from_secs(5),
                        &gauge!("vllm_router_worker_idle_seconds"),
                    )
                    .await;
                }
            });
            let mut receivers = mocks
                .iter_mut()
                .map(|mock| Box::pin(mock.calls.recv()))
                .collect::<Vec<_>>();
            let mut held = Vec::new();
            for _ in 0..CONCURRENCY {
                let (call, _, remaining) = futures_util::future::select_all(receivers).await;
                held.push(call.unwrap());
                receivers = remaining;
            }
            // The current-thread sweep has reserved its full first batch,
            // and all those requests will remain pending until we reply.
            let in_flight = workers
                .iter()
                .filter(|worker| {
                    worker
                        .canary_state()
                        .unwrap()
                        .inner
                        .traffic
                        .lock()
                        .as_ref()
                        .unwrap()
                        .probing
                })
                .count();
            assert_eq!(in_flight, CONCURRENCY);
            held.pop().unwrap().reply.send(served_one_token()).unwrap();
            let (call, _, remaining) = futures_util::future::select_all(receivers).await;
            assert!(remaining.is_empty());
            drop(remaining);
            held.push(call.unwrap());
            for call in held {
                call.reply.send(served_one_token()).unwrap();
            }
            sweep_task.await.unwrap();

            let (result, _held_response) = tokio::join!(
                generate(&client, workers[0].as_ref(), Duration::from_secs(1)),
                async { mocks[0].calls.recv().await.unwrap() }
            );
            assert_eq!(result, ResultKind::Timeout);
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn only_regular_http_roles_are_eligible_and_dp_is_pinned() {
        tokio::time::timeout(Duration::from_secs(10), async {
            let mut mock = Mock::start().await;
            let registry = Arc::new(WorkerRegistry::new());
            assert!(registry.enable_canary());
            let client = reqwest::Client::new();
            for (url, role) in [
                (
                    mock.url.clone(),
                    WorkerType::Prefill {
                        bootstrap_port: None,
                    },
                ),
                (mock.url.clone(), WorkerType::Decode),
                ("grpc://127.0.0.1:12345".into(), WorkerType::Regular),
            ] {
                let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(url, role));
                let id = registry.register(worker.clone());
                assert!(UserTraffic::begin(worker.as_ref()).is_none());
                run_one(
                    &registry,
                    &client,
                    &id,
                    &worker,
                    Duration::ZERO,
                    Duration::from_secs(1),
                )
                .await;
                assert!(mock.calls.try_recv().is_err());
            }
            let worker: Arc<dyn Worker> = Arc::new(
                DPAwareWorker::new(mock.url.clone(), 1, 2, WorkerType::Regular)
                    .with_worker_api_key(Some(Arc::from("rank-test-key"))),
            );
            let id = registry.register(worker.clone());
            tokio::join!(
                run_one(
                    &registry,
                    &client,
                    &id,
                    &worker,
                    Duration::ZERO,
                    Duration::from_secs(5),
                ),
                async {
                    let call = mock.calls.recv().await.unwrap();
                    assert_eq!(call.headers.get("x-data-parallel-rank").unwrap(), "1");
                    assert_eq!(call.headers["authorization"], "Bearer rank-test-key");
                    call.reply.send(served_one_token()).unwrap();
                }
            );
        })
        .await
        .unwrap();
    }
}
