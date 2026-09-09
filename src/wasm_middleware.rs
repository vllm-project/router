//! Startup-loaded WASM OnRequest middleware runtime for vLLM Router.
//!
//! Loads one Component Model artifact, optionally verifies a SHA-256 digest,
//! and executes it under bounded workers with a fresh Store per request.
//! The default attach point is `POST /v1/chat/completions`; additional paths
//! can be configured. Failures are fail-closed (no silent bypass).

use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    extract::{Request, State},
    http::{header, HeaderName, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokio::sync::{oneshot, Semaphore};
use tracing::{debug, warn};
use wasmtime::component::{Component, Linker, ResourceTable};
use wasmtime::{
    Config, Engine, InstanceAllocationStrategy, PoolingAllocationConfig, Store, StoreLimits,
    StoreLimitsBuilder,
};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiCtxView, WasiView};

wasmtime::component::bindgen!({
    path: "wit",
    world: "middleware",
});

use crate::wasm_middleware::vllm::router_middleware::types::{
    Action as WitAction, Header as WitHeader, ModifyAction as WitModifyAction,
    Request as WitRequest,
};

pub const DEFAULT_MAX_INPUT_BYTES: usize = 10 * 1024 * 1024;
pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 10 * 1024 * 1024;
pub const DEFAULT_MAX_MEMORY_BYTES: usize = 64 * 1024 * 1024;
pub const DEFAULT_EXECUTION_DEADLINE: Duration = Duration::from_millis(100);
pub const DEFAULT_ROUTE: &str = "/v1/chat/completions";
const EPOCH_INTERVAL: Duration = Duration::from_millis(10);

/// Headers plugins are never allowed to mutate.
const IMMUTABLE_HEADERS: &[&str] = &[
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "upgrade",
    "te",
    "trailer",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
    "forwarded",
];

#[derive(Debug, Clone)]
pub struct WasmMiddlewareConfig {
    pub component_path: PathBuf,
    pub sha256_hex: Option<String>,
    pub routes: Vec<String>,
    pub max_input_bytes: usize,
    pub max_output_bytes: usize,
    pub max_memory_bytes: usize,
    pub execution_deadline: Duration,
    pub worker_count: usize,
    pub queue_capacity: usize,
}

impl WasmMiddlewareConfig {
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        let worker_count = std::thread::available_parallelism()
            .map(NonZeroUsize::get)
            .unwrap_or(1)
            .clamp(1, 4);
        Self {
            component_path: path.into(),
            sha256_hex: None,
            routes: vec![DEFAULT_ROUTE.to_string()],
            max_input_bytes: DEFAULT_MAX_INPUT_BYTES,
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
            max_memory_bytes: DEFAULT_MAX_MEMORY_BYTES,
            execution_deadline: DEFAULT_EXECUTION_DEADLINE,
            worker_count,
            queue_capacity: worker_count.saturating_mul(2).max(1),
        }
    }

    pub fn with_sha256_hex(mut self, digest: impl Into<String>) -> Self {
        self.sha256_hex = Some(normalize_digest(&digest.into()));
        self
    }

    pub fn with_routes(mut self, routes: Vec<String>) -> Self {
        self.routes = routes
            .into_iter()
            .map(|r| r.trim().to_string())
            .filter(|r| !r.is_empty())
            .collect();
        if self.routes.is_empty() {
            self.routes.push(DEFAULT_ROUTE.to_string());
        }
        self
    }

    pub fn validate(&self) -> Result<(), WasmMiddlewareError> {
        if self.max_input_bytes == 0 || self.max_output_bytes == 0 {
            return Err(WasmMiddlewareError::InvalidConfig(
                "wasm body limits must be greater than zero".into(),
            ));
        }
        if self.max_memory_bytes < 64 * 1024 {
            return Err(WasmMiddlewareError::InvalidConfig(
                "wasm memory limit must be at least 64KiB".into(),
            ));
        }
        if self.execution_deadline.is_zero() {
            return Err(WasmMiddlewareError::InvalidConfig(
                "wasm execution deadline must be greater than zero".into(),
            ));
        }
        if self.worker_count == 0 || self.queue_capacity == 0 {
            return Err(WasmMiddlewareError::InvalidConfig(
                "wasm worker_count and queue_capacity must be greater than zero".into(),
            ));
        }
        if self.routes.is_empty() {
            return Err(WasmMiddlewareError::InvalidConfig(
                "wasm middleware requires at least one route".into(),
            ));
        }
        if let Some(digest) = &self.sha256_hex {
            if digest.len() != 64 || !digest.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err(WasmMiddlewareError::InvalidConfig(
                    "wasm sha256 digest must be 64 hex characters".into(),
                ));
            }
        }
        Ok(())
    }
}

fn normalize_digest(digest: &str) -> String {
    digest.trim().to_ascii_lowercase()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WasmAction {
    Continue,
    Modify {
        headers_set: Vec<(String, Vec<u8>)>,
        headers_add: Vec<(String, Vec<u8>)>,
        headers_remove: Vec<String>,
        body: Option<Vec<u8>>,
    },
    Reject {
        status: u16,
    },
}

#[derive(Debug, Error)]
pub enum WasmMiddlewareError {
    #[error("invalid wasm middleware config: {0}")]
    InvalidConfig(String),
    #[error("failed to read wasm component {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("wasm component digest mismatch: expected {expected}, got {actual}")]
    DigestMismatch { expected: String, actual: String },
    #[error("failed to configure wasmtime engine: {0}")]
    Engine(String),
    #[error("failed to compile wasm component: {0}")]
    Compile(String),
    #[error("failed to instantiate wasm component: {0}")]
    Instantiate(String),
    #[error("wasm input body exceeds limit ({limit} bytes)")]
    InputTooLarge { limit: usize },
    #[error("wasm output body exceeds limit ({limit} bytes)")]
    OutputTooLarge { limit: usize },
    #[error("wasm execution queue is full")]
    QueueFull,
    #[error("wasm execution timed out after {0:?}")]
    Timeout(Duration),
    #[error("wasm component trap or call failure: {0}")]
    Trap(String),
    #[error("invalid wasm action: {0}")]
    InvalidAction(String),
    #[error("wasm worker pool failed: {0}")]
    Worker(String),
}

struct HostState {
    ctx: WasiCtx,
    table: ResourceTable,
    limits: StoreLimits,
}

impl WasiView for HostState {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.ctx,
            table: &mut self.table,
        }
    }
}

struct InvokeRequest {
    request: WitRequest,
    response: oneshot::Sender<Result<WasmAction, WasmMiddlewareError>>,
}

struct RuntimeInner {
    config: WasmMiddlewareConfig,
    route_set: HashSet<String>,
    queue_tx: async_channel::Sender<InvokeRequest>,
    _workers: Vec<std::thread::JoinHandle<()>>,
    queue_permits: Arc<Semaphore>,
    invocations: AtomicU64,
    continues: AtomicU64,
    modifies: AtomicU64,
    rejects: AtomicU64,
    traps: AtomicU64,
    timeouts: AtomicU64,
}

#[derive(Clone)]
pub struct WasmMiddlewareRuntime {
    inner: Arc<RuntimeInner>,
}

#[derive(Debug, Clone, Default)]
pub struct WasmRuntimeMetrics {
    pub invocations: u64,
    pub continues: u64,
    pub modifies: u64,
    pub rejects: u64,
    pub traps: u64,
    pub timeouts: u64,
}

impl WasmMiddlewareRuntime {
    pub fn load(config: WasmMiddlewareConfig) -> Result<Self, WasmMiddlewareError> {
        config.validate()?;
        let bytes =
            std::fs::read(&config.component_path).map_err(|source| WasmMiddlewareError::Read {
                path: config.component_path.clone(),
                source,
            })?;
        let actual = hex_digest(&bytes);
        if let Some(expected) = config.sha256_hex.as_ref().map(|d| normalize_digest(d)) {
            if expected != actual {
                return Err(WasmMiddlewareError::DigestMismatch { expected, actual });
            }
        }

        let mut engine_config = Config::new();
        engine_config.wasm_component_model(true);
        engine_config.epoch_interruption(true);
        let mut pooling = PoolingAllocationConfig::default();
        pooling.total_component_instances(32);
        pooling.total_core_instances(128);
        pooling.max_core_instances_per_component(8);
        pooling.total_memories(64);
        pooling.max_memories_per_component(2);
        pooling.total_tables(64);
        pooling.max_tables_per_component(4);
        engine_config.allocation_strategy(InstanceAllocationStrategy::Pooling(pooling));

        let engine = Engine::new(&engine_config)
            .map_err(|err| WasmMiddlewareError::Engine(err.to_string()))?;
        let component = Component::new(&engine, &bytes)
            .map_err(|err| WasmMiddlewareError::Compile(err.to_string()))?;

        let mut linker = Linker::new(&engine);
        wasmtime_wasi::p2::add_to_linker_sync(&mut linker)
            .map_err(|err| WasmMiddlewareError::Engine(err.to_string()))?;

        {
            let mut store = make_store(&engine, &config)?;
            let _ = Middleware::instantiate(&mut store, &component, &linker)
                .map_err(|err| WasmMiddlewareError::Instantiate(err.to_string()))?;
        }

        let engine_for_epoch = engine.clone();
        std::thread::Builder::new()
            .name("wasm-middleware-epoch".into())
            .spawn(move || loop {
                std::thread::sleep(EPOCH_INTERVAL);
                engine_for_epoch.increment_epoch();
            })
            .map_err(|err| WasmMiddlewareError::Worker(err.to_string()))?;

        let (queue_tx, queue_rx) = async_channel::bounded::<InvokeRequest>(config.queue_capacity);
        let mut workers = Vec::with_capacity(config.worker_count);
        for worker_id in 0..config.worker_count {
            let queue_rx = queue_rx.clone();
            let engine = engine.clone();
            let component = component.clone();
            let linker = linker.clone();
            let worker_config = config.clone();
            let handle = std::thread::Builder::new()
                .name(format!("wasm-middleware-worker-{worker_id}"))
                .spawn(move || {
                    worker_loop(
                        worker_id,
                        queue_rx,
                        engine,
                        component,
                        linker,
                        worker_config,
                    )
                })
                .map_err(|err| WasmMiddlewareError::Worker(err.to_string()))?;
            workers.push(handle);
        }

        let route_set = config.routes.iter().cloned().collect();
        let queue_permits = Arc::new(Semaphore::new(config.queue_capacity));
        Ok(Self {
            inner: Arc::new(RuntimeInner {
                config,
                route_set,
                queue_tx,
                _workers: workers,
                queue_permits,
                invocations: AtomicU64::new(0),
                continues: AtomicU64::new(0),
                modifies: AtomicU64::new(0),
                rejects: AtomicU64::new(0),
                traps: AtomicU64::new(0),
                timeouts: AtomicU64::new(0),
            }),
        })
    }

    pub fn component_path(&self) -> &Path {
        &self.inner.config.component_path
    }

    pub fn config(&self) -> &WasmMiddlewareConfig {
        &self.inner.config
    }

    pub fn matches_route(&self, path: &str) -> bool {
        self.inner.route_set.contains(path)
    }

    pub fn metrics(&self) -> WasmRuntimeMetrics {
        WasmRuntimeMetrics {
            invocations: self.inner.invocations.load(Ordering::Relaxed),
            continues: self.inner.continues.load(Ordering::Relaxed),
            modifies: self.inner.modifies.load(Ordering::Relaxed),
            rejects: self.inner.rejects.load(Ordering::Relaxed),
            traps: self.inner.traps.load(Ordering::Relaxed),
            timeouts: self.inner.timeouts.load(Ordering::Relaxed),
        }
    }

    pub async fn handle_request(
        &self,
        request: WitRequest,
    ) -> Result<WasmAction, WasmMiddlewareError> {
        if request.body.len() > self.inner.config.max_input_bytes {
            return Err(WasmMiddlewareError::InputTooLarge {
                limit: self.inner.config.max_input_bytes,
            });
        }

        let permit = self
            .inner
            .queue_permits
            .clone()
            .try_acquire_owned()
            .map_err(|_| WasmMiddlewareError::QueueFull)?;

        let (response_tx, response_rx) = oneshot::channel();
        self.inner
            .queue_tx
            .try_send(InvokeRequest {
                request,
                response: response_tx,
            })
            .map_err(|_| WasmMiddlewareError::QueueFull)?;

        let result = response_rx
            .await
            .map_err(|err| WasmMiddlewareError::Worker(err.to_string()))?;
        drop(permit);

        self.inner.invocations.fetch_add(1, Ordering::Relaxed);
        match &result {
            Ok(WasmAction::Continue) => {
                self.inner.continues.fetch_add(1, Ordering::Relaxed);
            }
            Ok(WasmAction::Modify { .. }) => {
                self.inner.modifies.fetch_add(1, Ordering::Relaxed);
            }
            Ok(WasmAction::Reject { .. }) => {
                self.inner.rejects.fetch_add(1, Ordering::Relaxed);
            }
            Err(WasmMiddlewareError::Timeout(_)) => {
                self.inner.timeouts.fetch_add(1, Ordering::Relaxed);
            }
            Err(WasmMiddlewareError::Trap(_)) => {
                self.inner.traps.fetch_add(1, Ordering::Relaxed);
            }
            Err(_) => {}
        }
        result
    }
}

fn hex_digest(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn make_store(
    engine: &Engine,
    config: &WasmMiddlewareConfig,
) -> Result<Store<HostState>, WasmMiddlewareError> {
    let ctx = WasiCtxBuilder::new().build();
    let limits = StoreLimitsBuilder::new()
        .memory_size(config.max_memory_bytes)
        .trap_on_grow_failure(true)
        .build();
    let mut store = Store::new(
        engine,
        HostState {
            ctx,
            table: ResourceTable::new(),
            limits,
        },
    );
    store.limiter(|state| &mut state.limits);
    let ticks =
        (config.execution_deadline.as_millis() / EPOCH_INTERVAL.as_millis().max(1)).max(1) as u64;
    store.set_epoch_deadline(ticks);
    Ok(store)
}

fn worker_loop(
    worker_id: usize,
    queue_rx: async_channel::Receiver<InvokeRequest>,
    engine: Engine,
    component: Component,
    linker: Linker<HostState>,
    config: WasmMiddlewareConfig,
) {
    debug!(worker_id, "wasm middleware worker started");
    while let Ok(request) = queue_rx.recv_blocking() {
        let started = Instant::now();
        let result = execute_once(&engine, &component, &linker, &config, request.request);
        if let Err(err) = &result {
            warn!(
                worker_id,
                elapsed_ms = started.elapsed().as_millis() as u64,
                "wasm middleware invocation failed: {err}"
            );
        }
        let _ = request.response.send(result);
    }
    debug!(worker_id, "wasm middleware worker stopped");
}

fn execute_once(
    engine: &Engine,
    component: &Component,
    linker: &Linker<HostState>,
    config: &WasmMiddlewareConfig,
    request: WitRequest,
) -> Result<WasmAction, WasmMiddlewareError> {
    let mut store = make_store(engine, config)?;
    let bindings = Middleware::instantiate(&mut store, component, linker)
        .map_err(|err| WasmMiddlewareError::Instantiate(err.to_string()))?;

    let action = bindings
        .vllm_router_middleware_on_request()
        .call_handle(&mut store, &request)
        .map_err(|err| map_call_error(err, config.execution_deadline))?;

    map_action(action, config.max_output_bytes)
}

fn map_call_error(err: wasmtime::Error, deadline: Duration) -> WasmMiddlewareError {
    let message = err.to_string();
    if message.contains("epoch")
        || message.contains("interrupt")
        || message.contains("deadline")
        || message.contains("execution time limit")
    {
        WasmMiddlewareError::Timeout(deadline)
    } else {
        WasmMiddlewareError::Trap(message)
    }
}

fn map_action(
    action: WitAction,
    max_output_bytes: usize,
) -> Result<WasmAction, WasmMiddlewareError> {
    match action {
        WitAction::Continue => Ok(WasmAction::Continue),
        WitAction::Reject(status) => {
            if status < 400 {
                return Err(WasmMiddlewareError::InvalidAction(format!(
                    "reject status must be >= 400, got {status}"
                )));
            }
            Ok(WasmAction::Reject { status })
        }
        WitAction::Modify(WitModifyAction {
            headers_set,
            headers_add,
            headers_remove,
            body_replace,
        }) => {
            let body = match body_replace {
                Some(body) => {
                    if body.len() > max_output_bytes {
                        return Err(WasmMiddlewareError::OutputTooLarge {
                            limit: max_output_bytes,
                        });
                    }
                    Some(body)
                }
                None => None,
            };
            Ok(WasmAction::Modify {
                headers_set: headers_set.into_iter().map(|h| (h.name, h.value)).collect(),
                headers_add: headers_add.into_iter().map(|h| (h.name, h.value)).collect(),
                headers_remove,
                body,
            })
        }
    }
}

fn is_immutable_header(name: &str) -> bool {
    IMMUTABLE_HEADERS
        .iter()
        .any(|forbidden| name.eq_ignore_ascii_case(forbidden))
}

fn header_bytes_to_value(value: &[u8]) -> Result<HeaderValue, WasmMiddlewareError> {
    HeaderValue::from_bytes(value).map_err(|err| {
        WasmMiddlewareError::InvalidAction(format!("invalid header value bytes: {err}"))
    })
}

fn apply_header_mutations(
    headers: &mut http::HeaderMap,
    headers_set: &[(String, Vec<u8>)],
    headers_add: &[(String, Vec<u8>)],
    headers_remove: &[String],
) -> Result<(), WasmMiddlewareError> {
    for name in headers_remove {
        if is_immutable_header(name) {
            warn!(header = %name, "ignoring wasm attempt to remove immutable header");
            continue;
        }
        if let Ok(header_name) = HeaderName::from_bytes(name.as_bytes()) {
            headers.remove(header_name);
        }
    }
    for (name, value) in headers_set {
        if is_immutable_header(name) {
            warn!(header = %name, "ignoring wasm attempt to set immutable header");
            continue;
        }
        let header_name = HeaderName::from_bytes(name.as_bytes()).map_err(|err| {
            WasmMiddlewareError::InvalidAction(format!("invalid header name {name}: {err}"))
        })?;
        headers.insert(header_name, header_bytes_to_value(value)?);
    }
    for (name, value) in headers_add {
        if is_immutable_header(name) {
            warn!(header = %name, "ignoring wasm attempt to add immutable header");
            continue;
        }
        let header_name = HeaderName::from_bytes(name.as_bytes()).map_err(|err| {
            WasmMiddlewareError::InvalidAction(format!("invalid header name {name}: {err}"))
        })?;
        headers.append(header_name, header_bytes_to_value(value)?);
    }
    Ok(())
}

fn request_id_from_headers(headers: &http::HeaderMap) -> String {
    for name in [
        "x-request-id",
        "x-correlation-id",
        "x-trace-id",
        "request-id",
    ] {
        if let Some(value) = headers.get(name).and_then(|v| v.to_str().ok()) {
            if !value.is_empty() {
                return value.to_string();
            }
        }
    }
    String::new()
}

fn collect_wit_headers(headers: &http::HeaderMap) -> Vec<WitHeader> {
    headers
        .iter()
        .map(|(name, value)| WitHeader {
            name: name.as_str().to_string(),
            value: value.as_bytes().to_vec(),
        })
        .collect()
}

/// Axum state for the WASM OnRequest middleware layer.
#[derive(Clone)]
pub struct WasmRouteMiddlewareState {
    pub runtime: Arc<WasmMiddlewareRuntime>,
    pub max_payload_size: usize,
}

/// Fail-closed WASM OnRequest adapter. Non-configured paths pass through unchanged.
pub async fn wasm_on_request_middleware(
    State(state): State<WasmRouteMiddlewareState>,
    request: Request,
    next: Next,
) -> Response {
    let path = request.uri().path().to_string();
    if !state.runtime.matches_route(&path) {
        return next.run(request).await;
    }

    let method = request.method().as_str().to_string();
    let query = request.uri().query().unwrap_or_default().to_string();
    let request_id = request_id_from_headers(request.headers());
    let wit_headers = collect_wit_headers(request.headers());

    let (mut parts, body) = request.into_parts();
    let bytes = match axum::body::to_bytes(body, state.max_payload_size).await {
        Ok(bytes) => bytes,
        Err(err) => {
            let message = err.to_string();
            if message.contains("length limit exceeded") {
                warn!(
                    "wasm middleware rejected oversized body (limit {} bytes)",
                    state.max_payload_size
                );
                return StatusCode::PAYLOAD_TOO_LARGE.into_response();
            }
            warn!("Failed to read request body for wasm middleware: {err}");
            return StatusCode::BAD_REQUEST.into_response();
        }
    };

    let wit_request = WitRequest {
        method,
        path,
        query,
        headers: wit_headers,
        body: bytes.to_vec(),
        request_id,
    };

    match state.runtime.handle_request(wit_request).await {
        Ok(WasmAction::Continue) => {
            let request = Request::from_parts(parts, axum::body::Body::from(bytes));
            next.run(request).await
        }
        Ok(WasmAction::Modify {
            headers_set,
            headers_add,
            headers_remove,
            body,
        }) => {
            if let Err(err) = apply_header_mutations(
                &mut parts.headers,
                &headers_set,
                &headers_add,
                &headers_remove,
            ) {
                warn!("wasm middleware returned invalid header mutation: {err}");
                return StatusCode::INTERNAL_SERVER_ERROR.into_response();
            }
            let body = body.unwrap_or_else(|| bytes.to_vec());
            parts.headers.remove(header::CONTENT_LENGTH);
            let request = Request::from_parts(parts, axum::body::Body::from(body));
            next.run(request).await
        }
        Ok(WasmAction::Reject { status }) => {
            warn!(status, "wasm middleware rejected request");
            StatusCode::from_u16(status)
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
                .into_response()
        }
        Err(WasmMiddlewareError::InputTooLarge { limit }) => {
            warn!(limit, "wasm middleware input exceeds configured limit");
            StatusCode::PAYLOAD_TOO_LARGE.into_response()
        }
        Err(WasmMiddlewareError::QueueFull) => {
            warn!("wasm middleware execution queue is full");
            StatusCode::SERVICE_UNAVAILABLE.into_response()
        }
        Err(err) => {
            warn!("wasm middleware failed closed: {err}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

/// Resolve the example component path for tests.
pub fn example_component_artifact_path() -> Option<PathBuf> {
    let example_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/wasm_middleware");
    [
        example_dir.join("wasm_middleware_example.component.wasm"),
        example_dir.join("target/wasm32-wasip2/release/wasm_middleware_example.wasm"),
    ]
    .into_iter()
    .find(|p| p.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, routing::post, Json, Router};
    use http_body_util::BodyExt;
    use serde_json::{json, Value};
    use tower::ServiceExt;

    fn require_component_path() -> PathBuf {
        example_component_artifact_path().unwrap_or_else(|| {
            panic!(
                "WASM component artifact missing; run \
                 `examples/wasm_middleware/build.sh` before \
                 `cargo test --lib wasm_middleware::`"
            )
        })
    }

    #[tokio::test]
    async fn example_plugin_sets_marker_header() {
        let path = require_component_path();
        let runtime = WasmMiddlewareRuntime::load(WasmMiddlewareConfig::from_path(path))
            .expect("load wasm runtime");
        let action = runtime
            .handle_request(WitRequest {
                method: "POST".into(),
                path: "/v1/chat/completions".into(),
                query: String::new(),
                headers: Vec::new(),
                body: br#"{"messages":[]}"#.to_vec(),
                request_id: "req-1".into(),
            })
            .await
            .expect("invoke");
        match action {
            WasmAction::Modify {
                headers_set, body, ..
            } => {
                assert!(body.is_none());
                assert!(headers_set.iter().any(|(name, value)| {
                    name.eq_ignore_ascii_case("x-wasm-middleware") && value == b"example"
                }));
            }
            other => panic!("expected Modify, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn example_plugin_rejects_marker_body() {
        let path = require_component_path();
        let runtime = WasmMiddlewareRuntime::load(WasmMiddlewareConfig::from_path(path))
            .expect("load wasm runtime");
        let action = runtime
            .handle_request(WitRequest {
                method: "POST".into(),
                path: "/v1/chat/completions".into(),
                query: String::new(),
                headers: Vec::new(),
                body: br#"{"note":"__wasm_reject__"}"#.to_vec(),
                request_id: "req-2".into(),
            })
            .await
            .expect("invoke");
        assert_eq!(action, WasmAction::Reject { status: 400 });
    }

    #[test]
    fn digest_mismatch_fails_startup() {
        let path = require_component_path();
        match WasmMiddlewareRuntime::load(
            WasmMiddlewareConfig::from_path(path).with_sha256_hex(
                "0000000000000000000000000000000000000000000000000000000000000000",
            ),
        ) {
            Err(WasmMiddlewareError::DigestMismatch { .. }) => {}
            Ok(_) => panic!("expected digest mismatch"),
            Err(other) => panic!("expected digest mismatch, got {other}"),
        }
    }

    #[test]
    fn input_too_large_is_rejected_before_queue() {
        let path = require_component_path();
        let mut config = WasmMiddlewareConfig::from_path(path);
        config.max_input_bytes = 8;
        let runtime = WasmMiddlewareRuntime::load(config).expect("load");
        let err = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(runtime.handle_request(WitRequest {
                method: "POST".into(),
                path: "/v1/chat/completions".into(),
                query: String::new(),
                headers: Vec::new(),
                body: b"0123456789".to_vec(),
                request_id: String::new(),
            }))
            .expect_err("too large");
        assert!(matches!(err, WasmMiddlewareError::InputTooLarge { .. }));
    }

    async fn echo_headers(request: Request) -> Json<Value> {
        let marker = request
            .headers()
            .get("x-wasm-middleware")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        Json(json!({ "x-wasm-middleware": marker }))
    }

    #[tokio::test]
    async fn axum_layer_applies_on_configured_route() {
        let path = require_component_path();
        let runtime = Arc::new(
            WasmMiddlewareRuntime::load(WasmMiddlewareConfig::from_path(path)).expect("load"),
        );
        let app = Router::new()
            .route("/v1/chat/completions", post(echo_headers))
            .route("/v1/completions", post(echo_headers))
            .layer(axum::middleware::from_fn_with_state(
                WasmRouteMiddlewareState {
                    runtime,
                    max_payload_size: 1024 * 1024,
                },
                wasm_on_request_middleware,
            ));

        let chat = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"messages":[]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(chat.status(), StatusCode::OK);
        let chat_body = chat.into_body().collect().await.unwrap().to_bytes();
        let chat_json: Value = serde_json::from_slice(&chat_body).unwrap();
        assert_eq!(chat_json["x-wasm-middleware"], "example");

        let other = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"prompt":"hi"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(other.status(), StatusCode::OK);
        let other_body = other.into_body().collect().await.unwrap().to_bytes();
        let other_json: Value = serde_json::from_slice(&other_body).unwrap();
        assert_eq!(other_json["x-wasm-middleware"], "");
    }
}
