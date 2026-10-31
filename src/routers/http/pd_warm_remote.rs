//! Explicitly gated, exact-token NIXL multi-turn orchestration.
//! Metadata is engine-owned and single-use. Neither leases nor cost estimates grant capacity.
use super::{dp_utils, pd_program_adapter::PdProgramRuntime, program_adapter::ProgramCompletion};
use crate::config::types::PdProgramSchedulingConfig;
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::program_scheduling::pd_cost::{bounded_pd_duration, solve_pd_ttl, PdPath};
use crate::program_scheduling::{ProgramDispatch, ProgramRef, ProgramScheduler, ProgramTarget};
use crate::routers::header_utils;
use axum::{
    body::Body,
    http::{HeaderMap, StatusCode},
    response::Response,
};
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, VecDeque},
    pin::Pin,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub(super) struct EngineIdentity {
    protocol_revision: String,
    instance_epoch: String,
    profile: String,
    dp_rank: usize,
    bidirectional: bool,
    fail_on_load_error: bool,
    read_margin_millis: u64,
    local_miss_guard: bool,
    local_prefix_probe: bool,
    store: bool,
}
#[derive(Debug)]
struct Candidate {
    topology_revision: u64,
    produced_attempt: u64,
    produced_placement: u64,
    source: ProgramTarget,
    engine: EngineIdentity,
    params: Value,
    prefix: Vec<u32>,
    identity: [u8; 32],
    expires: Instant,
}
#[derive(Debug, Default, Serialize)]
struct Feedback {
    samples: usize,
    errors: usize,
    mean_residual: f64,
    max_absolute_error: f64,
    external_cached_tokens: u64,
    local_cached_tokens: u64,
    computed_tokens: u64,
}
#[derive(Debug, Default)]
struct WarmRemoteState {
    candidates: HashMap<ProgramRef, Candidate>,
    /// (base_url, dp_rank) -> (capabilities, fetched_at, topology_revision).
    identities: HashMap<(String, Option<usize>), (EngineIdentity, Instant, u64)>,
    invalidation_revision: u64,
    source_revisions: HashMap<(String, Option<usize>), u64>,
    read_outcomes: VecDeque<bool>,
    residuals: HashMap<String, VecDeque<f64>>,
    feedback: HashMap<String, Feedback>,
    counters: HashMap<&'static str, u64>,
    local_gates: HashMap<String, Arc<Semaphore>>,
}
#[derive(Debug)]
pub(super) struct PdWarmRemoteRuntime {
    pub config: PdProgramSchedulingConfig,
    state: Mutex<WarmRemoteState>,
    loss_epoch: std::sync::atomic::AtomicU64,
    event_tracking: std::sync::atomic::AtomicBool,
}
impl PdWarmRemoteRuntime {
    pub fn new(config: PdProgramSchedulingConfig) -> Self {
        Self {
            config,
            state: Mutex::new(WarmRemoteState::default()),
            loss_epoch: std::sync::atomic::AtomicU64::new(crate::kv_index::signal_loss_epoch()),
            event_tracking: std::sync::atomic::AtomicBool::new(false),
        }
    }
    pub fn set_event_tracking(&self, enabled: bool) {
        self.event_tracking
            .store(enabled, std::sync::atomic::Ordering::Release);
    }
    fn refresh_loss(&self) {
        use std::sync::atomic::Ordering;
        if !self.event_tracking.load(Ordering::Acquire) {
            return;
        }
        let current = crate::kv_index::signal_loss_epoch();
        if self.loss_epoch.swap(current, Ordering::AcqRel) != current {
            self.invalidate_all();
        }
    }
    fn count(&self, key: &'static str) {
        let mut s = self.state.lock();
        *s.counters.entry(key).or_default() += 1;
    }
    pub fn diagnostics(&self) -> Value {
        let s = self.state.lock();
        json!({"protocol_revision":self.config.protocol_revision,"warm_enabled":self.config.warm_remote,"local_enabled":self.config.local_append,"automatic_ttl":self.config.automatic_ttl,"candidate_count":s.candidates.len(),"counters":s.counters,"feedback":s.feedback,"calibration_revision":self.config.calibration.as_ref().map(|c|&c.revision)})
    }
    /// Bounded router-side capability cache.
    ///
    /// The engine-side `/pd_capabilities` route may need an EngineCore utility
    /// round-trip; calling it twice per request stalls under load. Cache by
    /// (base_url, dp_rank) for `capability_cache_seconds`, fenced by the worker
    /// registry revision so a restart or topology change re-probes.
    async fn identify(
        &self,
        client: &reqwest::Client,
        target: &ProgramTarget,
        headers: Option<&HeaderMap>,
        topology_revision: u64,
    ) -> Result<EngineIdentity, String> {
        let key = (target.base_url.clone(), target.dp_rank);
        let cached = {
            let state = self.state.lock();
            state.identities.get(&key).cloned()
        };
        if let Some((identity, fetched_at, revision)) = cached {
            if revision == topology_revision
                && self.config.capability_cache_seconds > 0.0
                && fetched_at.elapsed()
                    < Duration::from_secs_f64(self.config.capability_cache_seconds)
            {
                self.count("capability_cache_hit");
                return Ok(identity);
            }
        }
        let identity = identify_uncached(client, target, headers, &self.config).await?;
        {
            let mut state = self.state.lock();
            state
                .identities
                .insert(key, (identity.clone(), Instant::now(), topology_revision));
        }
        Ok(identity)
    }
    pub fn invalidate_target(&self, base: &str, rank: Option<usize>) {
        let mut s = self.state.lock();
        s.identities.retain(|(cached_base, cached_rank), _| {
            cached_base != base || rank.is_some_and(|r| *cached_rank != Some(r))
        });
        let revision = s
            .source_revisions
            .entry((base.to_string(), rank))
            .or_default();
        *revision = revision.wrapping_add(1);
        s.candidates.retain(|_, c| {
            c.source.base_url != base || rank.is_some_and(|r| c.source.dp_rank.unwrap_or(0) != r)
        });
        *s.counters.entry("source_invalidated").or_default() += 1;
    }
    pub fn invalidate_all(&self) {
        let mut s = self.state.lock();
        s.invalidation_revision = s.invalidation_revision.wrapping_add(1);
        s.candidates.clear();
        s.identities.clear();
        *s.counters.entry("event_loss").or_default() += 1;
    }
    fn revision_for(&self, target: &ProgramTarget) -> (u64, u64, u64) {
        let state = self.state.lock();
        (
            state.invalidation_revision,
            *state
                .source_revisions
                .get(&(target.base_url.clone(), None))
                .unwrap_or(&0),
            *state
                .source_revisions
                .get(&(target.base_url.clone(), Some(target.dp_rank.unwrap_or(0))))
                .unwrap_or(&0),
        )
    }
    pub fn retain_sources(&self, targets: &[ProgramTarget]) {
        let mut s = self.state.lock();
        s.candidates
            .retain(|_, c| c.expires > Instant::now() && targets.contains(&c.source));
        s.source_revisions.retain(|(base, rank), _| {
            targets
                .iter()
                .any(|t| &t.base_url == base && rank.is_none_or(|r| t.dp_rank.unwrap_or(0) == r))
        });
        s.local_gates
            .retain(|id, gate| targets.iter().any(|t| &t.id == id) || Arc::strong_count(gate) > 1);
        s.residuals
            .retain(|key, _| targets.iter().any(|t| key.ends_with(&format!(":{}", t.id))));
        s.feedback
            .retain(|key, _| targets.iter().any(|t| key.ends_with(&format!(":{}", t.id))));
    }
    fn claim(
        &self,
        dispatch: &ProgramDispatch,
        prompt: &[u32],
        identity: [u8; 32],
    ) -> Option<Candidate> {
        self.refresh_loss();
        let mut s = self.state.lock();
        let c = s.candidates.remove(dispatch.program())?;
        let expiry_ok = c.expires
            > Instant::now()
                + Duration::from_secs_f64(
                    self.config
                        .lease_margin_seconds
                        .max(c.engine.read_margin_millis as f64 / 1000.0),
                );
        let identity_ok = c.identity == identity;
        let prefix_ok = prompt.starts_with(&c.prefix);
        let attempt_ok = c.produced_attempt < dispatch.attempt_id();
        let placement_ok = c.produced_placement <= dispatch.placement_epoch();
        if !(expiry_ok && identity_ok && prefix_ok && attempt_ok && placement_ok) {
            let reason = if !expiry_ok {
                "candidate_rejected_expiry"
            } else if !identity_ok {
                "candidate_rejected_identity"
            } else if !prefix_ok {
                "candidate_rejected_prefix"
            } else if !attempt_ok {
                "candidate_rejected_attempt"
            } else {
                "candidate_rejected_placement"
            };
            *s.counters.entry(reason).or_default() += 1;
            *s.counters.entry("candidate_rejected").or_default() += 1;
            return None;
        }
        *s.counters.entry("candidate_claimed").or_default() += 1;
        Some(c)
    }
    fn read_outcome(&self, success: bool) {
        let mut s = self.state.lock();
        if s.read_outcomes.len() == 128 {
            s.read_outcomes.pop_front();
        }
        s.read_outcomes.push_back(success);
    }
    fn read_outcome_in_profile(&self, success: bool, profile: &str) {
        if self
            .config
            .calibration
            .as_ref()
            .is_none_or(|c| c.profile == profile)
        {
            self.read_outcome(success);
        }
    }
    fn residency_probability(&self) -> f64 {
        let s = self.state.lock();
        if s.read_outcomes.len() < self.config.ttl_min_samples {
            return 0.0;
        }
        // Conservative Wilson lower confidence bound, with failures counted once.
        let n = s.read_outcomes.len() as f64;
        let p = s.read_outcomes.iter().filter(|v| **v).count() as f64 / n;
        let z = 1.96_f64;
        ((p + z * z / (2.0 * n) - z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt())
            / (1.0 + z * z / n))
            .max(0.0)
    }
    fn correction(&self, path: PdPath, target: &str) -> f64 {
        let s = self.state.lock();
        s.residuals
            .get(&format!("{path:?}:{target}"))
            .map_or(0.0, |v| {
                if self
                    .config
                    .calibration
                    .as_ref()
                    .is_some_and(|c| v.iter().any(|v| v.abs() > c.max_prediction_error_seconds))
                {
                    f64::NAN
                } else {
                    v.iter().sum::<f64>() / v.len().max(1) as f64
                }
            })
    }
    fn local_permit(&self, id: &str) -> Option<OwnedSemaphorePermit> {
        self.state
            .lock()
            .local_gates
            .entry(id.into())
            .or_insert_with(|| Arc::new(Semaphore::new(self.config.max_local_requests_per_target)))
            .clone()
            .try_acquire_owned()
            .ok()
    }
}

fn tokens(value: &Value) -> Option<Vec<u32>> {
    let v = value.as_array()?;
    v.iter()
        .map(|v| v.as_u64().and_then(|v| u32::try_from(v).ok()))
        .collect()
}
pub(super) fn exact_request(request: &Value, path: &str, max: usize) -> Option<Vec<u32>> {
    if path != "/v1/completions"
        || request.get("return_token_ids") != Some(&Value::Bool(true))
        || request.get("n").and_then(Value::as_u64).unwrap_or(1) != 1
        || request.get("best_of").and_then(Value::as_u64).unwrap_or(1) != 1
        || request
            .get("echo")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        || request.get("prompt_embeds").is_some()
        || request.get("multi_modal_data").is_some()
    {
        return None;
    }
    if [
        "lora_path",
        "lora_name",
        "lora_request",
        "prompt_adapter",
        "multi_modal_uuids",
        "mm_processor_kwargs",
    ]
    .iter()
    .any(|k| request.get(*k).is_some_and(|v| !v.is_null()))
    {
        return None;
    }
    if request.get("stream") == Some(&Value::Bool(true))
        && request.pointer("/stream_options/include_usage") != Some(&Value::Bool(true))
    {
        return None;
    }
    let p = tokens(request.get("prompt")?)?;
    (!p.is_empty() && p.len() <= max).then_some(p)
}
fn request_identity(request: &Value) -> [u8; 32] {
    let mut v = request.clone();
    let o = v.as_object_mut().unwrap();
    for k in [
        "prompt",
        "max_tokens",
        "max_completion_tokens",
        "min_tokens",
        "stream",
        "stream_options",
        "return_token_ids",
        "vllm_xargs",
        "kv_transfer_params",
        "seed",
        "temperature",
        "top_p",
        "top_k",
        "request_id",
    ] {
        o.remove(k);
    }
    Sha256::digest(serde_json::to_vec(&v).unwrap_or_default()).into()
}
fn bounded_metadata(params: &Value, max: usize) -> bool {
    params.is_object()
        && serde_json::to_vec(params).is_ok_and(|v| v.len() <= max)
        && params
            .get("remote_engine_id")
            .and_then(Value::as_str)
            .is_some_and(|s| !s.is_empty() && s.len() <= 256)
        && params
            .get("remote_request_id")
            .and_then(Value::as_str)
            .is_some_and(|s| !s.is_empty() && s.len() <= 512)
        && params
            .get("remote_host")
            .and_then(Value::as_str)
            .is_some_and(|s| !s.is_empty() && s.len() <= 256)
        && params
            .get("remote_port")
            .and_then(Value::as_u64)
            .is_some_and(|p| p > 0 && p <= 65535)
        && params
            .get("remote_block_ids")
            .and_then(Value::as_array)
            .is_some_and(|groups| {
                !groups.is_empty()
                    && groups.len() <= 128
                    && groups.iter().all(|g| {
                        g.as_array()
                            .is_some_and(|ids| ids.iter().all(|id| id.as_u64().is_some()))
                    })
            })
        && params.get("transfer_mode").and_then(Value::as_str) == Some("pull")
}
fn builder(
    client: &reqwest::Client,
    target: &ProgramTarget,
    path: &str,
    headers: Option<&HeaderMap>,
) -> reqwest::RequestBuilder {
    let mut b = dp_utils::add_dp_rank_header(
        client.post(format!("{}{path}", target.base_url)),
        target.dp_rank,
    )
    .header("X-Request-Id", uuid::Uuid::new_v4().to_string());
    // The native two-stage path authenticates to the engines with the
    // operator's OPENAI_API_KEY. Match it when configured, and otherwise
    // fall back to the client's Authorization so client-key deployments
    // keep working.
    let auth = std::env::var("OPENAI_API_KEY")
        .ok()
        .filter(|key| !key.is_empty())
        .and_then(|key| axum::http::HeaderValue::from_str(&format!("Bearer {key}")).ok())
        .or_else(|| {
            headers
                .and_then(|h| h.get(axum::http::header::AUTHORIZATION))
                .cloned()
        });
    if let Some(auth) = auth {
        b = b.header(axum::http::header::AUTHORIZATION, auth);
    }
    header_utils::propagate_trace_headers(b, headers)
}
async fn identify_uncached(
    client: &reqwest::Client,
    target: &ProgramTarget,
    headers: Option<&HeaderMap>,
    config: &PdProgramSchedulingConfig,
) -> Result<EngineIdentity, String> {
    let mut b = dp_utils::add_dp_rank_header(
        client.get(format!("{}/pd_capabilities", target.base_url)),
        target.dp_rank,
    )
    .timeout(Duration::from_secs(30));
    if let Some(auth) = headers.and_then(|h| h.get(axum::http::header::AUTHORIZATION)) {
        b = b.header(axum::http::header::AUTHORIZATION, auth.clone());
    }
    let response = b
        .send()
        .await
        .map_err(|_| "PD capability endpoint unavailable")?;
    if !response.status().is_success() {
        return Err("PD capability endpoint unsupported".into());
    }
    let bytes = read_bounded(response, 65536, None).await?;
    let e: EngineIdentity =
        serde_json::from_slice(&bytes).map_err(|_| "Invalid PD capability manifest")?;
    if e.protocol_revision != config.protocol_revision
        || e.instance_epoch.is_empty()
        || e.instance_epoch.len() > 128
        || e.profile.is_empty()
        || e.profile.len() > 128
        || e.dp_rank != target.dp_rank.unwrap_or(0)
        || e.read_margin_millis > 86400000
    {
        return Err("PD capability identity/rank mismatch".into());
    }
    Ok(e)
}
/// Why an internal prefill call failed.
enum PrefillError {
    /// The prefill target rejected the request; forward this response intact.
    Upstream(reqwest::Response),
    /// Transport or protocol failure detected before any output.
    Failed(String),
}

/// Outcome of the pre-output gate on the decode response.
enum GateOutcome {
    /// Output was produced; track the response as a Program turn.
    Ready(Response),
    /// The decode target rejected the request; forward it untouched.
    Forward(Response),
}

/// How one Program turn ended before response tracking.
enum TurnResult {
    /// A decode response that should be tracked as this Program's turn.
    Dispatch(Response, Instant),
    /// An upstream error response that must be forwarded unchanged.
    Forward(Response),
}

/// Rebuild a forwarded backend response with its original status and headers.
fn forward_backend_response(response: reqwest::Response) -> Response {
    let status = response.status();
    let headers = header_utils::preserve_response_headers(response.headers());
    let body = Body::from_stream(response.bytes_stream());
    let mut result = Response::new(body);
    *result.status_mut() = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    *result.headers_mut() = headers;
    result
}

async fn read_bounded(
    response: reqwest::Response,
    max: usize,
    stage: Option<&'static str>,
) -> Result<Vec<u8>, String> {
    let mut body = response.bytes_stream();
    let mut bytes = Vec::new();
    while let Some(chunk) = body.next().await {
        let chunk = chunk.map_err(|_| "PD body transport failure")?;
        if bytes.len().saturating_add(chunk.len()) > max {
            if let Some(stage) = stage {
                RouterMetrics::record_pd_body_over_limit(stage);
            }
            return Err("PD response exceeded bounded metadata envelope".into());
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}
async fn prefill(
    client: &reqwest::Client,
    p: &ProgramTarget,
    request: &Value,
    headers: Option<&HeaderMap>,
    params: Option<&Value>,
    max_metadata: usize,
    max_response: usize,
) -> Result<Value, PrefillError> {
    let mut r = request.clone();
    r["max_tokens"] = json!(1);
    r["stream"] = json!(false);
    let o = r
        .as_object_mut()
        .ok_or_else(|| PrefillError::Failed("Invalid PD request".into()))?;
    o.remove("stream_options");
    o.remove("max_completion_tokens");
    // Prevent EOS/stop sampling on the only prefill output token; an
    // immediate stop would produce no KV handoff metadata.
    r["min_tokens"] = json!(1);
    r["kv_transfer_params"] = params
        .cloned()
        .unwrap_or_else(|| json!({"do_remote_prefill":false,"do_remote_decode":true}));
    let response = builder(client, p, "/v1/completions", headers)
        .json(&r)
        .send()
        .await
        .map_err(|_| PrefillError::Failed("PD prefill transport failure".into()))?;
    if !response.status().is_success() {
        return Err(PrefillError::Upstream(response));
    }
    let bytes = read_bounded(response, max_response, Some("prefill"))
        .await
        .map_err(PrefillError::Failed)?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|_| PrefillError::Failed("Invalid PD prefill JSON".into()))?;
    if value.get("error").is_some()
        || !value
            .get("kv_transfer_params")
            .is_some_and(|p| bounded_metadata(p, max_metadata))
    {
        return Err(PrefillError::Failed(
            "Missing verified P transfer metadata".into(),
        ));
    }
    Ok(value)
}

async fn local_prefix(
    client: &reqwest::Client,
    d: &ProgramTarget,
    request: &Value,
    headers: Option<&HeaderMap>,
    engine: &EngineIdentity,
    max: usize,
) -> Option<usize> {
    let mut r = request.clone();
    r["max_tokens"] = json!(1);
    r["stream"] = json!(false);
    r.as_object_mut()?.remove("stream_options");
    r.as_object_mut()?.remove("max_completion_tokens");
    // Prevent EOS/stop sampling on the only prefill output token; an
    // immediate stop would produce no KV handoff metadata.
    r["min_tokens"] = json!(1);
    r["kv_transfer_params"] =
        json!({"pd_probe_only":true,"do_remote_decode":false,"do_remote_prefill":false});
    let response = builder(client, d, "/v1/completions", headers)
        .timeout(Duration::from_secs(2))
        .json(&r)
        .send()
        .await
        .ok()?;
    if !response.status().is_success() {
        return None;
    }
    let bytes = read_bounded(response, max, None).await.ok()?;
    let value: Value = serde_json::from_slice(&bytes).ok()?;
    let params = value.get("kv_transfer_params")?;
    if params.get("pd_probe") != Some(&Value::Bool(true))
        || params.get("pd_instance_epoch").and_then(Value::as_str)
            != Some(engine.instance_epoch.as_str())
    {
        return None;
    }
    params
        .pointer("/pd_feedback/local_cached_tokens")
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
}

/// Fully separate from native PD: unsupported exact-token/capability requests use M1.
#[allow(clippy::too_many_arguments)]
pub(super) async fn execute(
    runtime: &PdProgramRuntime,
    warm_remote: Arc<PdWarmRemoteRuntime>,
    client: &reqwest::Client,
    completion: &ProgramCompletion,
    request: &Value,
    path: &str,
    headers: Option<&HeaderMap>,
    prefills: &[ProgramTarget],
    targets: &[ProgramTarget],
    policy_prefill: usize,
    pd_load: &Arc<super::pd_types::PdLoadManager>,
    prefill_worker: Arc<dyn Worker>,
    decode_worker: Arc<dyn Worker>,
) -> Option<Result<(Response, Option<PdResponseContext>), String>> {
    // Discovery lacks a registration incarnation fence for this protocol.
    if runtime.discovery {
        warm_remote.count("unverified_discovery_fallback");
        return None;
    }
    let topology_revision = runtime.worker_registry.revision();
    let prompt = match exact_request(request, path, warm_remote.config.max_prefix_tokens) {
        Some(prompt) => prompt,
        None => {
            // Text prompts, n>1, missing return_token_ids, or an over-budget
            // context: all fall back to M1, but operators need to see why.
            warm_remote.count("request_shape_unsupported");
            return None;
        }
    };
    let mut dispatch = completion.dispatch().clone();
    let d = targets.iter().find(|t| t.id == dispatch.target_id)?;
    let p = prefills.get(policy_prefill)?;
    // Capability failure is a conservative M1 fallback, never a warm/local guess.
    let (de, pe) = match tokio::join!(
        warm_remote.identify(client, d, headers, topology_revision),
        warm_remote.identify(client, p, headers, topology_revision)
    ) {
        (Ok(d), Ok(p)) if d.profile == p.profile => (d, p),
        _ => {
            warm_remote.count("capability_cold_fallback");
            warm_remote.invalidate_target(&d.base_url, d.dp_rank);
            return None;
        }
    };
    let identity = request_identity(request);
    let mut candidate = warm_remote.claim(&dispatch, &prompt, identity);
    if let Some(c) = candidate.as_ref() {
        // A candidate must be validated against a *fresh* source identity:
        // a restart at the same URL keeps the URL but drops the retained KV.
        let source = identify_uncached(client, &c.source, headers, &warm_remote.config).await;
        if c.topology_revision != topology_revision
            || source.as_ref().ok() != Some(&c.engine)
            || c.engine.profile != de.profile
            || !c.engine.bidirectional
        {
            candidate = None;
            warm_remote.count("source_epoch_rejected");
        }
    }
    let coverage = candidate.as_ref().map_or(0, |c| c.prefix.len());
    let local_coverage = if warm_remote.config.local_append
        && de.local_miss_guard
        && de.local_prefix_probe
        && candidate
            .as_ref()
            .is_some_and(|c| c.source == *d && c.engine == de)
    {
        local_prefix(
            client,
            d,
            request,
            headers,
            &de,
            warm_remote.config.max_metadata_bytes,
        )
        .await
        .filter(|n| *n <= prompt.len())
    } else {
        None
    };
    let running = runtime.scheduler.pd_rank_load(&d.id);
    // P targets have no rank observation: read the per-P metrics scrape.
    // Native PD metrics label engines without the http:// scheme.
    let p_metric = p.base_url.strip_prefix("http://").unwrap_or(&p.base_url);
    let d_metric = d.base_url.strip_prefix("http://").unwrap_or(&d.base_url);
    let p_kv_usage = runtime.scheduler.pd_prefill_kv_usage(&p.base_url);
    let p_running = runtime.scheduler.pd_prefill_load(&p.base_url);
    let warm_pressure_ok = p_kv_usage
        .is_none_or(|usage| usage >= warm_remote.config.warm_min_p_kv_usage)
        && p_running.is_none_or(|running| running <= warm_remote.config.warm_max_p_running);
    let mut route = if warm_remote.config.warm_remote
        && pe.bidirectional
        && pe.fail_on_load_error
        && candidate.is_some()
        && warm_pressure_ok
    {
        PdPath::WarmRemote
    } else {
        if warm_remote.config.warm_remote && candidate.is_some() && !warm_pressure_ok {
            warm_remote.count("warm_skipped_pressure");
        }
        PdPath::ColdRemote
    };
    let mut predicted = None;
    let mut local_permit = None;
    let mut local_running_limit = 0;
    if let (Some(table), Some(running)) = (&warm_remote.config.calibration, running) {
        if table.profile == de.profile {
            predicted = table.estimate(
                route,
                prompt.len(),
                prompt.len().saturating_sub(if route == PdPath::WarmRemote {
                    coverage
                } else {
                    0
                }),
                running,
                warm_remote.correction(route, &d.id),
            );
            // Warm availability alone does not establish it as the cheapest route.
            if route == PdPath::WarmRemote {
                let cold = table.estimate(
                    PdPath::ColdRemote,
                    prompt.len(),
                    prompt.len(),
                    running,
                    warm_remote.correction(PdPath::ColdRemote, &d.id),
                );
                if let (Some(warm), Some(cold)) = (predicted, cold) {
                    if warm + table.min_route_gain_seconds >= cold {
                        route = PdPath::ColdRemote;
                        predicted = Some(cold);
                        warm_remote.count("calibrated_cold_preferred");
                    }
                }
            }
            if warm_remote.config.local_append
                && de.local_miss_guard
                && candidate
                    .as_ref()
                    .is_some_and(|c| c.source == *d && c.engine == de)
            {
                let miss = local_coverage.map(|h| prompt.len().saturating_sub(h));
                if let Some(miss) =
                    miss.filter(|m| *m <= warm_remote.config.local_execution_budget_tokens)
                {
                    let local = table.estimate(
                        PdPath::LocalAppend,
                        prompt.len(),
                        miss,
                        running,
                        warm_remote.correction(PdPath::LocalAppend, &d.id),
                    );
                    if let (Some(remote), Some(local)) = (predicted, local) {
                        if local + table.min_route_gain_seconds < remote {
                            local_permit = warm_remote.local_permit(&d.id);
                            if local_permit.is_some() {
                                route = PdPath::LocalAppend;
                                predicted = Some(local);
                                local_running_limit = table
                                    .cells
                                    .iter()
                                    .find(|c| {
                                        c.path == PdPath::LocalAppend
                                            && (c.context_min..=c.context_max)
                                                .contains(&prompt.len())
                                            && (c.miss_min..=c.miss_max).contains(&miss)
                                            && (c.running_min..=c.running_max).contains(&running)
                                    })
                                    .map_or(0, |c| c.running_max);
                            }
                        }
                    }
                }
            }
        }
    }
    let event_revision = warm_remote.revision_for(d);
    let started = Instant::now();
    // Charge the same native P/D accounting M1 uses so load-aware policies and
    // the published PD occupancy metrics see Program traffic too.
    // admit_bound charges PdLoadManager and attaches the workers itself, so
    // do not attach again: finish/Drop decrement exactly once.
    let attempt = pd_load.admit_bound(prefill_worker, decode_worker);
    let result = async {
        let d_params = if route == PdPath::LocalAppend {
            attempt.finish_prefill();
            json!({"do_remote_decode":false,"do_remote_prefill":false,"pd_local_max_miss_tokens":warm_remote.config.local_execution_budget_tokens,"pd_local_max_running_requests":local_running_limit,"pd_local_max_local_prefills":warm_remote.config.max_local_requests_per_target})
        } else {
            let permit = runtime
                .reserve_prefill(&p.id)
                .await
                .map_err(|_| "P capacity timeout")?;
            if runtime.worker_registry.revision() != topology_revision
                || warm_remote
                    .identify(client, p, headers, topology_revision)
                    .await?
                    != pe
            {
                warm_remote.invalidate_target(&p.base_url, p.dp_rank);
                return Err("P epoch changed after capacity wait".into());
            }
            let warm = candidate.as_ref().filter(|c| {
                route == PdPath::WarmRemote
                    && c.expires
                        > Instant::now()
                            + Duration::from_secs_f64(
                                warm_remote
                                    .config
                                    .lease_margin_seconds
                                    .max(c.engine.read_margin_millis as f64 / 1000.0),
                            )
            });
            if route == PdPath::WarmRemote && warm.is_none() {
                route = PdPath::ColdRemote;
                predicted = None;
                warm_remote.count("expired_before_read");
            }
            RouterMetrics::record_pd_prefill_request(p_metric);
            let mut value = prefill(
                client,
                p,
                request,
                headers,
                warm.map(|c| &c.params),
                warm_remote.config.max_metadata_bytes,
                warm_remote.config.max_response_bytes,
            )
            .await;
            // Only a warm read failure is retried cold. Ordinary upstream
            // rejections are forwarded so the client keeps the status/body.
            let retry_cold = warm.is_some()
                && match &value {
                    Err(PrefillError::Upstream(response)) => response.status().is_server_error(),
                    Err(PrefillError::Failed(_)) => true,
                    Ok(_) => false,
                };
            if retry_cold {
                dispatch = completion
                    .restart_pd_attempt()
                    .ok_or("Cold retry resource recheck failed")?;
                warm_remote.count("warm_read_cold_retry");
                warm_remote.read_outcome_in_profile(false, &de.profile);
                route = PdPath::ColdRemote;
                predicted = None;
                // New transport ID, one bounded retry; no source metadata is reinserted.
                value = prefill(
                    client,
                    p,
                    request,
                    headers,
                    None,
                    warm_remote.config.max_metadata_bytes,
                    warm_remote.config.max_response_bytes,
                )
                .await;
            }
            drop(permit);
            attempt.finish_prefill();
            let value = match value {
                Ok(value) => value,
                Err(PrefillError::Upstream(response)) => {
                    RouterMetrics::record_pd_prefill_error(&p.base_url);
                    return Ok(TurnResult::Forward(forward_backend_response(response)));
                }
                Err(PrefillError::Failed(error)) => {
                    RouterMetrics::record_pd_prefill_error(&p.base_url);
                    return Err(error);
                }
            };
            if value
                .pointer("/kv_transfer_params/pd_instance_epoch")
                .and_then(Value::as_str)
                != Some(pe.instance_epoch.as_str())
            {
                warm_remote.invalidate_target(&p.base_url, p.dp_rank);
                return Err("P response instance epoch mismatch".into());
            }
            if let Some(f) = value.pointer("/kv_transfer_params/pd_feedback") {
                let external = f
                    .get("external_cached_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                if route == PdPath::WarmRemote {
                    warm_remote.count(if external > 0 {
                        "warm_external_read"
                    } else {
                        "warm_local_cache_hit"
                    });
                    if external > 0 {
                        warm_remote.read_outcome_in_profile(true, &de.profile);
                    }
                }
            }
            let mut params = value["kv_transfer_params"].clone();
            params["do_remote_prefill"] = json!(true);
            params["do_remote_decode"] = json!(false);
            params
        };
        // Revalidate endpoint epochs after P waiting/compute and before D dispatch.
        if runtime.worker_registry.revision() != topology_revision
            || warm_remote.identify(client, d, headers, topology_revision).await? != de
        {
            warm_remote.invalidate_target(&d.base_url, d.dp_rank);
            return Err("D epoch changed before dispatch".into());
        }
        let mut dr = request.clone();
        dr["kv_transfer_params"] = d_params;
        let d_started = Instant::now();
        RouterMetrics::record_pd_decode_request(d_metric);
        attempt.begin_decode();
        let response = builder(client, d, path, headers)
            .json(&dr)
            .send()
            .await
            .map_err(|_| "PD decode transport failure")?;
        let streaming = request.get("stream") == Some(&Value::Bool(true));
        match gate_response(
            response,
            streaming,
            warm_remote.config.max_metadata_bytes,
            warm_remote.config.max_response_bytes,
        )
        .await?
        {
            GateOutcome::Ready(response) => Ok(TurnResult::Dispatch(response, d_started)),
            GateOutcome::Forward(response) => {
                RouterMetrics::record_pd_decode_error(d_metric);
                Ok(TurnResult::Forward(response))
            }
        }
    }
    .await;
    match result {
        Ok(TurnResult::Dispatch(response, d_started)) => {
            drop(local_permit.take());
            let response = attempt.track_response(response);
            Some(Ok((
                response,
                Some(PdResponseContext {
                    runtime: warm_remote,
                    scheduler: runtime.scheduler.clone(),
                    worker_registry: runtime.worker_registry.clone(),
                    topology_revision,
                    dispatch: dispatch.clone(),
                    source: d.clone(),
                    engine: de,
                    prompt,
                    identity,
                    path: route,
                    predicted,
                    started,
                    d_started,
                    output: Vec::new(),
                    observed_prompt: None,
                    observed_output_count: None,
                    params: None,
                    invalid: false,
                    local_permit,
                    first_output: None,
                    event_revision,
                }),
            )))
        }
        Ok(TurnResult::Forward(response)) => {
            drop(local_permit.take());
            attempt.finish_decode();
            // The upstream error is forwarded unchanged; the body tracker
            // settles the Program turn as failed without publishing candidates.
            Some(Ok((response, None)))
        }
        Err(error) if route == PdPath::LocalAppend => {
            drop(local_permit);
            dispatch = match completion.restart_pd_attempt() {
                Some(d) => d,
                None => return Some(Err("Cold retry resource recheck failed".into())),
            };
            warm_remote.count("local_guard_cold_retry");
            // One remote retry while the same D reservation still covers the exact work.
            // Local rejected before output; use a fresh backend attempt and P permit.
            let permit = match runtime.reserve_prefill(&p.id).await {
                Ok(p) => p,
                Err(_) => return Some(Err(error)),
            };
            if runtime.worker_registry.revision() != topology_revision
                || warm_remote
                    .identify(client, p, headers, topology_revision)
                    .await
                    .as_ref()
                    .ok()
                    != Some(&pe)
            {
                return Some(Err("P epoch changed before local cold retry".into()));
            }
            RouterMetrics::record_pd_prefill_request(p_metric);
            let value = prefill(
                client,
                p,
                request,
                headers,
                None,
                warm_remote.config.max_metadata_bytes,
                warm_remote.config.max_response_bytes,
            )
            .await;
            drop(permit);
            let fallback = async {
                let value = match value {
                    Ok(value) => value,
                    Err(PrefillError::Upstream(response)) => {
                        RouterMetrics::record_pd_prefill_error(&p.base_url);
                        return Ok(TurnResult::Forward(forward_backend_response(response)));
                    }
                    Err(PrefillError::Failed(error)) => {
                        RouterMetrics::record_pd_prefill_error(&p.base_url);
                        return Err(error);
                    }
                };
                if value
                    .pointer("/kv_transfer_params/pd_instance_epoch")
                    .and_then(Value::as_str)
                    != Some(pe.instance_epoch.as_str())
                {
                    warm_remote.invalidate_target(&p.base_url, p.dp_rank);
                    return Err("P retry response instance epoch mismatch".into());
                }
                if runtime.worker_registry.revision() != topology_revision
                    || warm_remote
                        .identify(client, d, headers, topology_revision)
                        .await?
                        != de
                {
                    warm_remote.invalidate_target(&d.base_url, d.dp_rank);
                    return Err("D epoch changed during cold fallback".into());
                }
                let mut dr = request.clone();
                dr["kv_transfer_params"] = value["kv_transfer_params"].clone();
                dr["kv_transfer_params"]["do_remote_decode"] = json!(false);
                dr["kv_transfer_params"]["do_remote_prefill"] = json!(true);
                let d_started = Instant::now();
                RouterMetrics::record_pd_decode_request(d_metric);
                // The first local attempt may have failed before decode began.
                attempt.begin_decode();
                let response = builder(client, d, path, headers)
                    .json(&dr)
                    .send()
                    .await
                    .map_err(|_| "Cold fallback transport failure")?;
                let gate = gate_response(
                    response,
                    request.get("stream") == Some(&Value::Bool(true)),
                    warm_remote.config.max_metadata_bytes,
                    warm_remote.config.max_response_bytes,
                )
                .await?;
                match gate {
                    GateOutcome::Ready(response) => Ok(TurnResult::Dispatch(response, d_started)),
                    GateOutcome::Forward(response) => {
                        RouterMetrics::record_pd_decode_error(d_metric);
                        Ok(TurnResult::Forward(response))
                    }
                }
            }
            .await;
            Some(fallback.map(|turn| match turn {
                TurnResult::Dispatch(response, d_started) => (
                    attempt.track_response(response),
                    Some(PdResponseContext {
                        runtime: warm_remote,
                        scheduler: runtime.scheduler.clone(),
                        worker_registry: runtime.worker_registry.clone(),
                        topology_revision,
                        dispatch: dispatch.clone(),
                        source: d.clone(),
                        engine: de,
                        prompt,
                        identity,
                        path: PdPath::ColdRemote,
                        predicted: None,
                        started,
                        d_started,
                        output: Vec::new(),
                        observed_prompt: None,
                        observed_output_count: None,
                        params: None,
                        invalid: false,
                        local_permit: None,
                        first_output: None,
                        event_revision,
                    }),
                ),
                TurnResult::Forward(response) => {
                    attempt.finish_decode();
                    (response, None)
                }
            }))
        }
        Err(e) => Some(Err(e)),
    }
}

type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;
/// Buffer only until a first output or a terminal pre-output error. Never splice
/// a second generation after exposing output. Original bytes remain unchanged.
async fn gate_response(
    response: reqwest::Response,
    streaming: bool,
    max_metadata: usize,
    max_response: usize,
) -> Result<GateOutcome, String> {
    let status = response.status();
    if !status.is_success() {
        // Keep the upstream status, body and headers (validation errors,
        // overload with Retry-After) instead of collapsing them into a 503.
        return Ok(GateOutcome::Forward(forward_backend_response(response)));
    }
    let headers = header_utils::preserve_response_headers(response.headers());
    let mut stream: ByteStream = Box::pin(response.bytes_stream());
    let mut prefix = Vec::new();
    let mut offset = 0;
    if streaming {
        let mut saw_output = false;
        'read: while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|_| "Pre-output D stream failed")?;
            if prefix.len().saturating_add(chunk.len()) > max_metadata {
                RouterMetrics::record_pd_body_over_limit("decode");
                return Err("PD pre-output envelope too large".into());
            }
            prefix.extend_from_slice(&chunk);
            while let Some(n) = prefix[offset..].iter().position(|b| *b == b'\n') {
                let line = &prefix[offset..offset + n];
                offset += n + 1;
                if let Ok(line) = std::str::from_utf8(line) {
                    if let Some(data) = line.trim().strip_prefix("data:") {
                        if data.trim() == "[DONE]" {
                            break 'read;
                        }
                        if let Ok(v) = serde_json::from_str::<Value>(data) {
                            if v.get("error").is_some() {
                                return Err("PD initial stream error".into());
                            }
                            if v.get("choices").and_then(Value::as_array).is_some_and(|c| {
                                c.iter().any(|c| {
                                    c.get("text")
                                        .and_then(Value::as_str)
                                        .is_some_and(|s| !s.is_empty())
                                        || c.get("token_ids")
                                            .and_then(Value::as_array)
                                            .is_some_and(|t| !t.is_empty())
                                })
                            }) {
                                saw_output = true;
                                break 'read;
                            }
                        }
                    }
                }
            }
        }
        if !saw_output {
            return Err("PD stream ended before generated output".into());
        }
    } else {
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|_| "Pre-output D response failed")?;
            if prefix.len().saturating_add(chunk.len()) > max_response {
                RouterMetrics::record_pd_body_over_limit("decode");
                return Err("PD response envelope too large".into());
            }
            prefix.extend_from_slice(&chunk);
        }
        let value: Value =
            serde_json::from_slice(&prefix).map_err(|_| "Invalid D response JSON")?;
        let output = value
            .get("choices")
            .and_then(Value::as_array)
            .is_some_and(|choices| {
                choices.iter().any(|c| {
                    c.get("token_ids")
                        .and_then(Value::as_array)
                        .is_some_and(|t| !t.is_empty())
                        || c.get("text")
                            .and_then(Value::as_str)
                            .is_some_and(|t| !t.is_empty())
                })
            });
        if value.get("error").is_some() || !output {
            return Err("PD response completed before generated output".into());
        }
    }
    let body = Body::from_stream(
        futures_util::stream::once(async move { Ok::<_, reqwest::Error>(Bytes::from(prefix)) })
            .chain(stream),
    );
    let mut result = Response::new(body);
    *result.status_mut() =
        StatusCode::from_u16(status.as_u16()).map_err(|_| "Invalid backend status")?;
    *result.headers_mut() = headers;
    Ok(GateOutcome::Ready(result))
}

pub(super) struct PdResponseContext {
    runtime: Arc<PdWarmRemoteRuntime>,
    scheduler: Arc<ProgramScheduler>,
    worker_registry: Arc<crate::core::WorkerRegistry>,
    topology_revision: u64,
    dispatch: ProgramDispatch,
    source: ProgramTarget,
    engine: EngineIdentity,
    prompt: Vec<u32>,
    identity: [u8; 32],
    path: PdPath,
    predicted: Option<f64>,
    started: Instant,
    d_started: Instant,
    output: Vec<u32>,
    observed_prompt: Option<Vec<u32>>,
    observed_output_count: Option<usize>,
    params: Option<Value>,
    invalid: bool,
    event_revision: (u64, u64, u64),
    local_permit: Option<OwnedSemaphorePermit>,
    first_output: Option<Instant>,
}
impl PdResponseContext {
    /// Body-buffer bound for the response tracker. It must match the
    /// pre-output gate so a forwarded completion is still inspected and can
    /// publish a candidate.
    pub(super) fn response_buffer_limit(&self) -> usize {
        self.runtime.config.max_response_bytes
    }

    pub fn observe(&mut self, value: &Value) {
        if value.get("error").is_some() {
            self.invalid = true;
        }
        if let Some(usage) = value.get("usage").filter(|v| !v.is_null()) {
            if usage.get("prompt_tokens").and_then(Value::as_u64) != Some(self.prompt.len() as u64)
            {
                self.invalid = true;
            }
            self.observed_output_count = usage
                .get("completion_tokens")
                .and_then(Value::as_u64)
                .and_then(|v| usize::try_from(v).ok());
        }
        if let Some(p) = value.get("kv_transfer_params").filter(|v| !v.is_null()) {
            if bounded_metadata(p, self.runtime.config.max_metadata_bytes) {
                self.params = Some(p.clone());
            } else {
                self.invalid = true;
            }
        }
        if let Some(choices) = value.get("choices").and_then(Value::as_array) {
            if choices.len() > 1 {
                self.invalid = true;
            }
            for c in choices {
                if c.get("index").and_then(Value::as_u64).unwrap_or(0) != 0 {
                    self.invalid = true;
                }
                if let Some(p) = c.get("prompt_token_ids").filter(|v| !v.is_null()) {
                    match tokens(p) {
                        Some(t) if t == self.prompt => self.observed_prompt = Some(t),
                        _ => self.invalid = true,
                    }
                }
                if let Some(t) = c.get("token_ids").filter(|v| !v.is_null()) {
                    match tokens(t) {
                        Some(t)
                            if self.output.len() + t.len() + self.prompt.len()
                                <= self.runtime.config.max_prefix_tokens =>
                        {
                            if !t.is_empty() && self.first_output.is_none() {
                                self.first_output = Some(Instant::now());
                            }
                            self.output.extend(t);
                        }
                        _ => self.invalid = true,
                    }
                }
            }
        }
    }
    pub fn finish(&mut self, success: bool, completion: &ProgramCompletion) {
        self.local_permit.take();
        self.runtime.refresh_loss();
        if !success
            || self.invalid
            || self.worker_registry.revision() != self.topology_revision
            || self.runtime.revision_for(&self.source) != self.event_revision
            || !self.scheduler.dispatch_is_current(&self.dispatch)
        {
            self.runtime.count("incomplete_candidate_discarded");
            return;
        }
        let key = format!("{:?}:{}", self.path, self.source.id);
        let actual = self
            .first_output
            .map(|t| t.saturating_duration_since(self.started).as_secs_f64());
        if self.runtime.config.feedback {
            if let (Some(predicted), Some(actual)) = (self.predicted, actual) {
                let residual = actual - predicted;
                let mut s = self.runtime.state.lock();
                let window = s.residuals.entry(key.clone()).or_default();
                if window.len() == 64 {
                    window.pop_front();
                }
                window.push_back(residual);
                let avg = window.iter().sum::<f64>() / window.len() as f64;
                let f = s.feedback.entry(key.clone()).or_default();
                f.samples += 1;
                f.mean_residual = avg;
                f.max_absolute_error = f.max_absolute_error.max(residual.abs());
                if self
                    .runtime
                    .config
                    .calibration
                    .as_ref()
                    .is_some_and(|c| residual.abs() > c.max_prediction_error_seconds)
                {
                    f.errors += 1;
                }
            }
        }
        let Some(params) = self.params.as_ref() else {
            self.runtime.count("missing_candidate_metadata");
            return;
        };
        if let Some(f) = params.get("pd_feedback") {
            let mut s = self.runtime.state.lock();
            let entry = s.feedback.entry(key).or_default();
            entry.computed_tokens += f
                .get("computed_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            entry.local_cached_tokens += f
                .get("local_cached_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            entry.external_cached_tokens += f
                .get("external_cached_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
        }
        let remaining = params
            .get("remote_blocks_ttl_remaining_seconds")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let coverage = params
            .get("remote_num_tokens")
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(0);
        if self.observed_output_count.is_none() {
            // Streaming clients must request the final usage chunk
            // (stream_options.include_usage); without it the completed turn
            // cannot be fenced against its generated output count.
            self.runtime.count("missing_usage");
            return;
        }
        if self.observed_prompt.as_ref() != Some(&self.prompt)
            || self.observed_output_count != Some(self.output.len())
            || params.get("pd_protocol_revision").and_then(Value::as_str)
                != Some(self.engine.protocol_revision.as_str())
            || params.get("pd_instance_epoch").and_then(Value::as_str)
                != Some(self.engine.instance_epoch.as_str())
            || params.get("do_remote_decode") != Some(&Value::Bool(true))
            || !remaining.is_finite()
            || remaining <= 0.0
            || remaining > 86400.0
            || coverage == 0
            || coverage > self.prompt.len() + self.output.len()
        {
            self.runtime.count("invalid_prefix_or_expiry");
            return;
        }
        let expires = self.d_started + Duration::from_secs_f64(remaining);
        let budget = expires
            .saturating_duration_since(Instant::now())
            .as_secs_f64()
            - self
                .runtime
                .config
                .lease_margin_seconds
                .max(self.engine.read_margin_millis as f64 / 1000.0);
        if budget <= 0.0 {
            self.runtime.count("expired_candidate");
            return;
        }
        // Reuse only the exact conversation prefix. A successor prompt is not
        // guaranteed to contain this turn's generated tokens (trace replay
        // replays historical assistant turns), but it always starts with this
        // turn's prompt tokens, whose KV the D source still holds.
        let coverage = coverage.min(self.prompt.len());
        let prefix = self.prompt[..coverage].to_vec();
        let mut params = params.clone();
        params["remote_num_tokens"] = json!(coverage as u64);
        // Same future input and same rank snapshot for keep/drop. Independent P
        // cache is never assumed absent unless the calibration profile states it.
        if self.runtime.config.automatic_ttl {
            let mut chosen = 0.0_f64;
            if let (Some(table), Some(running)) = (
                &self.runtime.config.calibration,
                self.scheduler.pd_rank_load(&self.source.id),
            ) {
                if table.profile == self.engine.profile {
                    let (intervals, growth, max) = self.scheduler.pd_ttl_factors(&self.source.id);
                    let context = prefix.len().saturating_add(growth);
                    let keep = table.estimate(
                        PdPath::WarmRemote,
                        context,
                        growth,
                        running,
                        self.runtime.correction(PdPath::WarmRemote, &self.source.id),
                    );
                    let cold = table.estimate(
                        PdPath::ColdRemote,
                        context,
                        context,
                        running,
                        self.runtime.correction(PdPath::ColdRemote, &self.source.id),
                    );
                    // NIXL's physical source lease is independent of logical seating.
                    // Dropping cannot be priced as destroying this verified source.
                    let drop = match (keep, cold) {
                        (Some(warm), Some(cold)) => {
                            Some(warm.min(cold) + table.drop_re_admission_seconds)
                        }
                        _ => None,
                    };
                    if let (Some(keep), Some(drop)) = (keep, drop) {
                        let expected_drop = if self.runtime.config.feedback {
                            keep + (drop - keep) * self.runtime.residency_probability()
                        } else {
                            drop
                        };
                        chosen = solve_pd_ttl(
                            &intervals,
                            keep,
                            expected_drop,
                            self.runtime.config.ttl_gamma,
                            max,
                            budget,
                            self.runtime.config.ttl_min_samples,
                            self.runtime.config.ttl_min_gain_seconds,
                        )
                        .seconds;
                    }
                }
            }
            if chosen <= 0.0 && self.runtime.config.ttl_online {
                // Calibration-free fallback: fit the seat window from this rank's
                // observed inter-turn intervals and the estimated cost of
                // recomputing the served prefix. The verified source lease still
                // bounds the result.
                if let Some(online) = self.scheduler.pd_fitted_ttl(&self.source.id, prefix.len()) {
                    chosen = online.as_secs_f64();
                    self.runtime.count("automatic_ttl_online");
                }
            }
            let usable = chosen.min(budget.max(0.0));
            if usable > 0.0 {
                completion.set_pd_ttl(bounded_pd_duration(usable));
            }
            self.runtime.count(if usable > 0.0 {
                "automatic_ttl_positive"
            } else {
                "automatic_ttl_zero"
            });
        }
        // Explicit TTL is still bounded by source availability when it relies on it.
        if self.scheduler.pd_explicit_ttl_enabled() {
            if let Some(hint) = self.dispatch.request_hints().kv_retention_ttl {
                completion.set_pd_ttl(hint.min(bounded_pd_duration(budget)));
            }
        }
        if prefix.len() > self.runtime.config.max_candidate_tokens {
            // A single candidate larger than the whole retention budget can never fit.
            self.runtime.count("candidate_over_budget");
            return;
        }
        let mut s = self.runtime.state.lock();
        s.candidates.retain(|_, c| c.expires > Instant::now());
        // Bound retained prompt-prefix memory, not only the candidate count.
        let token_budget = self.runtime.config.max_candidate_tokens;
        let mut retained: usize = s.candidates.values().map(|c| c.prefix.len()).sum();
        while s.candidates.len() >= self.runtime.config.max_candidates
            || retained.saturating_add(prefix.len()) > token_budget
        {
            let Some(key) = s
                .candidates
                .iter()
                .min_by_key(|(_, c)| c.expires)
                .map(|(k, _)| k.clone())
            else {
                break;
            };
            if let Some(removed) = s.candidates.remove(&key) {
                retained = retained.saturating_sub(removed.prefix.len());
            }
        }
        s.candidates.insert(
            self.dispatch.program().clone(),
            Candidate {
                topology_revision: self.topology_revision,
                produced_attempt: self.dispatch.attempt_id(),
                produced_placement: self.dispatch.placement_epoch(),
                source: self.source.clone(),
                engine: self.engine.clone(),
                params: params.clone(),
                prefix,
                identity: self.identity,
                expires,
            },
        );
        *s.counters.entry("candidate_published").or_default() += 1;
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::program_scheduling::ProgramRequestHints;
    fn dispatch(attempt: u64, epoch: u64, generation: u64) -> ProgramDispatch {
        ProgramDispatch::new(
            attempt,
            ProgramRef::new("tenant/model".into(), "task".into(), generation),
            "d".into(),
            epoch,
            Instant::now(),
            Instant::now(),
            3,
            false,
            None,
            ProgramRequestHints::default(),
        )
    }
    fn identity(instance_epoch: &str) -> EngineIdentity {
        EngineIdentity {
            protocol_revision: "vllm-0.28.0-pd-feedback-v1".into(),
            instance_epoch: instance_epoch.into(),
            profile: "profile".into(),
            dp_rank: 0,
            bidirectional: true,
            fail_on_load_error: true,
            read_margin_millis: 5000,
            local_miss_guard: true,
            local_prefix_probe: true,
            store: false,
        }
    }
    #[test]
    fn invalidate_target_without_rank_clears_all_cached_identities_for_base() {
        let runtime = PdWarmRemoteRuntime::new(PdProgramSchedulingConfig::default());
        {
            let mut state = runtime.state.lock();
            for rank in [Some(0), Some(1)] {
                state.identities.insert(
                    ("http://d".to_string(), rank),
                    (identity("instance"), Instant::now(), 1),
                );
            }
            state.identities.insert(
                ("http://other".to_string(), Some(0)),
                (identity("instance"), Instant::now(), 1),
            );
        }
        runtime.invalidate_target("http://d", None);
        let state = runtime.state.lock();
        assert!(
            !state.identities.keys().any(|(base, _)| base == "http://d"),
            "rank-less invalidation kept stale identities for the base"
        );
        assert!(state
            .identities
            .keys()
            .any(|(base, _)| base == "http://other"));
    }
    fn runtime_with_candidate() -> (PdWarmRemoteRuntime, ProgramDispatch) {
        let runtime = PdWarmRemoteRuntime::new(PdProgramSchedulingConfig::default());
        let next = dispatch(2, 1, 0);
        runtime.state.lock().candidates.insert(
            next.program().clone(),
            Candidate {
                topology_revision: 0,
                produced_attempt: 1,
                produced_placement: 1,
                source: ProgramTarget {
                    id: "d".into(),
                    base_url: "http://d".into(),
                    dp_rank: Some(0),
                },
                engine: EngineIdentity {
                    protocol_revision: "vllm-0.28.0-pd-feedback-v1".into(),
                    instance_epoch: "instance".into(),
                    profile: "profile".into(),
                    dp_rank: 0,
                    bidirectional: true,
                    fail_on_load_error: true,
                    read_margin_millis: 5000,
                    local_miss_guard: true,
                    local_prefix_probe: true,
                    store: false,
                },
                params: json!({}),
                prefix: vec![1, 2, 3],
                identity: [7; 32],
                expires: Instant::now() + Duration::from_secs(30),
            },
        );
        (runtime, next)
    }
    #[test]
    fn lease_is_single_use_and_prefix_must_be_exact() {
        let (r, d) = runtime_with_candidate();
        assert!(r.claim(&d, &[1, 2, 3, 4], [7; 32]).is_some());
        assert!(r.claim(&d, &[1, 2, 3, 4], [7; 32]).is_none());
        let (r, d) = runtime_with_candidate();
        assert!(r.claim(&d, &[1, 2, 8], [7; 32]).is_none());
        assert!(r.state.lock().candidates.is_empty());
    }
    #[test]
    fn generation_template_and_old_attempt_rejected() {
        let (r, _) = runtime_with_candidate();
        assert!(r.claim(&dispatch(2, 1, 1), &[1, 2, 3], [7; 32]).is_none());
        let (r, d) = runtime_with_candidate();
        assert!(r.claim(&d, &[1, 2, 3], [8; 32]).is_none());
        let (r, _) = runtime_with_candidate();
        assert!(r.claim(&dispatch(1, 1, 0), &[1, 2, 3], [7; 32]).is_none());
    }
    #[test]
    fn expiry_and_rank_invalidation_fail_closed() {
        let (r, d) = runtime_with_candidate();
        r.state
            .lock()
            .candidates
            .get_mut(d.program())
            .unwrap()
            .expires = Instant::now();
        assert!(r.claim(&d, &[1, 2, 3], [7; 32]).is_none());
        let (r, d) = runtime_with_candidate();
        r.invalidate_target("http://d", Some(1));
        assert!(r.claim(&d, &[1, 2, 3], [7; 32]).is_some());
        let (r, d) = runtime_with_candidate();
        r.invalidate_target("http://d", Some(0));
        assert!(r.claim(&d, &[1, 2, 3], [7; 32]).is_none());
        let (r, d) = runtime_with_candidate();
        r.invalidate_all();
        assert!(r.claim(&d, &[1, 2, 3], [7; 32]).is_none());
    }
    #[test]
    fn event_fences_scope_to_source_and_rank_until_global_loss() {
        let (r, d) = runtime_with_candidate();
        let target = r
            .state
            .lock()
            .candidates
            .get(d.program())
            .unwrap()
            .source
            .clone();
        let before = r.revision_for(&target);
        r.invalidate_target("http://other-d", None);
        r.invalidate_target(&target.base_url, Some(1));
        assert_eq!(r.revision_for(&target), before);
        r.invalidate_target(&target.base_url, Some(0));
        assert_ne!(r.revision_for(&target), before);
        let before = r.revision_for(&target);
        r.invalidate_all();
        assert_ne!(r.revision_for(&target), before);
    }
    #[test]
    fn no_hot_route_from_text_or_client_metadata() {
        assert!(exact_request(
            &json!({"prompt":"text","return_token_ids":true}),
            "/v1/completions",
            100
        )
        .is_none());
        assert!(exact_request(
            &json!({"prompt":[1,2],"return_token_ids":true,"n":2}),
            "/v1/completions",
            100
        )
        .is_none());
        assert!(exact_request(
            &json!({"prompt":[1,2],"return_token_ids":true,"stream":true}),
            "/v1/completions",
            100
        )
        .is_none());
        assert_eq!(
            exact_request(
                &json!({"prompt":[1,2],"return_token_ids":true}),
                "/v1/completions",
                100
            ),
            Some(vec![1, 2])
        );
    }
    #[test]
    fn residency_requires_observations_and_includes_failures() {
        let r = PdWarmRemoteRuntime::new(PdProgramSchedulingConfig::default());
        assert_eq!(r.residency_probability(), 0.0);
        for _ in 0..32 {
            r.read_outcome(true);
        }
        let good = r.residency_probability();
        assert!(good > 0.8 && good < 1.0);
        for _ in 0..32 {
            r.read_outcome(false);
        }
        assert!(r.residency_probability() < good);
    }
}
