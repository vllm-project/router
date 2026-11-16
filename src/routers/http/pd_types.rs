use crate::config::PdHeldKvMode;
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::policies::{AffinityKey, AffinityRecord, RequestHeaders};
use axum::body::Body;
use futures_util::StreamExt;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

// Custom error type for PD router operations
#[derive(Debug, thiserror::Error)]
pub enum PDRouterError {
    #[error("Worker already exists: {url}")]
    WorkerAlreadyExists { url: String },

    #[error("Worker not found: {url}")]
    WorkerNotFound { url: String },

    #[error("Lock acquisition failed: {operation}")]
    LockError { operation: String },

    #[error("Health check failed for worker: {url}")]
    HealthCheckFailed { url: String },

    #[error("Invalid worker configuration: {reason}")]
    InvalidConfiguration { reason: String },

    #[error("Network error: {message}")]
    NetworkError { message: String },

    #[error("Timeout waiting for worker: {url}")]
    Timeout { url: String },
}

/// Format a full error chain for debugging (walks source() recursively).
/// Produces output like: "outer error caused by: middle error caused by: root cause"
pub fn error_chain(err: &dyn std::error::Error) -> String {
    let mut chain = vec![err.to_string()];
    let mut source = err.source();
    while let Some(s) = source {
        chain.push(s.to_string());
        source = s.source();
    }
    chain.join(" caused by: ")
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PdEndpoint {
    pub http_url: String,
    pub dp_rank: Option<usize>,
}

impl PdEndpoint {
    pub(super) fn for_worker(worker: &dyn Worker) -> Self {
        Self {
            http_url: worker.base_url().to_owned(),
            dp_rank: worker.dp_rank(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Stage {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PdLoadSample {
    pub prefill_requests: usize,
    pub active_prefill_tokens: u64,
    pub unknown_prefill_requests: usize,
    pub pending_decode_requests: usize,
    pub active_decode_requests: usize,
}

impl PdLoadSample {
    fn publish(self, stage: Stage, endpoint: &PdEndpoint) {
        let rank = endpoint
            .dp_rank
            .map_or_else(|| "aggregate".to_owned(), |rank| rank.to_string());
        let record = |name: &'static str, value: f64| {
            metrics::gauge!(name, "worker" => endpoint.http_url.clone(), "rank" => rank.clone())
                .set(value);
        };
        match stage {
            Stage::Prefill => {
                record(
                    "vllm_router_pd_prefill_occupancy",
                    self.prefill_requests as f64,
                );
                record(
                    "vllm_router_pd_active_prefill_tokens",
                    self.active_prefill_tokens as f64,
                );
                record(
                    "vllm_router_pd_unknown_prefill_requests",
                    self.unknown_prefill_requests as f64,
                );
            }
            Stage::Decode => {
                record(
                    "vllm_router_pd_pending_decode_requests",
                    self.pending_decode_requests as f64,
                );
                record(
                    "vllm_router_pd_active_decode_requests",
                    self.active_decode_requests as f64,
                );
            }
        }
    }

    pub(super) fn occupancy(self, stage: Stage) -> usize {
        match stage {
            Stage::Prefill => self.prefill_requests,
            Stage::Decode => self.pending_decode_requests + self.active_decode_requests,
        }
    }

    fn idle(self) -> bool {
        self.prefill_requests == 0
            && self.active_prefill_tokens == 0
            && self.unknown_prefill_requests == 0
            && self.pending_decode_requests == 0
            && self.active_decode_requests == 0
    }
}

pub(super) type Loads = HashMap<(Stage, PdEndpoint), PdLoadSample>;

/// Backend-effective tokens prepared outside the admission transaction.
#[derive(Debug, Clone)]
pub struct PdPrefillInput {
    pub prompts: Option<Arc<[Vec<u32>]>>,
    /// Per-rank placement snapshot.
    /// Hashing and index queries are completed before the admission lock.
    pub work_by_rank: HashMap<Option<usize>, crate::policies::PrefillWorkEstimate>,
    pub allow_credit: bool,
    pub fallback_reason: Option<&'static str>,
}

/// Token inputs are keyed by the P base URL: each backend renders its own
/// effective input, so different tokenizer/template configurations never mix.
pub struct PdRequestContext<'a> {
    pub prefill_inputs: Option<&'a HashMap<String, PdPrefillInput>>,
    pub model_id: Option<&'a str>,
    pub legacy_policy_input: Option<&'a str>,
    pub(crate) prefill_cache: Option<&'a crate::policies::PreparedCacheSelection>,
    pub(crate) decode_cache: Option<&'a crate::policies::PreparedCacheSelection>,
    pub headers: Option<&'a RequestHeaders>,
    /// Stable session identity for the optional kv_aware decode binding.
    /// Resolved at request entry, only when the affinity switch is on;
    /// `None` selects per request on both stages.
    pub affinity_key: Option<AffinityKey>,
    /// Borrowed LoRA name from the request; used for Decode binding only with an eligible affinity key.
    pub lora_name: Option<&'a str>,
    /// Request cache salt scoping the binding digest.
    pub cache_salt: Option<&'a str>,
    pub route: &'a str,
}

#[derive(Debug, Clone)]
pub struct StageDecision {
    pub fallback_reason: Option<&'static str>,
    pub credited_prefix_tokens: Option<u64>,
    pub worker_index: usize,
    pub endpoint: PdEndpoint,
    pub score: Option<f64>,
    /// Immutable token work charged to this attempt after KV credit.
    pub new_prefill_tokens: Option<u64>,
    /// How session affinity shaped this stage's choice: bind / sticky_hit /
    /// no_session / capacity_bypass, set only while the switch is on.
    pub affinity_outcome: Option<&'static str>,
    /// Held binding reference for this stage's decision; the admission
    /// attempt releases it on cleanup.
    pub affinity: Option<Arc<AffinityRecord>>,
}

#[derive(Debug, Clone)]
pub struct PdSelection {
    pub prefill: StageDecision,
    pub decode: StageDecision,
}

#[derive(Debug, Default)]
pub struct PdLoadManager {
    loads: Arc<Mutex<Loads>>,
}

impl PdLoadManager {
    #[cfg(test)]
    pub(super) fn snapshot(&self) -> Loads {
        self.loads.lock().clone()
    }

    /// The router supplies selection; this manager owns only admission and
    /// lifecycle state. No policy, worker registry, or connector lives here.
    ///
    /// Both stages are charged here. Callers that dispatch Decode only after
    /// Prefill completes use [`Self::admit_deferred`] instead.
    pub(super) fn admit(
        &self,
        select: impl FnOnce(&Loads) -> Option<PdSelection>,
    ) -> Option<(PdSelection, PdAttempt)> {
        self.admit_inner(select, false)
    }

    /// Admit the Prefill stage only and leave the Decode commitment open.
    ///
    /// The supplied selection still carries a Decode decision, but it is
    /// provisional: it takes no affinity binding and nothing is charged for it
    /// until [`Self::commit_decode`] runs. A provisional pick must therefore
    /// never be dispatched.
    pub(super) fn admit_deferred(
        &self,
        select: impl FnOnce(&Loads) -> Option<PdSelection>,
    ) -> Option<(PdSelection, PdAttempt)> {
        self.admit_inner(select, true)
    }

    fn admit_inner(
        &self,
        select: impl FnOnce(&Loads) -> Option<PdSelection>,
        defer_decode: bool,
    ) -> Option<(PdSelection, PdAttempt)> {
        let mut loads = self.loads.lock();
        let selection = select(&loads)?;
        let prefill = &selection.prefill;
        let decode = &selection.decode;
        // Neither stage is charged until both selections have succeeded.
        let p = loads
            .entry((Stage::Prefill, prefill.endpoint.clone()))
            .or_default();
        p.prefill_requests += 1;
        if let Some(tokens) = prefill.new_prefill_tokens {
            p.active_prefill_tokens += tokens;
        } else {
            p.unknown_prefill_requests += 1;
        }
        p.publish(Stage::Prefill, &prefill.endpoint);
        let now = std::time::Instant::now();
        let (decode_endpoint, affinity, decode_state, pending_since) = if defer_decode {
            (None, None, DecodeState::Unbound, None)
        } else {
            let d = loads
                .entry((Stage::Decode, decode.endpoint.clone()))
                .or_default();
            d.pending_decode_requests += 1;
            d.publish(Stage::Decode, &decode.endpoint);
            (
                Some(decode.endpoint.clone()),
                decode.affinity.clone(),
                DecodeState::Pending,
                Some(now),
            )
        };
        let attempt = PdAttempt(Arc::new(AttemptInner {
            admitted_at: now,
            loads: self.loads.clone(),
            prefill: selection.prefill.endpoint.clone(),
            prefill_worker: Mutex::new(None),
            decode_worker: Mutex::new(None),
            prefill_tokens: selection.prefill.new_prefill_tokens,
            state: Mutex::new(AttemptState {
                prefill: true,
                decode: decode_state,
                decode_endpoint,
                affinity,
                pending_since,
                held_kv: HeldKvState::default(),
            }),
        }));
        Some((selection, attempt))
    }

    /// Charge and bind the Decode stage chosen at dispatch time.
    ///
    /// The attempt state lock serializes this against the release paths: a
    /// concurrent drop either finishes first, so this finds a non-`Unbound`
    /// state and charges nothing, or it waits and then releases what was
    /// charged. Returns `None` when the attempt already left `Unbound` or the
    /// selector refused every candidate.
    pub(super) fn commit_decode(
        &self,
        attempt: &PdAttempt,
        select: impl FnOnce(&Loads) -> Option<StageDecision>,
    ) -> Option<StageDecision> {
        let inner = &attempt.0;
        let mut state = inner.state.lock();
        if state.decode != DecodeState::Unbound {
            return None;
        }
        let mut loads = self.loads.lock();
        let decision = select(&loads)?;
        let d = loads
            .entry((Stage::Decode, decision.endpoint.clone()))
            .or_default();
        d.pending_decode_requests += 1;
        d.publish(Stage::Decode, &decision.endpoint);
        state.decode_endpoint = Some(decision.endpoint.clone());
        state.affinity = decision.affinity.clone();
        state.pending_since = Some(std::time::Instant::now());
        state.decode = DecodeState::Pending;
        metrics::histogram!("vllm_router_pd_decode_selection_gap_seconds")
            .record(inner.admitted_at.elapsed().as_secs_f64());
        Some(decision)
    }

    /// Charge a P/D pair the caller already bound.
    ///
    /// Program scheduling picks the decode engine by affinity before dispatch,
    /// so it cannot go through policy selection, but its occupancy must stay
    /// visible to native admission and to `/metrics`. The prefill token count is
    /// unknown here, so the attempt is charged as an unknown-size prefill.
    pub(super) fn admit_bound(
        &self,
        prefill_worker: Arc<dyn Worker>,
        decode_worker: Arc<dyn Worker>,
    ) -> PdAttempt {
        let prefill = PdEndpoint::for_worker(prefill_worker.as_ref());
        let decode = PdEndpoint::for_worker(decode_worker.as_ref());
        {
            let mut loads = self.loads.lock();
            let p = loads.entry((Stage::Prefill, prefill.clone())).or_default();
            p.prefill_requests += 1;
            p.unknown_prefill_requests += 1;
            p.publish(Stage::Prefill, &prefill);
            let d = loads.entry((Stage::Decode, decode.clone())).or_default();
            d.pending_decode_requests += 1;
            d.publish(Stage::Decode, &decode);
        }
        let attempt = PdAttempt(Arc::new(AttemptInner {
            admitted_at: std::time::Instant::now(),
            loads: self.loads.clone(),
            prefill,
            prefill_worker: Mutex::new(None),
            decode_worker: Mutex::new(None),
            prefill_tokens: None,
            state: Mutex::new(AttemptState {
                prefill: true,
                decode: DecodeState::Pending,
                decode_endpoint: Some(decode),
                affinity: None,
                pending_since: Some(std::time::Instant::now()),
                held_kv: HeldKvState::default(),
            }),
        }));
        attempt.attach_workers(prefill_worker, decode_worker);
        attempt
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecodeState {
    /// Admitted without a decode commitment; nothing charged.
    Unbound,
    Pending,
    Active,
    Finished,
}

#[derive(Debug)]
struct AttemptState {
    prefill: bool,
    decode: DecodeState,
    /// Committed decode endpoint; `None` while the commitment is deferred.
    decode_endpoint: Option<PdEndpoint>,
    /// Decode binding held alive until this attempt finishes.
    affinity: Option<Arc<AffinityRecord>>,
    /// When the decode slot was charged, for the pending histogram.
    pending_since: Option<std::time::Instant>,
    held_kv: HeldKvState,
}

/// Independent of load accounting: releasing a charge says nothing about engine blocks.
#[derive(Debug, Default)]
struct HeldKvState {
    mode: Option<PdHeldKvMode>,
    decode_outcome: Option<bool>,
    recorded: bool,
    external_decode_outcome: bool,
    producer_succeeded: bool,
    body_eof: bool,
}

impl HeldKvState {
    fn record_abandonment(&mut self) {
        if self.recorded || self.decode_outcome != Some(false) {
            return;
        }
        let Some(mode) = self.mode else { return };
        let result = if mode == PdHeldKvMode::Transferred {
            "not_held"
        } else {
            "unsupported"
        };
        self.recorded = true;
        RouterMetrics::record_pd_held_kv(result);
        tracing::debug!(
            ?mode,
            result,
            "Completed Prefill abandoned; engine reclamation unavailable"
        );
    }
}

struct AttemptInner {
    admitted_at: std::time::Instant,
    loads: Arc<Mutex<Loads>>,
    prefill: PdEndpoint,
    prefill_worker: Mutex<Option<Arc<dyn Worker>>>,
    decode_worker: Mutex<Option<Arc<dyn Worker>>>,
    prefill_tokens: Option<u64>,
    /// Guards every stage transition and the decode fields below, so a commit
    /// and a release cannot interleave halfway through.
    state: Mutex<AttemptState>,
}

impl AttemptInner {
    fn record_decode_outcome(&self, success: bool) {
        let mut state = self.state.lock();
        if success && state.held_kv.external_decode_outcome {
            state.held_kv.producer_succeeded = true;
            if !state.held_kv.body_eof {
                return;
            }
        }
        state.held_kv.decode_outcome.get_or_insert(success);
        state.held_kv.record_abandonment();
    }

    fn decode_body_eof(&self) {
        let success = {
            let mut state = self.state.lock();
            state.held_kv.body_eof = true;
            !state.held_kv.external_decode_outcome || state.held_kv.producer_succeeded
        };
        if success {
            self.record_decode_outcome(true);
        }
    }

    fn finish_prefill(&self) {
        let mut state = self.state.lock();
        if !state.prefill {
            return;
        }
        let mut loads = self.loads.lock();
        let key = (Stage::Prefill, self.prefill.clone());
        if let Some(load) = loads.get_mut(&key) {
            load.prefill_requests -= 1;
            if let Some(tokens) = self.prefill_tokens {
                load.active_prefill_tokens -= tokens;
            } else {
                load.unknown_prefill_requests -= 1;
            }
            load.publish(Stage::Prefill, &self.prefill);
            if load.idle() {
                loads.remove(&key);
            }
        }
        if let Some(worker) = self.prefill_worker.lock().as_ref() {
            worker.decrement_load();
        }
        state.prefill = false;
    }

    fn begin_decode(&self) {
        let mut state = self.state.lock();
        if state.decode != DecodeState::Pending {
            return;
        }
        let Some(endpoint) = state.decode_endpoint.clone() else {
            return;
        };
        let mut loads = self.loads.lock();
        let load = loads
            .get_mut(&(Stage::Decode, endpoint.clone()))
            .expect("admitted decode");
        load.pending_decode_requests -= 1;
        load.active_decode_requests += 1;
        load.publish(Stage::Decode, &endpoint);
        if let Some(since) = state.pending_since {
            metrics::histogram!("vllm_router_pd_decode_pending_seconds")
                .record(since.elapsed().as_secs_f64());
        }
        state.decode = DecodeState::Active;
    }

    fn finish_decode(&self) {
        let mut state = self.state.lock();
        match state.decode {
            DecodeState::Finished => return,
            // Nothing was charged, and closing the state here is what makes a
            // later commit fail instead of leaking a charge.
            DecodeState::Unbound => {
                state.decode = DecodeState::Finished;
                return;
            }
            DecodeState::Pending | DecodeState::Active => {}
        }
        let Some(endpoint) = state.decode_endpoint.clone() else {
            state.decode = DecodeState::Finished;
            return;
        };
        let mut loads = self.loads.lock();
        let key = (Stage::Decode, endpoint.clone());
        if let Some(load) = loads.get_mut(&key) {
            match state.decode {
                DecodeState::Pending => load.pending_decode_requests -= 1,
                DecodeState::Active => load.active_decode_requests -= 1,
                DecodeState::Unbound | DecodeState::Finished => unreachable!(),
            }
            load.publish(Stage::Decode, &endpoint);
            if load.idle() {
                loads.remove(&key);
            }
        }
        if let Some(worker) = self.decode_worker.lock().as_ref() {
            worker.decrement_load();
        }
        state.pending_since = None;
        state.decode = DecodeState::Finished;
    }
}

impl Drop for AttemptInner {
    fn drop(&mut self) {
        self.record_decode_outcome(false);
        self.finish_prefill();
        self.finish_decode();
        // Restart the binding's idle clock after this attempt finishes.
        if let Some(record) = self.state.lock().affinity.take() {
            record.release_ref();
        }
    }
}

#[derive(Clone)]
pub struct PdAttempt(Arc<AttemptInner>);

impl std::fmt::Debug for AttemptInner {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AttemptInner")
            .field("prefill", &self.prefill)
            .field("state", &self.state)
            .finish()
    }
}

impl std::fmt::Debug for PdAttempt {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.0, formatter)
    }
}

/// Body ownership releases D even when a producer still holds another clone
/// of the attempt handle. The last attempt clone separately cleans up P.
struct DecodeBodyGuard {
    attempt: PdAttempt,
    eof: bool,
    sse_errors: Option<super::sse::ErrorObserver>,
}

impl Drop for DecodeBodyGuard {
    fn drop(&mut self) {
        if !self.eof {
            self.attempt.record_decode_outcome(false);
        }
        self.attempt.finish_decode();
    }
}

impl PdAttempt {
    /// An internal cold retry abandons the preceding backend Prefill hold.
    pub(super) fn restart_prefill_hold(&self) {
        let mut state = self.0.state.lock();
        state.held_kv.decode_outcome.get_or_insert(false);
        state.held_kv.record_abandonment();
        state.held_kv = HeldKvState::default();
    }

    /// Called only after a successful Prefill response was fully consumed.
    pub(super) fn prefill_completed(&self, mode: PdHeldKvMode) {
        let mut state = self.0.state.lock();
        state.held_kv.mode = Some(mode);
        state.held_kv.record_abandonment();
    }

    pub(super) fn has_unreclaimable_hold(&self) -> bool {
        let state = self.0.state.lock();
        state
            .held_kv
            .mode
            .is_some_and(|mode| mode != PdHeldKvMode::Transferred)
            && state.held_kv.decode_outcome != Some(true)
    }

    pub(super) fn record_decode_outcome(&self, success: bool) {
        self.0.record_decode_outcome(success);
    }

    /// gRPC producer status, rather than transport EOF, decides success.
    pub(super) fn expect_decode_outcome(&self) {
        self.0.state.lock().held_kv.external_decode_outcome = true;
    }

    pub(super) fn attach_workers(
        &self,
        prefill_worker: Arc<dyn Worker>,
        decode_worker: Arc<dyn Worker>,
    ) {
        self.attach_prefill_worker(prefill_worker);
        self.attach_decode_worker(decode_worker);
    }

    pub(super) fn attach_prefill_worker(&self, prefill_worker: Arc<dyn Worker>) {
        prefill_worker.increment_load();
        *self.0.prefill_worker.lock() = Some(prefill_worker);
    }

    /// Register the decode worker once it is known. Deferred commitments learn
    /// it at dispatch time instead of at admission.
    pub(super) fn attach_decode_worker(&self, decode_worker: Arc<dyn Worker>) {
        decode_worker.increment_load();
        *self.0.decode_worker.lock() = Some(decode_worker);
    }

    /// True until the decode stage is charged, i.e. for a deferred commitment.
    pub(super) fn decode_pending_commit(&self) -> bool {
        self.0.state.lock().decode == DecodeState::Unbound
    }

    /// Committed decode endpoint, or `None` while the commitment is deferred.
    #[cfg(test)]
    pub(super) fn decode_endpoint(&self) -> Option<PdEndpoint> {
        self.0.state.lock().decode_endpoint.clone()
    }
    pub fn finish_prefill(&self) {
        self.0.finish_prefill();
    }
    pub fn begin_decode(&self) {
        self.0.begin_decode();
    }
    pub fn finish_decode(&self) {
        self.0.finish_decode();
    }
    pub fn release(&self) {
        self.record_decode_outcome(false);
        self.finish_prefill();
        self.finish_decode();
    }

    pub fn track_response(self, response: axum::response::Response) -> axum::response::Response {
        if !response.status().is_success() {
            self.record_decode_outcome(false);
        }
        if self.0.state.lock().decode != DecodeState::Active {
            self.release();
            return response;
        }
        let observe_sse = {
            let state = self.0.state.lock();
            state.held_kv.mode.is_some() && !state.held_kv.external_decode_outcome
        } && response
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| {
                value
                    .split(';')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .eq_ignore_ascii_case("text/event-stream")
            })
            && response
                .headers()
                .get_all(axum::http::header::CONTENT_ENCODING)
                .iter()
                .all(|value| {
                    value
                        .to_str()
                        .is_ok_and(|value| value.trim().eq_ignore_ascii_case("identity"))
                });
        let (parts, body) = response.into_parts();
        axum::response::Response::from_parts(parts, self.track_body_inner(body, observe_sse))
    }

    /// Own occupancy through EOF, stream failure, or early client body drop.
    /// Capturing the guard in unfold's state also covers an unpolled body.
    pub fn track_body(self, body: Body) -> Body {
        self.track_body_inner(body, false)
    }

    fn track_body_inner(self, body: Body, observe_sse: bool) -> Body {
        let stream = futures_util::stream::unfold(
            (
                body.into_data_stream(),
                DecodeBodyGuard {
                    attempt: self,
                    eof: false,
                    sse_errors: observe_sse.then(super::sse::ErrorObserver::default),
                },
            ),
            |(mut stream, mut guard)| async move {
                match stream.next().await {
                    Some(Ok(bytes)) => {
                        if guard
                            .sse_errors
                            .as_mut()
                            .is_some_and(|observer| observer.observe(&bytes))
                        {
                            guard.attempt.record_decode_outcome(false);
                            guard.sse_errors = None;
                        }
                        Some((Ok(bytes), (stream, guard)))
                    }
                    Some(Err(error)) => {
                        guard.attempt.record_decode_outcome(false);
                        guard.attempt.finish_decode();
                        Some((Err(error), (stream, guard)))
                    }
                    None => {
                        guard.attempt.0.decode_body_eof();
                        guard.eof = true;
                        guard.attempt.finish_decode();
                        None
                    }
                }
            },
        );
        Body::from_stream(stream)
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod held_kv_tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};
    use axum::response::IntoResponse;
    use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

    fn admit(manager: &PdLoadManager) -> PdAttempt {
        manager.admit_bound(
            Arc::new(BasicWorker::new(
                "http://p".into(),
                WorkerType::Prefill {
                    bootstrap_port: None,
                },
            )),
            Arc::new(BasicWorker::new("http://d".into(), WorkerType::Decode)),
        )
    }

    fn count(handle: &PrometheusHandle, result: &str) -> usize {
        let prefix = format!("vllm_router_pd_held_kv_total{{result=\"{result}\"}} ");
        handle
            .render()
            .lines()
            .find_map(|line| {
                line.strip_prefix(&prefix)
                    .and_then(|value| value.parse().ok())
            })
            .unwrap_or(0)
    }

    #[test]
    fn held_kv_cleanup_is_once_and_never_fabricates_release() {
        for mode in [
            PdHeldKvMode::Unknown,
            PdHeldKvMode::Prefill,
            PdHeldKvMode::Transferred,
        ] {
            let recorder = PrometheusBuilder::new().build_recorder();
            let handle = recorder.handle();
            let _local = metrics::set_default_local_recorder(&recorder);
            let manager = PdLoadManager::default();
            let attempt = admit(&manager);
            let other = admit(&manager);
            attempt.prefill_completed(mode);
            attempt.finish_prefill();
            let clone = attempt.clone();
            attempt.release();
            clone.release();
            drop(attempt);
            drop(clone);
            let expected = if mode == PdHeldKvMode::Transferred {
                "not_held"
            } else {
                "unsupported"
            };
            assert_eq!(count(&handle, expected), 1);
            assert_eq!(count(&handle, "released"), 0);
            assert_eq!(count(&handle, "failed"), 0);
            assert_eq!(
                manager
                    .snapshot()
                    .values()
                    .map(|load| load.pending_decode_requests)
                    .sum::<usize>(),
                1
            );
            drop(other);
            assert!(manager.snapshot().is_empty());
        }
    }

    #[test]
    fn load_cleanup_does_not_imply_completed_prefill() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _local = metrics::set_default_local_recorder(&recorder);
        let manager = PdLoadManager::default();
        let attempt = admit(&manager);
        attempt.release();
        assert_eq!(count(&handle, "unsupported"), 0);
        // Concurrent P can complete after D has already failed.
        attempt.prefill_completed(PdHeldKvMode::Unknown);
        assert_eq!(count(&handle, "unsupported"), 1);
        drop(attempt);
        assert_eq!(count(&handle, "unsupported"), 1);
    }

    #[test]
    fn internal_prefill_retry_reports_the_abandoned_hold_once() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _local = metrics::set_default_local_recorder(&recorder);
        let manager = PdLoadManager::default();
        let attempt = admit(&manager);
        attempt.prefill_completed(PdHeldKvMode::Prefill);
        attempt.restart_prefill_hold();
        assert_eq!(count(&handle, "unsupported"), 1);
        attempt.prefill_completed(PdHeldKvMode::Prefill);
        attempt.release();
        attempt.release();
        drop(attempt);
        assert_eq!(count(&handle, "unsupported"), 2);
        assert!(manager.snapshot().is_empty());
    }

    #[tokio::test]
    async fn held_kv_tracks_eof_failure_and_unpolled_body_drop() {
        for terminal in [
            "success",
            "drop",
            "error",
            "http_error",
            "grpc_failure",
            "grpc_success",
            "grpc_drop",
            "sse_error",
        ] {
            let recorder = PrometheusBuilder::new().build_recorder();
            let handle = recorder.handle();
            let _local = metrics::set_default_local_recorder(&recorder);
            let manager = PdLoadManager::default();
            let attempt = admit(&manager);
            attempt.prefill_completed(PdHeldKvMode::Prefill);
            attempt.finish_prefill();
            attempt.begin_decode();
            if terminal.starts_with("grpc_") {
                attempt.expect_decode_outcome();
                if terminal != "grpc_failure" {
                    attempt.record_decode_outcome(true);
                }
            }
            let response = match terminal {
                "sse_error" => (
                    [("content-type", "text/event-stream")],
                    Body::from_stream(futures_util::stream::iter(
                        b"data: {\"error\":{\"message\":\"decode failed\"}}\n\n"
                            .iter()
                            .map(|byte| {
                                Ok::<_, std::io::Error>(bytes::Bytes::copy_from_slice(&[*byte]))
                            }),
                    )),
                )
                    .into_response(),
                "error" => Body::from_stream(futures_util::stream::iter([Err::<bytes::Bytes, _>(
                    std::io::Error::other("broken decode"),
                )]))
                .into_response(),
                "http_error" => (axum::http::StatusCode::BAD_GATEWAY, "failed").into_response(),
                _ => "done".into_response(),
            };
            let response = attempt.clone().track_response(response);
            if terminal == "drop" || terminal == "grpc_drop" {
                drop(response);
            } else {
                let result = axum::body::to_bytes(response.into_body(), usize::MAX).await;
                assert_eq!(result.is_err(), terminal == "error");
            }
            if terminal == "grpc_failure" {
                // Transport EOF cannot mask a producer-reported error.
                assert_eq!(count(&handle, "unsupported"), 0);
                attempt.record_decode_outcome(false);
            }
            drop(attempt);
            assert_eq!(
                count(&handle, "unsupported"),
                usize::from(terminal != "success" && terminal != "grpc_success"),
                "{terminal}"
            );
            assert!(manager.snapshot().is_empty());
        }
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use std::fmt;

    // Simple custom error for testing error chains
    #[derive(Debug)]
    struct TestError {
        msg: String,
        source: Option<Box<dyn std::error::Error>>,
    }

    impl fmt::Display for TestError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.msg)
        }
    }

    impl std::error::Error for TestError {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            self.source.as_deref()
        }
    }

    #[test]
    fn test_error_chain_single_error() {
        let err = TestError {
            msg: "something broke".into(),
            source: None,
        };
        assert_eq!(error_chain(&err), "something broke");
    }

    #[test]
    fn test_error_chain_nested_errors() {
        let inner = TestError {
            msg: "root cause".into(),
            source: None,
        };
        let outer = TestError {
            msg: "outer error".into(),
            source: Some(Box::new(inner)),
        };
        assert_eq!(error_chain(&outer), "outer error caused by: root cause");
    }

    #[test]
    fn test_error_chain_triple_nested() {
        let root = TestError {
            msg: "connection reset".into(),
            source: None,
        };
        let middle = TestError {
            msg: "HTTP send failed".into(),
            source: Some(Box::new(root)),
        };
        let top = TestError {
            msg: "prefill request failed".into(),
            source: Some(Box::new(middle)),
        };
        assert_eq!(
            error_chain(&top),
            "prefill request failed caused by: HTTP send failed caused by: connection reset"
        );
    }
}
