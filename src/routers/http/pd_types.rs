use crate::core::Worker;
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
    pub token_ids: Option<Arc<[u32]>>,
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
    pub headers: Option<&'a RequestHeaders>,
    /// Stable session identity for the optional kv_aware per-stage binding.
    /// Resolved at request entry, only when the affinity switch is on;
    /// `None` selects per request on both stages.
    pub affinity_key: Option<AffinityKey>,
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
    /// Conservative transfer estimate; unknown until route inputs are verified.
    pub estimated_transfer_tokens: Option<u64>,
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
    pub(super) fn admit(
        self: &Arc<Self>,
        select: impl FnOnce(&Loads) -> Option<PdSelection>,
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
        let d = loads
            .entry((Stage::Decode, decode.endpoint.clone()))
            .or_default();
        d.pending_decode_requests += 1;
        d.publish(Stage::Decode, &decode.endpoint);
        let attempt = PdAttempt(Arc::new(AttemptInner {
            admitted_at: std::time::Instant::now(),
            loads: self.loads.clone(),
            prefill: selection.prefill.endpoint.clone(),
            decode: selection.decode.endpoint.clone(),
            prefill_worker: Mutex::new(None),
            decode_worker: Mutex::new(None),
            prefill_tokens: selection.prefill.new_prefill_tokens,
            affinity: selection
                .prefill
                .affinity
                .iter()
                .chain(selection.decode.affinity.iter())
                .cloned()
                .collect(),
            state: Mutex::new(AttemptState {
                prefill: true,
                decode: DecodeState::Pending,
            }),
        }));
        Some((selection, attempt))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecodeState {
    Pending,
    Active,
    Finished,
}

#[derive(Debug)]
struct AttemptState {
    prefill: bool,
    decode: DecodeState,
}

struct AttemptInner {
    admitted_at: std::time::Instant,
    loads: Arc<Mutex<Loads>>,
    prefill: PdEndpoint,
    decode: PdEndpoint,
    prefill_worker: Mutex<Option<Arc<dyn Worker>>>,
    decode_worker: Mutex<Option<Arc<dyn Worker>>>,
    prefill_tokens: Option<u64>,
    /// Session bindings this attempt holds alive until cleanup.
    affinity: Vec<Arc<AffinityRecord>>,
    state: Mutex<AttemptState>,
}

impl AttemptInner {
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
        let mut loads = self.loads.lock();
        let load = loads
            .get_mut(&(Stage::Decode, self.decode.clone()))
            .expect("admitted decode");
        load.pending_decode_requests -= 1;
        load.active_decode_requests += 1;
        load.publish(Stage::Decode, &self.decode);
        metrics::histogram!("vllm_router_pd_decode_pending_seconds")
            .record(self.admitted_at.elapsed().as_secs_f64());
        state.decode = DecodeState::Active;
    }

    fn finish_decode(&self) {
        let mut state = self.state.lock();
        if state.decode == DecodeState::Finished {
            return;
        }
        let mut loads = self.loads.lock();
        let key = (Stage::Decode, self.decode.clone());
        if let Some(load) = loads.get_mut(&key) {
            match state.decode {
                DecodeState::Pending => load.pending_decode_requests -= 1,
                DecodeState::Active => load.active_decode_requests -= 1,
                DecodeState::Finished => unreachable!(),
            }
            load.publish(Stage::Decode, &self.decode);
            if load.idle() {
                loads.remove(&key);
            }
        }
        if let Some(worker) = self.decode_worker.lock().as_ref() {
            worker.decrement_load();
        }
        state.decode = DecodeState::Finished;
    }
}

impl Drop for AttemptInner {
    fn drop(&mut self) {
        self.finish_prefill();
        self.finish_decode();
        // Restart each binding's idle clock now that this attempt is done;
        // while the references were held, the bindings could not expire.
        for record in &self.affinity {
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
            .field("decode", &self.decode)
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
struct DecodeBodyGuard(PdAttempt);

impl Drop for DecodeBodyGuard {
    fn drop(&mut self) {
        self.0.finish_decode();
    }
}

impl PdAttempt {
    pub(super) fn attach_workers(
        &self,
        prefill_worker: Arc<dyn Worker>,
        decode_worker: Arc<dyn Worker>,
    ) {
        prefill_worker.increment_load();
        decode_worker.increment_load();
        *self.0.prefill_worker.lock() = Some(prefill_worker);
        *self.0.decode_worker.lock() = Some(decode_worker);
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
        self.finish_prefill();
        self.finish_decode();
    }

    pub fn track_response(self, response: axum::response::Response) -> axum::response::Response {
        if self.0.state.lock().decode != DecodeState::Active {
            self.release();
            return response;
        }
        let (parts, body) = response.into_parts();
        axum::response::Response::from_parts(parts, self.track_body(body))
    }

    /// Own occupancy through EOF, stream failure, or early client body drop.
    /// Capturing the guard in unfold's state also covers an unpolled body.
    pub fn track_body(self, body: Body) -> Body {
        let stream = futures_util::stream::unfold(
            (body.into_data_stream(), DecodeBodyGuard(self)),
            |(mut stream, attempt)| async move {
                match stream.next().await {
                    Some(Ok(bytes)) => Some((Ok(bytes), (stream, attempt))),
                    Some(Err(error)) => {
                        attempt.0.finish_decode();
                        Some((Err(error), (stream, attempt)))
                    }
                    None => {
                        attempt.0.finish_decode();
                        None
                    }
                }
            },
        );
        Body::from_stream(stream)
    }
}

#[cfg(test)]
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
