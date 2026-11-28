use crate::core::Worker;
use crate::policies::AffinityRecord;
use axum::body::Body;
use futures_util::StreamExt;
use parking_lot::Mutex;
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

// Keep the existing HTTP type paths for callers of the load manager.
pub(crate) use crate::policies::pd::{Loads, Stage};
pub use crate::policies::pd::{
    PdEndpoint, PdLoadSample, PdPrefillInput, PdRequestContext, PdSelection, StageDecision,
};

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
                record(
                    "vllm_router_pd_pending_decode_tokens",
                    self.pending_decode_tokens as f64,
                );
                record(
                    "vllm_router_pd_active_decode_tokens",
                    self.active_decode_tokens as f64,
                );
                record(
                    "vllm_router_pd_unknown_decode_requests",
                    self.unknown_decode_requests as f64,
                );
            }
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

/// Prompt tokens a Prefill decision established, including credited prefix.
fn prompt_tokens(prefill: &StageDecision) -> Option<u64> {
    prefill
        .new_prefill_tokens
        .map(|new| new.saturating_add(prefill.credited_prefix_tokens.unwrap_or(0)))
}

#[derive(Debug, Default)]
pub struct PdLoadManager {
    loads: Arc<Mutex<Loads>>,
}

impl PdLoadManager {
    pub(super) fn snapshot(&self) -> Loads {
        self.loads.lock().clone()
    }

    /// The router supplies selection; this manager owns only admission and
    /// lifecycle state. No policy, worker registry, or connector lives here.
    ///
    /// Both stages are charged here. Callers that dispatch Decode only after
    /// Prefill completes use [`Self::admit_deferred`] instead.
    #[cfg(test)]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(super) fn admit(
        &self,
        select: impl FnOnce(&Loads) -> Option<PdSelection>,
    ) -> Option<(PdSelection, PdAttempt)> {
        self.admit_inner(select, true)
    }

    /// Admission with Decode occupancy charged now or left to the deferred
    /// commit. A probe reserves nothing (`reserve_decode = false`): it takes a
    /// pair but starts no sequence, so booking Decode for it would flip the
    /// whole pool to request counts for the length of every probe.
    pub(super) fn admit_with(
        &self,
        select: impl FnOnce(&Loads) -> Option<PdSelection>,
        reserve_decode: bool,
    ) -> Option<(PdSelection, PdAttempt)> {
        self.admit_inner(select, reserve_decode)
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
        self.admit_inner(select, false)
    }

    fn admit_inner(
        &self,
        select: impl FnOnce(&Loads) -> Option<PdSelection>,
        charge_decode: bool,
    ) -> Option<(PdSelection, PdAttempt)> {
        let mut loads = self.loads.lock();
        let selection = select(&loads)?;
        let prefill = &selection.prefill;
        let decode = &selection.decode;
        let prompt = prompt_tokens(prefill);
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
        let (decode_endpoint, affinity, decode_state, pending_since) = if charge_decode {
            let d = loads
                .entry((Stage::Decode, decode.endpoint.clone()))
                .or_default();
            d.pending_decode_requests += 1;
            match prompt {
                Some(tokens) => d.pending_decode_tokens += tokens,
                None => d.unknown_decode_requests += 1,
            }
            d.publish(Stage::Decode, &decode.endpoint);
            (
                Some(decode.endpoint.clone()),
                decode.affinity.clone(),
                DecodeState::Pending,
                Some(now),
            )
        } else {
            (None, None, DecodeState::Unbound, None)
        };
        let attempt = PdAttempt(Arc::new(AttemptInner {
            admitted_at: now,
            loads: self.loads.clone(),
            prefill: selection.prefill.endpoint.clone(),
            prefill_worker: Mutex::new(None),
            decode_worker: Mutex::new(None),
            decode_prompt_tokens: prompt,
            state: Mutex::new(AttemptState {
                prefill: true,
                prefill_tokens: selection.prefill.new_prefill_tokens,
                decode: decode_state,
                decode_endpoint,
                affinity,
                pending_since,
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
        // The request's logical context is the prompt it carries: shared prefix blocks are
        // counted once per request, never once per block and never once per sequence -
        // sequences share that copy, which is what makes this a physical KV approximation.
        match inner.decode_prompt_tokens {
            Some(tokens) => d.pending_decode_tokens += tokens,
            None => d.unknown_decode_requests += 1,
        }
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
    /// visible to native admission and to `/metrics`. Known P token costs share the native ledger.
    pub(super) fn admit_bound(
        &self,
        prefill_worker: Arc<dyn Worker>,
        decode_worker: Arc<dyn Worker>,
        prefill_tokens: Option<u64>,
    ) -> PdAttempt {
        let prefill = PdEndpoint::for_worker(prefill_worker.as_ref());
        let decode = PdEndpoint::for_worker(decode_worker.as_ref());
        {
            let mut loads = self.loads.lock();
            let p = loads.entry((Stage::Prefill, prefill.clone())).or_default();
            p.prefill_requests += 1;
            match prefill_tokens {
                Some(tokens) => p.active_prefill_tokens += tokens,
                None => p.unknown_prefill_requests += 1,
            }
            p.publish(Stage::Prefill, &prefill);
            let d = loads.entry((Stage::Decode, decode.clone())).or_default();
            d.pending_decode_requests += 1;
            d.unknown_decode_requests += 1;
            d.publish(Stage::Decode, &decode);
        }
        let attempt = PdAttempt(Arc::new(AttemptInner {
            admitted_at: std::time::Instant::now(),
            loads: self.loads.clone(),
            prefill,
            prefill_worker: Mutex::new(None),
            decode_worker: Mutex::new(None),
            // The prompt length is unknown here, so the Decode charge is
            // unknown too and the pool stays on request counts for this
            // attempt.
            decode_prompt_tokens: None,

            state: Mutex::new(AttemptState {
                prefill: true,
                prefill_tokens,
                decode: DecodeState::Pending,
                decode_endpoint: Some(decode),
                affinity: None,
                pending_since: Some(std::time::Instant::now()),
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
    prefill_tokens: Option<u64>,
    decode: DecodeState,
    /// Committed decode endpoint; `None` while the commitment is deferred.
    decode_endpoint: Option<PdEndpoint>,
    /// Decode binding held alive until this attempt finishes.
    affinity: Option<Arc<AffinityRecord>>,
    /// When the decode slot was charged, for the pending histogram.
    pending_since: Option<std::time::Instant>,
}

struct AttemptInner {
    admitted_at: std::time::Instant,
    loads: Arc<Mutex<Loads>>,
    prefill: PdEndpoint,
    prefill_worker: Mutex<Option<Arc<dyn Worker>>>,
    decode_worker: Mutex<Option<Arc<dyn Worker>>>,
    /// Prompt length this attempt will hold on a Decode engine, counted once per request:
    /// the request's sequences share one copy of those blocks. `None` when the prompt was
    /// never measured.
    decode_prompt_tokens: Option<u64>,

    /// Guards every stage transition and the decode fields below, so a commit
    /// and a release cannot interleave halfway through.
    state: Mutex<AttemptState>,
}

impl AttemptInner {
    /// Tokens this attempt charged to a Decode engine, derived from the prompt length it
    /// recorded at admission. Deriving instead of mirroring removes two fields that had to
    /// stay in step with an immutable one.
    fn decode_charge(&self) -> u64 {
        self.decode_prompt_tokens.unwrap_or(0)
    }

    /// Whether this attempt charged as an unknown-length Decode request, which the release
    /// must undo exactly once.
    fn decode_unknown(&self) -> bool {
        self.decode_prompt_tokens.is_none()
    }
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
            if let Some(tokens) = state.prefill_tokens {
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
        let tokens = self.decode_charge();
        load.pending_decode_tokens = load.pending_decode_tokens.saturating_sub(tokens);
        load.active_decode_tokens += tokens;
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
                DecodeState::Pending => {
                    load.pending_decode_requests -= 1;
                    let tokens = self.decode_charge();
                    load.pending_decode_tokens = load.pending_decode_tokens.saturating_sub(tokens);
                }
                DecodeState::Active => {
                    load.active_decode_requests -= 1;
                    let tokens = self.decode_charge();
                    load.active_decode_tokens = load.active_decode_tokens.saturating_sub(tokens);
                }
                DecodeState::Unbound | DecodeState::Finished => unreachable!(),
            }
            // Undo the unknown-length charge too, or this attempt keeps the pool on
            // request counts after it has already finished.
            if self.decode_unknown() {
                load.unknown_decode_requests = load.unknown_decode_requests.saturating_sub(1);
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
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(super) fn decode_endpoint(&self) -> Option<PdEndpoint> {
        self.0.state.lock().decode_endpoint.clone()
    }
    /// A rejected LocalAppend can start one remote P leg on the same D reservation.
    pub(super) fn restart_prefill(&self, tokens: Option<u64>) {
        let mut state = self.0.state.lock();
        debug_assert!(!state.prefill);
        let mut loads = self.0.loads.lock();
        let load = loads
            .entry((Stage::Prefill, self.0.prefill.clone()))
            .or_default();
        load.prefill_requests += 1;
        match tokens {
            Some(tokens) => load.active_prefill_tokens += tokens,
            None => load.unknown_prefill_requests += 1,
        }
        load.publish(Stage::Prefill, &self.0.prefill);
        if let Some(worker) = self.0.prefill_worker.lock().as_ref() {
            worker.increment_load();
        }
        state.prefill = true;
        state.prefill_tokens = tokens;
    }

    /// Adjust an in-flight P leg on a warm-to-cold retry.
    pub(super) fn reprice_prefill(&self, tokens: Option<u64>) {
        let mut state = self.0.state.lock();
        if !state.prefill || state.prefill_tokens == tokens {
            return;
        }
        let mut loads = self.0.loads.lock();
        let load = loads
            .get_mut(&(Stage::Prefill, self.0.prefill.clone()))
            .expect("active prefill charge");
        match state.prefill_tokens {
            Some(old) => load.active_prefill_tokens -= old,
            None => load.unknown_prefill_requests -= 1,
        }
        match tokens {
            Some(new) => load.active_prefill_tokens += new,
            None => load.unknown_prefill_requests += 1,
        }
        state.prefill_tokens = tokens;
        load.publish(Stage::Prefill, &self.0.prefill);
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

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
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
