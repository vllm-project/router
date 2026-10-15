//! Observational inference probe.
//!
//! Periodically sends a real one-token generation (`POST /v1/completions` with
//! `max_tokens: 1`) to every registered worker and reports the outcome. This is
//! the detection half of the failure class described in
//! [#158](https://github.com/vllm-project/router/issues/158): a worker whose
//! process is up — and whose `/health` therefore answers 200 — but which cannot
//! complete a generation.
//!
//! # Non-mutation invariant
//!
//! This module must never call [`Worker::set_healthy`], must never record a
//! circuit-breaker outcome, and must never change policy-visible availability.
//! Worker health stays the exclusive responsibility of the health checker
//! ([`crate::core::worker`]'s `http_health_probe` and the `HealthConfig`
//! thresholds). Detection and enforcement are deliberately separated, so a
//! defect here can produce a wrong *report* but never an outage.
//!
//! Probe requests also never touch worker load counters, so the sweep cannot
//! perturb load-based routing (see PR #216).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::core::worker::{Worker, WORKER_CLIENT};

/// Result of a single observational inference probe.
#[derive(Debug, Clone)]
pub struct InferenceProbeOutcome {
    /// The worker completed a real one-token generation and returned a
    /// well-formed completion choice.
    pub ok: bool,
    /// Low-cardinality classifier for metric labels: one of `ok`,
    /// `request_failed`, `http_error`, `unreadable_body`, `no_choice`,
    /// `abnormal_finish`, `zero_tokens`.
    pub reason: &'static str,
    /// Human-readable reason, suitable for a log line.
    pub detail: String,
    /// Wall-clock latency of the probe request.
    pub latency_ms: u128,
}

impl InferenceProbeOutcome {
    fn failed(reason: &'static str, detail: String, elapsed: Duration) -> Self {
        Self {
            ok: false,
            reason,
            detail,
            latency_ms: elapsed.as_millis(),
        }
    }
}

/// Probe one worker with a real one-token generation.
///
/// `api_key` is the router's worker credential (`RouterConfig::api_key`, "the
/// api key used for the authorization with the worker"). When set it is sent as
/// `Authorization: Bearer <key>`, the same way the proxy authenticates user
/// requests, so that protected workers are not reported as broken merely
/// because the probe was unauthenticated.
///
/// Success requires more than a 2xx status: the worker is only reported as
/// serving when it returns a well-formed completion choice, i.e. the inference
/// actually produced a token. A bare status check would report success for a
/// worker whose engine answered but produced nothing.
pub async fn probe_inference(
    worker: &dyn Worker,
    timeout_secs: u64,
    api_key: Option<&str>,
) -> InferenceProbeOutcome {
    use crate::routers::http::dp_utils::add_dp_rank_header;

    let url = format!("{}/v1/completions", worker.base_url());
    let body = serde_json::json!({
        "prompt": "hi",
        "max_tokens": 1,
    });

    let mut request = WORKER_CLIENT
        .post(&url)
        .timeout(Duration::from_secs(timeout_secs))
        .json(&body);
    if worker.is_dp_aware() {
        // Pin the logical rank, exactly as a proxied user request would be.
        request = add_dp_rank_header(request, worker.dp_rank());
    }
    if let Some(key) = api_key {
        request = request.header("Authorization", format!("Bearer {key}"));
    }

    let started = Instant::now();
    let response = match request.send().await {
        Ok(response) => response,
        Err(error) => {
            return InferenceProbeOutcome::failed(
                "request_failed",
                format!("request failed: {error}"),
                started.elapsed(),
            )
        }
    };

    let status = response.status();
    if !status.is_success() {
        return InferenceProbeOutcome::failed(
            "http_error",
            format!("HTTP {status}"),
            started.elapsed(),
        );
    }

    let payload: serde_json::Value = match response.json().await {
        Ok(payload) => payload,
        Err(error) => {
            return InferenceProbeOutcome::failed(
                "unreadable_body",
                format!("unreadable response body: {error}"),
                started.elapsed(),
            )
        }
    };

    // A successful probe must demonstrate a completed generation:
    //   1. a completion choice is present, and
    //   2. it terminated normally (finish_reason is "stop" or "length"; a
    //      non-null abnormal reason such as "error" means the engine itself
    //      failed mid-generation), and
    //   3. at least one token was actually generated, so an empty
    //      `choices: []` or a zero-token response cannot masquerade as success.
    //
    // We do *not* require non-empty `text`: a legitimately generated
    // end-of-sequence token can yield empty text while still being a
    // completed generation.
    let first_choice = payload
        .get("choices")
        .and_then(|choices| choices.as_array())
        .and_then(|choices| choices.first());

    let choice_has_text_field = first_choice.is_some_and(|choice| {
        choice.get("text").is_some_and(|text| text.is_string())
            || choice
                .get("message")
                .and_then(|message| message.get("content"))
                .is_some_and(|content| content.is_string())
    });
    let finish_reason_ok = first_choice.is_some_and(|choice| {
        choice
            .get("finish_reason")
            .and_then(|reason| reason.as_str())
            .is_some_and(|reason| matches!(reason, "stop" | "length"))
    });
    let tokens_generated = payload
        .get("usage")
        .and_then(|usage| usage.get("completion_tokens"))
        .and_then(|tokens| tokens.as_u64())
        .unwrap_or(0);

    if !choice_has_text_field {
        return InferenceProbeOutcome::failed(
            "no_choice",
            "response carried no completion choice".to_string(),
            started.elapsed(),
        );
    }
    if !finish_reason_ok {
        return InferenceProbeOutcome::failed(
            "abnormal_finish",
            "completion did not terminate normally (missing or abnormal finish_reason)".to_string(),
            started.elapsed(),
        );
    }
    if tokens_generated < 1 {
        return InferenceProbeOutcome::failed(
            "zero_tokens",
            "completion produced no tokens".to_string(),
            started.elapsed(),
        );
    }

    InferenceProbeOutcome {
        ok: true,
        reason: "ok",
        detail: "ok".to_string(),
        latency_ms: started.elapsed().as_millis(),
    }
}

/// Shutdown handle for the periodic inference probe task.
///
/// Dropping the handle does not stop the sweep; call [`Self::stop`] to end it.
pub struct InferenceProbeHandle {
    shutdown: Arc<AtomicBool>,
}

impl InferenceProbeHandle {
    /// Signal the probe task to stop after its current sweep.
    pub fn stop(&self) {
        self.shutdown.store(true, Ordering::Release);
    }

    /// Whether the probe task has been asked to stop.
    pub fn is_stopped(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }
}

impl std::fmt::Debug for InferenceProbeHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceProbeHandle")
            .field("stopped", &self.is_stopped())
            .finish()
    }
}

/// Build a shutdown handle for a probe task.
pub(crate) fn probe_handle() -> (Arc<AtomicBool>, InferenceProbeHandle) {
    let shutdown = Arc::new(AtomicBool::new(false));
    let handle = InferenceProbeHandle {
        shutdown: shutdown.clone(),
    };
    (shutdown, handle)
}

/// Whether the probe should stop before its next sweep.
pub(crate) fn should_stop(shutdown: &AtomicBool) -> bool {
    shutdown.load(Ordering::Acquire)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, DPAwareWorker, WorkerType};
    use axum::response::IntoResponse;
    use axum::{extract::State, http::HeaderMap, routing::post, Json, Router};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc as StdArc;

    /// What the mock completions endpoint replies with.
    #[derive(Debug, Clone, Copy)]
    enum Reply {
        /// A well-formed one-token completion.
        Completion,
        /// A completion whose finish_reason is abnormal (engine failed).
        AbnormalFinish,
        /// A completion that reports zero generated tokens.
        ZeroTokens,
        /// A completion that reports null finish_reason with a token.
        NullFinishReason,
        /// HTTP 200, but the engine produced nothing.
        EmptyChoices,
        /// The worker cannot infer at all.
        ServerError,
    }

    /// Records what the mock endpoint saw.
    struct Observations {
        reply: Reply,
        saw_request: StdArc<AtomicBool>,
        got_minimal_payload: StdArc<AtomicBool>,
        got_dp_rank: StdArc<AtomicBool>,
        got_kv_params: StdArc<AtomicBool>,
        auth_header: StdArc<std::sync::Mutex<Option<String>>>,
    }

    async fn spawn_mock(reply: Reply) -> (String, StdArc<Observations>) {
        async fn completions_handler(
            State(obs): State<StdArc<Observations>>,
            headers: HeaderMap,
            Json(payload): Json<serde_json::Value>,
        ) -> axum::response::Response {
            obs.saw_request.store(true, Ordering::Release);
            if payload.get("max_tokens").and_then(|v| v.as_u64()) == Some(1)
                && payload.get("prompt").and_then(|v| v.as_str()).is_some()
            {
                obs.got_minimal_payload.store(true, Ordering::Release);
            }
            if headers.get("X-data-parallel-rank").is_some() {
                obs.got_dp_rank.store(true, Ordering::Release);
            }
            if payload.get("kv_transfer_params").is_some() {
                obs.got_kv_params.store(true, Ordering::Release);
            }
            if let Some(value) = headers.get("Authorization") {
                *obs.auth_header.lock().unwrap() = value.to_str().ok().map(str::to_owned);
            }

            match obs.reply {
                Reply::Completion => Json(serde_json::json!({
                    "id": "cmpl-probe",
                    "choices": [{"text": ".", "index": 0, "finish_reason": "length"}],
                    "usage": {"completion_tokens": 1}
                }))
                .into_response(),
                Reply::AbnormalFinish => Json(serde_json::json!({
                    "id": "cmpl-probe",
                    "choices": [{"text": ".", "index": 0, "finish_reason": "error"}],
                    "usage": {"completion_tokens": 1}
                }))
                .into_response(),
                Reply::ZeroTokens => Json(serde_json::json!({
                    "id": "cmpl-probe",
                    "choices": [{"text": ".", "index": 0, "finish_reason": "length"}],
                    "usage": {"completion_tokens": 0}
                }))
                .into_response(),
                Reply::NullFinishReason => Json(serde_json::json!({
                    "id": "cmpl-probe",
                    "choices": [{"text": ".", "index": 0, "finish_reason": null}],
                    "usage": {"completion_tokens": 1}
                }))
                .into_response(),
                Reply::EmptyChoices => Json(serde_json::json!({
                    "id": "cmpl-probe",
                    "choices": []
                }))
                .into_response(),
                Reply::ServerError => (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": "engine cannot infer"})),
                )
                    .into_response(),
            }
        }

        let obs = StdArc::new(Observations {
            reply,
            saw_request: StdArc::new(AtomicBool::new(false)),
            got_minimal_payload: StdArc::new(AtomicBool::new(false)),
            got_dp_rank: StdArc::new(AtomicBool::new(false)),
            got_kv_params: StdArc::new(AtomicBool::new(false)),
            auth_header: StdArc::new(std::sync::Mutex::new(None)),
        });

        let app = Router::new()
            .route("/v1/completions", post(completions_handler))
            .with_state(obs.clone());

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            axum::serve(listener, app).await.ok();
        });
        tokio::time::sleep(Duration::from_millis(50)).await;

        (format!("http://127.0.0.1:{}", port), obs)
    }

    /// The motivating case: a worker that the health checker considers healthy
    /// (its `/health` answers 200) but which cannot actually serve inference.
    /// The probe must report the failure and must *not* act on it.
    #[tokio::test]
    async fn probe_reports_worker_that_cannot_serve_without_mutating_health() {
        let (url, obs) = spawn_mock(Reply::ServerError).await;
        let worker = BasicWorker::new(url, WorkerType::Regular);
        assert!(worker.is_healthy(), "worker starts healthy");

        let outcome = probe_inference(&worker, 5, None).await;

        assert!(!outcome.ok, "a failing inference must be reported as such");
        assert_eq!(outcome.reason, "http_error");
        assert!(
            obs.saw_request.load(Ordering::Acquire),
            "the probe must reach POST /v1/completions"
        );
        assert!(
            obs.got_minimal_payload.load(Ordering::Acquire),
            "the probe body must carry max_tokens=1 and a prompt"
        );
        assert!(
            worker.is_healthy(),
            "the inference probe reports only; it must not mark the worker unhealthy"
        );
    }

    /// A successful probe must not flip a worker healthy either, and must not
    /// touch load accounting (PR #216).
    #[tokio::test]
    async fn probe_reports_success_without_mutating_health_or_load() {
        let (url, obs) = spawn_mock(Reply::Completion).await;
        let worker = BasicWorker::new(url, WorkerType::Regular).with_initial_health(false);
        assert_eq!(worker.load(), 0);

        let outcome = probe_inference(&worker, 5, None).await;

        assert!(outcome.ok, "unexpected failure: {}", outcome.detail);
        assert_eq!(outcome.reason, "ok");
        assert!(obs.saw_request.load(Ordering::Acquire));
        assert_eq!(worker.load(), 0, "the probe must not mutate worker load");
        assert!(
            !worker.is_healthy(),
            "the inference probe reports only; it must not mark the worker healthy"
        );
    }

    /// HTTP 200 is not enough: an engine that answers without producing a
    /// completion has not actually served the request.
    #[tokio::test]
    async fn probe_rejects_http_success_without_a_completion() {
        let (url, _obs) = spawn_mock(Reply::EmptyChoices).await;
        let worker = BasicWorker::new(url, WorkerType::Regular);

        let outcome = probe_inference(&worker, 5, None).await;

        assert!(!outcome.ok);
        assert_eq!(outcome.reason, "no_choice");
    }

    /// An abnormal finish_reason (e.g. "error") is not a completed generation
    /// even if the worker returned 200 and emitted a text field.
    #[tokio::test]
    async fn probe_rejects_abnormal_finish_reason() {
        let (url, _obs) = spawn_mock(Reply::AbnormalFinish).await;
        let worker = BasicWorker::new(url, WorkerType::Regular);

        let outcome = probe_inference(&worker, 5, None).await;

        assert!(
            !outcome.ok,
            "abnormal finish_reason must not count as success"
        );
        assert_eq!(outcome.reason, "abnormal_finish");
    }

    /// A null finish_reason is treated as abnormal for the same reason:
    /// the probe cannot confirm a normal termination.
    #[tokio::test]
    async fn probe_rejects_null_finish_reason() {
        let (url, _obs) = spawn_mock(Reply::NullFinishReason).await;
        let worker = BasicWorker::new(url, WorkerType::Regular);

        let outcome = probe_inference(&worker, 5, None).await;

        assert!(!outcome.ok, "null finish_reason must not count as success");
        assert_eq!(outcome.reason, "abnormal_finish");
    }

    /// Zero generated tokens is not a completed generation: a worker that
    /// answered but produced no output is still broken from the probe's
    /// perspective.
    #[tokio::test]
    async fn probe_rejects_zero_token_completion() {
        let (url, _obs) = spawn_mock(Reply::ZeroTokens).await;
        let worker = BasicWorker::new(url, WorkerType::Regular);

        let outcome = probe_inference(&worker, 5, None).await;

        assert!(
            !outcome.ok,
            "zero-token completion must not count as success"
        );
        assert_eq!(outcome.reason, "zero_tokens");
    }

    /// A well-formed one-token completion with a normal finish_reason is
    /// accepted even when the text is empty (legitimate end-of-sequence).
    #[tokio::test]
    async fn probe_accepts_completion_with_empty_text_and_valid_finish() {
        let (url, obs) = spawn_mock(Reply::Completion).await;
        let worker = BasicWorker::new(url, WorkerType::Regular);

        let outcome = probe_inference(&worker, 5, None).await;

        assert!(outcome.ok, "unexpected failure: {}", outcome.detail);
        assert_eq!(outcome.reason, "ok");
        assert!(obs.saw_request.load(Ordering::Acquire));
    }

    /// The probe pins the DP rank like a proxied user request, and sends no
    /// connector-specific P/D envelope: a worker-level probe has no decode peer
    /// to target (see the scope boundary in #367).
    #[tokio::test]
    async fn probe_pins_dp_rank_and_sends_no_pd_envelope() {
        let (url, obs) = spawn_mock(Reply::Completion).await;
        let worker = DPAwareWorker::new(
            url,
            2,
            4,
            WorkerType::Prefill {
                bootstrap_port: Some(9000),
            },
        );

        let outcome = probe_inference(&worker, 5, None).await;

        assert!(outcome.ok, "unexpected failure: {}", outcome.detail);
        assert!(
            obs.got_dp_rank.load(Ordering::Acquire),
            "the probe must pin X-data-parallel-rank"
        );
        assert!(
            !obs.got_kv_params.load(Ordering::Acquire),
            "a worker-level probe must not synthesize a P/D envelope"
        );
    }

    /// Workers behind an API key must not be reported as broken just because
    /// the probe was unauthenticated, so the router's worker credential rides
    /// along exactly as it does on proxied user requests.
    #[tokio::test]
    async fn probe_carries_the_worker_credential_when_configured() {
        let (url, obs) = spawn_mock(Reply::Completion).await;
        let worker = BasicWorker::new(url, WorkerType::Regular);

        let outcome = probe_inference(&worker, 5, Some("secret-key")).await;

        assert!(outcome.ok, "unexpected failure: {}", outcome.detail);
        assert_eq!(
            obs.auth_header.lock().unwrap().as_deref(),
            Some("Bearer secret-key"),
            "the probe must authenticate like a proxied user request"
        );

        // With no credential configured, no Authorization header is sent.
        let (url, obs) = spawn_mock(Reply::Completion).await;
        let worker = BasicWorker::new(url, WorkerType::Regular);
        let outcome = probe_inference(&worker, 5, None).await;
        assert!(outcome.ok, "unexpected failure: {}", outcome.detail);
        assert_eq!(*obs.auth_header.lock().unwrap(), None);
    }
}
