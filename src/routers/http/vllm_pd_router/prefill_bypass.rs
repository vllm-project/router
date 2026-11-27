//! Policy for allowing local prefill on an already selected Decode worker.
use super::super::pd_kv_state::PdKvState;
use super::super::pd_types::{Loads, Stage};
use super::*;
use crate::backend::preprocess::http_cache_read_fallback;
use crate::config::{PolicyConfig, RouterConfig};
use crate::policies::PromptHashes;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum PdPrefillBypassPolicy {
    #[default]
    #[value(name = "short_suffix")]
    ShortSuffix,
    #[value(name = "busy_prefill")]
    BusyPrefill,
    #[value(name = "short_suffix_or_busy_prefill")]
    ShortSuffixOrBusyPrefill,
}

impl PdPrefillBypassPolicy {
    fn uses_prefill_load(self) -> bool {
        self != Self::ShortSuffix
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, clap::Args)]
#[serde(default)]
pub struct PdPrefillBypassConfig {
    /// Maximum local prefill tokens on Decode. Zero disables bypass.
    #[arg(
        id = "pd_prefill_bypass_max_tokens",
        long = "pd-prefill-bypass-max-tokens",
        default_value_t = 0
    )]
    pub max_tokens: usize,
    /// Maximum fraction of the input left to prefill in the suffix check.
    #[arg(
        id = "pd_prefill_bypass_max_ratio",
        long = "pd-prefill-bypass-max-ratio",
        default_value_t = 0.7
    )]
    pub max_ratio: f64,
    /// Policy used to decide whether to bypass Prefill.
    #[arg(
        id = "pd_prefill_bypass_policy",
        long = "pd-prefill-bypass-policy",
        value_enum,
        default_value = "short_suffix"
    )]
    pub policy: PdPrefillBypassPolicy,
    /// Minimum number of other Prefill requests for Prefill to be busy.
    #[arg(
        id = "pd_prefill_bypass_p_busy_requests",
        long = "pd-prefill-bypass-p-busy-requests"
    )]
    pub p_busy_requests: Option<usize>,
}

impl Default for PdPrefillBypassConfig {
    fn default() -> Self {
        Self {
            max_tokens: 0,
            max_ratio: 0.7,
            policy: PdPrefillBypassPolicy::ShortSuffix,
            p_busy_requests: None,
        }
    }
}

impl PdPrefillBypassConfig {
    pub(crate) fn validate(self, config: &RouterConfig) -> Result<(), String> {
        if !(0.0 < self.max_ratio && self.max_ratio <= 1.0) {
            return Err("pd-prefill-bypass-max-ratio must be in (0, 1]".into());
        }
        if self.max_tokens > 0 {
            if !config.mode.is_pd_mode()
                || config.kv_connector != KvConnector::Nixl
                || !matches!(
                    config.mode.get_decode_policy(&config.policy),
                    PolicyConfig::KvAware { .. }
                )
            {
                return Err(
                    "prefill bypass requires vLLM P/D with NIXL and kv_aware Decode".into(),
                );
            }
            if self.policy.uses_prefill_load() && self.p_busy_requests.unwrap_or(0) == 0 {
                return Err("the prefill bypass load policy requires a positive pd-prefill-bypass-p-busy-requests".into());
            }
        }
        Ok(())
    }
}

impl PdPrefillBypassConfig {
    fn load_allows(
        self,
        loads: &Loads,
        prefill: &PdEndpoint,
        decode: &PdEndpoint,
        own_decode_reservations: usize,
        estimate: PrefillEstimate,
    ) -> bool {
        let decode_load = loads
            .get(&(Stage::Decode, decode.clone()))
            .copied()
            .unwrap_or_default();
        if decode_load.active_decode_requests != 0
            || decode_load.pending_decode_requests != own_decode_reservations
        {
            return false;
        }
        let other_prefills = if self.policy.uses_prefill_load() {
            loads
                .get(&(Stage::Prefill, prefill.clone()))
                .and_then(|load| load.prefill_requests.checked_sub(1))
        } else {
            None
        };
        estimate.allows(&self, other_prefills)
    }
}

const LIMIT_FIELD: &str = "max_local_prefill_tokens";
const ERROR_CODE: &str = "local_prefill_limit_exceeded";
const ERROR_HEADER: &str = "x-vllm-error-code";

#[derive(Clone, Copy)]
struct PrefillEstimate {
    prompt_tokens: usize,
    remaining_tokens: usize,
    worker_revision: u64,
}

impl PrefillEstimate {
    fn within_limit(self, limit: usize) -> bool {
        limit > 0 && self.remaining_tokens < self.prompt_tokens && self.remaining_tokens <= limit
    }

    fn allows(self, config: &PdPrefillBypassConfig, other_prefills: Option<usize>) -> bool {
        if !self.within_limit(config.max_tokens) {
            return false;
        }
        let suffix = self.remaining_tokens as f64 <= config.max_ratio * self.prompt_tokens as f64;
        let busy = config
            .p_busy_requests
            .zip(other_prefills)
            .is_some_and(|(threshold, count)| count >= threshold);
        match config.policy {
            PdPrefillBypassPolicy::ShortSuffix => suffix,
            PdPrefillBypassPolicy::BusyPrefill => busy,
            PdPrefillBypassPolicy::ShortSuffixOrBusyPrefill => suffix || busy,
        }
    }
}

fn eligible_request(path: &str, request: &Value) -> bool {
    if ["kv_transfer_params", "ec_transfer_params", "prompt_embeds"]
        .iter()
        .any(|key| request.get(key).is_some_and(|value| !value.is_null()))
        || ["n", "best_of"].iter().any(|key| {
            request
                .get(key)
                .is_some_and(|value| !value.is_null() && value.as_u64() != Some(1))
        })
        || request.get("use_beam_search").and_then(Value::as_bool) == Some(true)
    {
        return false;
    }
    match path {
        "/v1/chat/completions" => request
            .get("messages")
            .and_then(Value::as_array)
            .is_some_and(|messages| {
                !messages.is_empty()
                    && messages.iter().all(|message| {
                        message.get("content").is_some_and(|content| {
                            content.is_string()
                                || content.as_array().is_some_and(|parts| {
                                    parts.iter().all(|part| {
                                        part.get("type").and_then(Value::as_str) == Some("text")
                                    })
                                })
                        })
                    })
            }),
        "/v1/completions" => request.get("prompt").is_some_and(|prompt| {
            prompt.is_string()
                || prompt.as_array().is_some_and(|tokens| {
                    !tokens.is_empty() && tokens.iter().all(|token| token.as_u64().is_some())
                })
        }),
        _ => false,
    }
}

fn http_rejected(status: http::StatusCode, headers: &http::HeaderMap) -> bool {
    status == http::StatusCode::UNPROCESSABLE_ENTITY
        && headers
            .get(ERROR_HEADER)
            .and_then(|value| value.to_str().ok())
            == Some(ERROR_CODE)
}

fn local_params(limit: usize) -> Value {
    serde_json::json!({
        "do_remote_prefill": false,
        "do_remote_decode": false,
        (LIMIT_FIELD): limit,
    })
}

async fn prepare_estimate(
    state: &PdKvState,
    worker: &dyn Worker,
    policy: &dyn LoadBalancingPolicy,
    request: &Value,
    route: &str,
    headers: Option<&HeaderMap>,
    router_prompts: Option<&Arc<[Vec<u32>]>>,
) -> Option<PrefillEstimate> {
    let worker_revision = state.local_prefill_revision(worker)?;
    let policy = kv_aware_ops(policy)?;
    if http_cache_read_fallback(request).is_some() {
        return None;
    }
    let bases = vec![worker.base_url().to_owned()];
    let verified = state.credit_by_base(&bases, request).await;
    if verified.get(worker.base_url()) != Some(&true) {
        return None;
    }
    let inputs = state
        .tokenize(bases, request, route, headers, router_prompts)
        .await;
    let prompts = &inputs.get(worker.base_url())?.as_ref().ok()?.prompts;
    let [prompt] = prompts.as_ref() else {
        return None;
    };
    let estimates =
        policy.estimate_decode_work(&[(worker, prompts, true)], &mut PromptHashes::new());
    let remaining_tokens = usize::try_from(estimates.first()?.additional_tokens?).ok()?;
    (state.local_prefill_revision(worker) == Some(worker_revision)).then_some(PrefillEstimate {
        prompt_tokens: prompt.len(),
        remaining_tokens,
        worker_revision,
    })
}

fn grpc_rejected(status: &tonic::Status) -> bool {
    status.code() == tonic::Code::FailedPrecondition
        && status
            .metadata()
            .get(ERROR_HEADER)
            .and_then(|value| value.to_str().ok())
            == Some(ERROR_CODE)
}

/// Optional hook installed by the native P/D router at startup.
#[derive(Debug)]
pub(super) struct PrefillBypass {
    config: PdPrefillBypassConfig,
}

#[derive(Clone, Copy)]
pub(super) enum BypassTransport<'a> {
    Http,
    Grpc {
        prepared: &'a crate::backend::PreparedRequest,
        prompts: &'a Arc<[Vec<u32>]>,
    },
}

pub(super) struct BeforePrefill<'a> {
    pub request: &'a Value,
    pub path: &'a str,
    pub headers: Option<&'a HeaderMap>,
    pub prefill: &'a Arc<dyn Worker>,
    pub decode: &'a Arc<dyn Worker>,
    pub attempt: &'a PdAttempt,
    pub commit: Option<&'a DecodeCommitInput>,
    pub pool: DecodePoolSource,
    pub transport: BypassTransport<'a>,
}

pub(super) enum BypassOutcome {
    /// The admitted attempt can run the existing split path.
    Split,
    /// The hook dispatched Decode and owns response tracking.
    Respond(Response),
    /// The hook released its attempt after a pre-generation refusal.
    ReadmitSplit,
}

impl PrefillBypass {
    pub(super) fn enabled(config: PdPrefillBypassConfig) -> Option<Self> {
        (config.max_tokens > 0).then_some(Self { config })
    }

    /// Runs at most once for a client request, just before Prefill dispatch.
    pub(super) async fn before_prefill(
        &self,
        router: &VllmPDRouter,
        input: BeforePrefill<'_>,
    ) -> BypassOutcome {
        let Some(worker) = self.target(router, &input).await else {
            return BypassOutcome::Split;
        };
        let response = match input.transport {
            BypassTransport::Http => self.dispatch_http(router, &input, worker.as_ref()).await,
            BypassTransport::Grpc { prepared, .. } => {
                self.dispatch_grpc(
                    router,
                    prepared.clone(),
                    input.path,
                    worker.as_ref(),
                    input.attempt,
                )
                .await
            }
        };
        match response {
            Some(response) => {
                BypassOutcome::Respond(input.attempt.clone().track_response(response))
            }
            None => {
                input.attempt.release();
                BypassOutcome::ReadmitSplit
            }
        }
    }

    async fn dispatch_grpc(
        &self,
        router: &VllmPDRouter,
        prepared: crate::backend::PreparedRequest,
        path: &str,
        worker: &dyn Worker,
        attempt: &PdAttempt,
    ) -> Option<Response> {
        let start = Instant::now();
        attempt.finish_prefill();
        attempt.begin_decode();
        let response = router
            .grpc_frontend
            .dispatch_decode(
                worker.url(),
                prepared,
                Uuid::new_v4().to_string(),
                Some(local_params(self.config.max_tokens)),
            )
            .await;
        RouterMetrics::record_pd_decode_request(worker.base_url());
        if response
            .extensions()
            .get::<tonic::Status>()
            .is_some_and(grpc_rejected)
        {
            return None;
        }
        RouterMetrics::record_pd_request(path);
        RouterMetrics::record_pd_request_duration(path, start.elapsed());
        if response.status().is_success() {
            record_grpc_decode_stream_failure(grpc_decode_outcome(&response), worker.base_url());
        } else {
            RouterMetrics::record_pd_decode_error(worker.base_url());
        }
        let response = own_grpc_decode(response, attempt);
        Some(response)
    }

    async fn target(
        &self,
        router: &VllmPDRouter,
        input: &BeforePrefill<'_>,
    ) -> Option<Arc<dyn Worker>> {
        let BeforePrefill {
            request,
            path,
            headers,
            prefill,
            decode,
            attempt,
            commit,
            pool,
            transport,
        } = *input;
        let router_prompts = match transport {
            BypassTransport::Http => None,
            BypassTransport::Grpc { prompts, .. } => Some(prompts),
        };
        if !matches!(router.kv_connector, KvConnector::Nixl)
            || router.epd.is_some()
            || !eligible_request(path, request)
            || router
                .get_nixl_push_identity(prefill.base_url(), prefill.dp_rank())
                .is_some()
        {
            return None;
        }
        let state = router.pd_router.kv_state.as_ref()?;
        let policy = router.policy_registry.get_decode_policy();
        let estimate = prepare_estimate(
            state,
            decode.as_ref(),
            policy.as_ref(),
            request,
            path,
            headers,
            router_prompts,
        )
        .await?;
        if !estimate.within_limit(self.config.max_tokens) {
            return None;
        }
        if let Some(commit) = commit {
            let accept = |loads: &Loads, worker: &dyn Worker| {
                state.local_prefill_revision(worker) == Some(estimate.worker_revision)
                    && self.config.load_allows(
                        loads,
                        attempt.prefill_endpoint(),
                        &commit.provisional_endpoint,
                        0,
                        estimate,
                    )
            };
            return router
                .select_and_commit_decode(attempt, commit, request, headers, true, Some(&accept))
                .await
                .ok();
        }
        let endpoint = PdEndpoint::for_worker(decode.as_ref());
        let mut live = router.live_decode_candidates(pool);
        VllmPDRouter::retain_commit_candidates(
            &mut live,
            request.get("model").and_then(Value::as_str),
            Some(&endpoint),
        );
        let worker = live.into_iter().find(|worker| {
            (matches!(pool, DecodePoolSource::Discovery) || Arc::ptr_eq(worker, decode))
                && state.local_prefill_revision(worker.as_ref()) == Some(estimate.worker_revision)
        })?;
        attempt
            .check_reserved_decode(&endpoint, |loads| {
                self.config
                    .load_allows(loads, attempt.prefill_endpoint(), &endpoint, 1, estimate)
            })
            .then_some(worker)
    }

    /// None is reserved for a typed refusal before generation.
    async fn dispatch_http(
        &self,
        router: &VllmPDRouter,
        input: &BeforePrefill<'_>,
        worker: &dyn Worker,
    ) -> Option<Response> {
        let BeforePrefill {
            request,
            path,
            headers,
            attempt,
            ..
        } = *input;
        let start = Instant::now();
        let url = worker.endpoint_url(path);
        let mut request = request.clone();
        request["kv_transfer_params"] = local_params(self.config.max_tokens);
        let builder = router
            .pd_router
            .client
            .post(&url)
            .bearer_auth(std::env::var("OPENAI_API_KEY").unwrap_or_default())
            .header("X-Request-Id", Uuid::new_v4().to_string())
            .json(&request);
        let builder = dp_utils::add_dp_rank_header(builder, worker.dp_rank());
        attempt.finish_prefill();
        attempt.begin_decode();
        let result = otel_http::send_client_request(
            builder,
            headers,
            ClientRequestOptions {
                method: "POST",
                url: &url,
                route: Some(path),
                request_phase: Some("decode"),
            },
        )
        .await;
        RouterMetrics::record_pd_decode_request(worker.base_url());
        if let Ok(response) = &result {
            if http_rejected(response.status(), response.headers()) {
                return None;
            }
        }
        RouterMetrics::record_pd_request(path);
        RouterMetrics::record_pd_request_duration(path, start.elapsed());
        Some(match result {
            Ok(response) => {
                let status = response.status();
                if !status.is_success() {
                    RouterMetrics::record_pd_decode_error(worker.base_url());
                }
                let mut headers = header_utils::preserve_response_headers(response.headers());
                headers.remove(ERROR_HEADER);
                response_with_body(status, &headers, Body::from_stream(response.bytes_stream()))
                    .unwrap_or_else(|error| (StatusCode::BAD_GATEWAY, error).into_response())
            }
            Err(error) => {
                RouterMetrics::record_pd_decode_error(worker.base_url());
                let status = if error.is_timeout() {
                    StatusCode::GATEWAY_TIMEOUT
                } else {
                    StatusCode::BAD_GATEWAY
                };
                (
                    status,
                    format!("Decode request failed to {url}: {}", error_chain(&error)),
                )
                    .into_response()
            }
        })
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn policy_respects_ratio_load_and_hard_limit() {
        let mut config = PdPrefillBypassConfig {
            max_tokens: 256,
            max_ratio: 0.25,
            p_busy_requests: Some(4),
            ..Default::default()
        };
        let mut estimate = PrefillEstimate {
            prompt_tokens: 400,
            remaining_tokens: 100,
            worker_revision: 1,
        };
        assert!(estimate.allows(&config, None));
        estimate.remaining_tokens = 150;
        assert!(!estimate.allows(&config, Some(4)));
        config.policy = PdPrefillBypassPolicy::BusyPrefill;
        assert!(!estimate.allows(&config, Some(3)));
        assert!(estimate.allows(&config, Some(4)));
        assert!(!estimate.allows(&config, None));
        config.policy = PdPrefillBypassPolicy::ShortSuffixOrBusyPrefill;
        assert!(estimate.allows(&config, Some(4)));
        estimate.remaining_tokens = 100;
        assert!(estimate.allows(&config, None));
        estimate.remaining_tokens = 256;
        assert!(estimate.allows(&config, Some(4)));
        estimate.remaining_tokens = 257;
        assert!(!estimate.allows(&config, Some(4)));
        estimate.remaining_tokens = estimate.prompt_tokens;
        assert!(!estimate.allows(&config, Some(4)));
        config.max_tokens = 0;
        estimate.remaining_tokens = 1;
        assert!(!estimate.allows(&config, Some(4)));
    }

    #[test]
    fn bypass_only_accepts_single_text_requests_without_handoff() {
        let request = serde_json::json!({"prompt":[1,2,3]});
        assert!(eligible_request("/v1/completions", &request));
        for request in [
            serde_json::json!({"prompt":"text","n":2}),
            serde_json::json!({"prompt":["a","b"]}),
            serde_json::json!({"prompt":"text","kv_transfer_params":{}}),
            serde_json::json!({"prompt":"text","prompt_embeds":[]}),
        ] {
            assert!(!eligible_request("/v1/completions", &request));
        }
        assert!(!eligible_request(
            "/v1/chat/completions",
            &serde_json::json!({"messages":[{"content":[{"type":"image_url"}]}]})
        ));
    }
}
