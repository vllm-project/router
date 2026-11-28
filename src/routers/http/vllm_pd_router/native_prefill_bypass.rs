//! Native P/D adapter for optional Prefill bypass.

use std::future::Future;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use axum::{
    body::Body,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use serde_json::Value;
use uuid::Uuid;

use super::super::{
    dp_utils,
    pd_kv_state::PdKvState,
    pd_types::{error_chain, Loads, PdAttempt, PdEndpoint, Stage},
    prefill_bypass::{
        admit_with_bypass, grpc_rejected, http_rejected, BypassAttempt, PdPrefillBypassConfig,
        PrefillBypass, PrefillDispatch, PrefillEstimate, ERROR_HEADER,
    },
};
use super::{
    grpc_decode_outcome, own_grpc_decode, record_grpc_decode_stream_failure, response_with_body,
    Admission, AdmittedPair, DecodeCommitGuard, DecodeCommitInput, DecodePoolSource,
};
use crate::backend::{EngineFrontend, PreparedRequest};
use crate::config::{KvConnector, PolicyConfig, RouterConfig};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::otel_http::{self, ClientRequestOptions};
use crate::policies::{kv_aware_ops, LoadBalancingPolicy, PromptHashes};
use crate::routers::header_utils;

pub(crate) fn validate_prefill_bypass(
    config: PdPrefillBypassConfig,
    router: &RouterConfig,
) -> Result<(), String> {
    config.validate()?;
    if config.is_enabled()
        && (!router.mode.is_pd_mode()
            || router.kv_connector != KvConnector::Nixl
            || !matches!(
                router.mode.get_decode_policy(&router.policy),
                PolicyConfig::KvAware { .. }
            ))
    {
        return Err("prefill bypass requires vLLM P/D with NIXL and kv_aware Decode".into());
    }
    Ok(())
}

/// The native admission operations used while validating the selected pair.
#[async_trait]
pub(super) trait DecodeAdmission: Sync {
    fn supports_prefill(&self, prefill: &dyn Worker) -> bool;

    fn decode_policy(&self) -> Arc<dyn LoadBalancingPolicy>;

    fn eligible_decode_candidates(
        &self,
        pool: DecodePoolSource,
        model: Option<&str>,
        endpoint: &PdEndpoint,
    ) -> Vec<Arc<dyn Worker>>;

    async fn commit_provisional_decode(
        &self,
        attempt: &PdAttempt,
        commit: &DecodeCommitInput,
        request: &Value,
        headers: Option<&HeaderMap>,
        guard: &DecodeCommitGuard<'_>,
    ) -> Option<Arc<dyn Worker>>;
}

pub(super) struct NativeBypass<'a> {
    pub feature: &'a PrefillBypass,
    pub state: &'a PdKvState,
    pub admission: &'a dyn DecodeAdmission,
}

#[derive(Clone, Copy)]
pub(super) enum Transport<'a> {
    Http {
        client: &'a reqwest::Client,
        pool: DecodePoolSource,
    },
    Grpc {
        frontend: &'a EngineFrontend,
        prepared: &'a PreparedRequest,
        prompts: &'a Arc<[Vec<u32>]>,
    },
}

#[derive(Clone, Copy)]
pub(super) struct BeforePrefill<'a> {
    pub request: &'a Value,
    pub path: &'a str,
    pub headers: Option<&'a HeaderMap>,
    pub transport: Transport<'a>,
}

impl BeforePrefill<'_> {
    fn pool(&self) -> DecodePoolSource {
        match self.transport {
            Transport::Http { pool, .. } => pool,
            Transport::Grpc { .. } => DecodePoolSource::Registry,
        }
    }

    fn router_prompts(&self) -> Option<&Arc<[Vec<u32>]>> {
        match self.transport {
            Transport::Http { .. } => None,
            Transport::Grpc { prompts, .. } => Some(prompts),
        }
    }
}

pub(super) struct SplitDispatch {
    pub prefill: Arc<dyn Worker>,
    pub decode: Arc<dyn Worker>,
    pub attempt: PdAttempt,
    pub commit: Option<DecodeCommitInput>,
}

impl SplitDispatch {
    fn new(pair: AdmittedPair, input: BeforePrefill<'_>) -> Self {
        let AdmittedPair {
            prefill,
            decode,
            admission:
                Admission {
                    selection,
                    attempt,
                    decode_tokens,
                    affinity_key,
                },
        } = pair;
        let commit = attempt.decode_pending_commit().then(|| {
            DecodeCommitInput::for_request(
                input.pool(),
                selection.decode.endpoint,
                decode_tokens,
                affinity_key,
                input.request,
                input.path,
                input.router_prompts().cloned(),
            )
        });
        Self {
            prefill,
            decode,
            attempt,
            commit,
        }
    }
}

pub(super) async fn admit_for_dispatch<F, Fut>(
    bypass: Option<NativeBypass<'_>>,
    input: BeforePrefill<'_>,
    mut admit: F,
) -> PrefillDispatch<SplitDispatch>
where
    F: FnMut(bool) -> Fut,
    Fut: Future<Output = Result<AdmittedPair, Box<Response>>>,
{
    admit_with_bypass(
        |readmit| {
            let admitted = admit(readmit);
            async move { admitted.await.map(|pair| SplitDispatch::new(pair, input)) }
        },
        |split| try_bypass(bypass, input, split),
    )
    .await
}

async fn try_bypass(
    bypass: Option<NativeBypass<'_>>,
    input: BeforePrefill<'_>,
    split: SplitDispatch,
) -> BypassAttempt<SplitDispatch> {
    let Some(bypass) = bypass else {
        return BypassAttempt::Split(split);
    };
    let Some(worker) = bypass_target(&bypass, input, &split).await else {
        return BypassAttempt::Split(split);
    };
    let params = bypass.feature.transfer_params();
    let response = match input.transport {
        Transport::Http { client, .. } => {
            dispatch_http(client, input, worker.as_ref(), &split.attempt, params).await
        }
        Transport::Grpc {
            frontend, prepared, ..
        } => {
            dispatch_grpc(
                frontend,
                prepared.clone(),
                input.path,
                worker.as_ref(),
                &split.attempt,
                params,
            )
            .await
        }
    };
    match response {
        Some(response) => BypassAttempt::Respond(split.attempt.track_response(response)),
        None => {
            split.attempt.release();
            BypassAttempt::Refused
        }
    }
}

fn load_allows(
    feature: &PrefillBypass,
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
    let decode_idle = decode_load.active_decode_requests == 0
        && decode_load.pending_decode_requests == own_decode_reservations;
    let other_prefills = if feature.needs_prefill_load() {
        loads
            .get(&(Stage::Prefill, prefill.clone()))
            .and_then(|load| load.prefill_requests.checked_sub(1))
    } else {
        None
    };
    feature.allows(estimate, decode_idle, other_prefills)
}

async fn prepare_estimate(
    state: &PdKvState,
    worker: &dyn Worker,
    policy: &dyn LoadBalancingPolicy,
    request: &Value,
    route: &str,
    headers: Option<&HeaderMap>,
    router_prompts: Option<&Arc<[Vec<u32>]>>,
) -> Option<(PrefillEstimate, u64)> {
    let worker_revision = state.local_prefill_revision(worker)?;
    let policy = kv_aware_ops(policy)?;
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
    (state.local_prefill_revision(worker) == Some(worker_revision)).then_some((
        PrefillEstimate {
            prompt_tokens: prompt.len(),
            remaining_tokens,
        },
        worker_revision,
    ))
}

async fn bypass_target(
    bypass: &NativeBypass<'_>,
    input: BeforePrefill<'_>,
    split: &SplitDispatch,
) -> Option<Arc<dyn Worker>> {
    let BeforePrefill {
        request,
        path,
        headers,
        ..
    } = input;
    let SplitDispatch {
        prefill,
        decode,
        attempt,
        commit,
    } = split;
    if !PrefillBypass::eligible_request(path, request)
        || !bypass.admission.supports_prefill(prefill.as_ref())
    {
        return None;
    }
    let NativeBypass {
        feature,
        state,
        admission,
    } = bypass;
    let policy = admission.decode_policy();
    let (estimate, worker_revision) = prepare_estimate(
        state,
        decode.as_ref(),
        policy.as_ref(),
        request,
        path,
        headers,
        input.router_prompts(),
    )
    .await?;
    if !feature.within_limit(estimate) {
        return None;
    }
    if let Some(commit) = commit {
        let accept = |loads: &Loads, worker: &dyn Worker| {
            state.local_prefill_revision(worker) == Some(worker_revision)
                && load_allows(
                    feature,
                    loads,
                    attempt.prefill_endpoint(),
                    &commit.provisional_endpoint,
                    0,
                    estimate,
                )
        };
        return admission
            .commit_provisional_decode(attempt, commit, request, headers, &accept)
            .await;
    }
    let pool = input.pool();
    let endpoint = PdEndpoint::for_worker(decode.as_ref());
    let live = admission.eligible_decode_candidates(
        pool,
        request.get("model").and_then(Value::as_str),
        &endpoint,
    );
    let worker = live.into_iter().find(|worker| {
        (matches!(pool, DecodePoolSource::Discovery) || Arc::ptr_eq(worker, decode))
            && state.local_prefill_revision(worker.as_ref()) == Some(worker_revision)
    })?;
    attempt
        .check_reserved_decode(&endpoint, |loads| {
            load_allows(
                feature,
                loads,
                attempt.prefill_endpoint(),
                &endpoint,
                1,
                estimate,
            )
        })
        .then_some(worker)
}

async fn dispatch_grpc(
    frontend: &EngineFrontend,
    prepared: PreparedRequest,
    path: &str,
    worker: &dyn Worker,
    attempt: &PdAttempt,
    params: Value,
) -> Option<Response> {
    let start = Instant::now();
    attempt.finish_prefill();
    attempt.begin_decode();
    let response = frontend
        .dispatch_decode(
            worker.url(),
            prepared,
            Uuid::new_v4().to_string(),
            Some(params),
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

/// None is reserved for a typed refusal before generation.
async fn dispatch_http(
    client: &reqwest::Client,
    input: BeforePrefill<'_>,
    worker: &dyn Worker,
    attempt: &PdAttempt,
    params: Value,
) -> Option<Response> {
    let BeforePrefill {
        request,
        path,
        headers,
        ..
    } = input;
    let start = Instant::now();
    let url = worker.endpoint_url(path);
    let mut request = request.clone();
    request["kv_transfer_params"] = params;
    let builder = client
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
