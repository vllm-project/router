//! Native P/D adapter for optional Prefill bypass.

use std::future::Future;
use std::sync::Arc;
use std::time::Instant;

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
        grpc_rejected, http_rejected, PdPrefillBypassConfig, PrefillBypass, PrefillEstimate,
        ERROR_HEADER,
    },
};
use super::{
    grpc_decode_outcome, own_grpc_decode, record_grpc_decode_stream_failure, response_with_body,
    Admission, AdmittedPair, DecodeCommitInput, DecodePoolSource, GrpcPdWorkers, VllmPDRouter,
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

#[derive(Clone, Copy)]
enum Transport<'a> {
    Http(DecodePoolSource),
    Grpc {
        prepared: &'a PreparedRequest,
        prompts: &'a Arc<[Vec<u32>]>,
    },
}

#[derive(Clone, Copy)]
struct BeforePrefill<'a> {
    request: &'a Value,
    path: &'a str,
    headers: Option<&'a HeaderMap>,
    transport: Transport<'a>,
}

impl BeforePrefill<'_> {
    fn pool(&self) -> DecodePoolSource {
        match self.transport {
            Transport::Http(pool) => pool,
            Transport::Grpc { .. } => DecodePoolSource::Registry,
        }
    }

    fn router_prompts(&self) -> Option<&Arc<[Vec<u32>]>> {
        match self.transport {
            Transport::Http(_) => None,
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

pub(super) enum PrefillDispatch {
    Split(SplitDispatch),
    Respond(Response),
}

pub(super) async fn admit_http_for_dispatch(
    router: &VllmPDRouter,
    request: &Value,
    path: &str,
    headers: Option<&HeaderMap>,
) -> PrefillDispatch {
    let pool = if router.use_discovery {
        DecodePoolSource::Discovery
    } else {
        DecodePoolSource::Registry
    };
    let input = BeforePrefill {
        request,
        path,
        headers,
        transport: Transport::Http(pool),
    };
    admit_for_dispatch(router, input, |_| async {
        router
            .admit_http_pair(pool, request, path, headers)
            .await
            .map_err(|error| {
                RouterMetrics::record_pd_error("server_selection");
                Box::new((StatusCode::SERVICE_UNAVAILABLE, error).into_response())
            })
    })
    .await
}

pub(super) async fn admit_grpc_for_dispatch(
    router: &VllmPDRouter,
    request: &Value,
    path: &str,
    headers: Option<&HeaderMap>,
    prepared: &PreparedRequest,
    workers: &GrpcPdWorkers,
) -> PrefillDispatch {
    let prompts: Arc<[Vec<u32>]> = Arc::from([prepared.tokenized.token_ids.to_vec()]);
    let prompts = &prompts;
    let input = BeforePrefill {
        request,
        path,
        headers,
        transport: Transport::Grpc { prepared, prompts },
    };
    admit_for_dispatch(router, input, |readmit| async move {
        let refreshed = if readmit {
            Some(router.available_grpc_workers()?)
        } else {
            None
        };
        router
            .admit_grpc_pair(
                refreshed.as_ref().unwrap_or(workers),
                request,
                path,
                headers,
                prompts,
            )
            .await
    })
    .await
}

/// Admit a native pair and optionally dispatch straight to Decode. Only a typed
/// refusal permits fresh admission, and the second attempt always splits.
async fn admit_for_dispatch<F, Fut>(
    router: &VllmPDRouter,
    input: BeforePrefill<'_>,
    mut admit: F,
) -> PrefillDispatch
where
    F: FnMut(bool) -> Fut,
    Fut: Future<Output = Result<AdmittedPair, Box<Response>>>,
{
    let mut readmit = false;
    loop {
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
        } = match admit(readmit).await {
            Ok(pair) => pair,
            Err(response) => return PrefillDispatch::Respond(*response),
        };
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
        let split = SplitDispatch {
            prefill,
            decode,
            attempt,
            commit,
        };
        if let Some(feature) = router.prefill_bypass.as_ref().filter(|_| !readmit) {
            if let Some(worker) = bypass_target(router, feature, input, &split).await {
                let params = feature.transfer_params();
                let response = match input.transport {
                    Transport::Http(_) => {
                        dispatch_http(
                            &router.pd_router.client,
                            input,
                            worker.as_ref(),
                            &split.attempt,
                            params,
                        )
                        .await
                    }
                    Transport::Grpc { prepared, .. } => {
                        dispatch_grpc(
                            &router.grpc_frontend,
                            prepared.clone(),
                            input.path,
                            worker.as_ref(),
                            &split.attempt,
                            params,
                        )
                        .await
                    }
                };
                if let Some(response) = response {
                    return PrefillDispatch::Respond(split.attempt.track_response(response));
                }
                split.attempt.release();
                readmit = true;
                continue;
            }
        }
        return PrefillDispatch::Split(split);
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
    router: &VllmPDRouter,
    feature: &PrefillBypass,
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
    if !matches!(router.kv_connector, KvConnector::Nixl)
        || router.epd.is_some()
        || !PrefillBypass::eligible_request(path, request)
        || router
            .get_nixl_push_identity(prefill.base_url(), prefill.dp_rank())
            .is_some()
    {
        return None;
    }
    let state = router.pd_router.kv_state.as_ref()?;
    let policy = router.policy_registry.get_decode_policy();
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
        return router
            .select_and_commit_decode(attempt, commit, request, headers, true, Some(&accept))
            .await
            .ok();
    }
    let pool = input.pool();
    let endpoint = PdEndpoint::for_worker(decode.as_ref());
    let mut live = router.live_decode_candidates(pool);
    VllmPDRouter::retain_commit_candidates(
        &mut live,
        request.get("model").and_then(Value::as_str),
        Some(&endpoint),
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
