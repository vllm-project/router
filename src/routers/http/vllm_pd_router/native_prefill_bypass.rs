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
    pd_router::select_decode_stage,
    pd_types::{error_chain, Loads, PdAttempt, PdEndpoint, PdLoadManager, PdRequestContext, Stage},
    prefill_bypass::{
        admit_with_bypass, grpc_rejected, http_rejected, BypassAttempt, PdPrefillBypassConfig,
        PrefillBypass, PrefillDispatch, PrefillEstimate, ERROR_HEADER,
    },
};
use super::{
    grpc_decode_outcome, own_grpc_decode, record_grpc_decode_stream_failure, response_with_body,
    Admission, AdmittedPair, DecodeCommitInput, DecodePoolSource, GrpcPdWorkers, VllmPDRouter,
    NAMED_LORA_FIELD,
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

type LiveDecode<'a> =
    dyn Fn(DecodePoolSource, Option<&str>, &PdEndpoint) -> Vec<Arc<dyn Worker>> + Sync + 'a;

struct NativeBypass<'a> {
    feature: &'a PrefillBypass,
    state: &'a PdKvState,
    load: &'a PdLoadManager,
    policy: Arc<dyn LoadBalancingPolicy>,
    supports_prefill: &'a (dyn Fn(&dyn Worker) -> bool + Sync),
    live_decode: &'a LiveDecode<'a>,
}

impl VllmPDRouter {
    async fn admit_before_prefill<F, Fut>(
        &self,
        input: BeforePrefill<'_>,
        mut admit: F,
    ) -> PrefillDispatch<SplitDispatch>
    where
        F: FnMut(bool) -> Fut,
        Fut: Future<Output = Result<AdmittedPair, Box<Response>>>,
    {
        let supports_prefill = |prefill: &dyn Worker| {
            self.get_nixl_push_identity(prefill.base_url(), prefill.dp_rank())
                .is_none()
        };
        let live_decode = |pool, model: Option<&str>, endpoint: &PdEndpoint| {
            let mut candidates = self.live_decode_candidates(pool);
            Self::retain_commit_candidates(&mut candidates, model, Some(endpoint));
            candidates
        };
        let bypass = self
            .prefill_bypass
            .as_ref()
            .zip(self.pd_router.kv_state.as_deref())
            .filter(|_| self.kv_connector == KvConnector::Nixl && self.epd.is_none())
            .map(|(feature, state)| NativeBypass {
                feature,
                state,
                load: &self.pd_router.pd_load,
                policy: self.policy_registry.get_decode_policy(),
                supports_prefill: &supports_prefill,
                live_decode: &live_decode,
            });
        admit_with_bypass(
            |readmit| {
                let admitted = admit(readmit);
                async move { admitted.await.map(|pair| SplitDispatch::new(pair, input)) }
            },
            |split| try_bypass(bypass, input, split),
        )
        .await
    }

    pub(super) async fn admit_http_for_dispatch(
        &self,
        request: &Value,
        path: &str,
        headers: Option<&HeaderMap>,
    ) -> PrefillDispatch<SplitDispatch> {
        let pool = if self.use_discovery {
            DecodePoolSource::Discovery
        } else {
            DecodePoolSource::Registry
        };
        self.admit_before_prefill(
            BeforePrefill {
                request,
                path,
                headers,
                pool,
                prompts: None,
                transport: Transport::Http(&self.pd_router.client),
            },
            |_| async {
                self.admit_http_pair(pool, request, path, headers)
                    .await
                    .map_err(|error| {
                        RouterMetrics::record_pd_error("server_selection");
                        Box::new((StatusCode::SERVICE_UNAVAILABLE, error).into_response())
                    })
            },
        )
        .await
    }

    pub(super) async fn admit_grpc_for_dispatch(
        &self,
        request: &Value,
        path: &str,
        headers: Option<&HeaderMap>,
        prepared: &PreparedRequest,
        workers: &GrpcPdWorkers,
    ) -> PrefillDispatch<SplitDispatch> {
        let prompts: Arc<[Vec<u32>]> = Arc::from([prepared.tokenized.token_ids.to_vec()]);
        let prompts = &prompts;
        self.admit_before_prefill(
            BeforePrefill {
                request,
                path,
                headers,
                pool: DecodePoolSource::Registry,
                prompts: Some(prompts),
                transport: Transport::Grpc(&self.grpc_frontend, prepared),
            },
            |readmit| async move {
                let refreshed = if readmit {
                    Some(self.available_grpc_workers()?)
                } else {
                    None
                };
                self.admit_grpc_pair(
                    refreshed.as_ref().unwrap_or(workers),
                    request,
                    path,
                    headers,
                    prompts,
                )
                .await
            },
        )
        .await
    }
}

#[derive(Clone, Copy)]
enum Transport<'a> {
    Http(&'a reqwest::Client),
    Grpc(&'a EngineFrontend, &'a PreparedRequest),
}

#[derive(Clone, Copy)]
struct BeforePrefill<'a> {
    request: &'a Value,
    path: &'a str,
    headers: Option<&'a HeaderMap>,
    pool: DecodePoolSource,
    prompts: Option<&'a Arc<[Vec<u32>]>>,
    transport: Transport<'a>,
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
                input.pool,
                selection.decode.endpoint,
                decode_tokens,
                affinity_key,
                input.request,
                input.path,
                input.prompts.cloned(),
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
        Transport::Http(client) => {
            dispatch_http(client, input, worker.as_ref(), &split.attempt, params).await
        }
        Transport::Grpc(frontend, prepared) => {
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
        || !(bypass.supports_prefill)(prefill.as_ref())
    {
        return None;
    }
    let (estimate, revision) = prepare_estimate(
        bypass.state,
        decode.as_ref(),
        bypass.policy.as_ref(),
        request,
        path,
        headers,
        input.prompts,
    )
    .await?;
    if !bypass.feature.within_limit(estimate) {
        return None;
    }

    let pool = input.pool;
    let endpoint = PdEndpoint::for_worker(decode.as_ref());
    let model = request.get("model").and_then(Value::as_str);
    let worker = (bypass.live_decode)(pool, model, &endpoint)
        .into_iter()
        .find(|worker| {
            matches!(pool, DecodePoolSource::Discovery) || Arc::ptr_eq(worker, decode)
        })?;
    let prefill_endpoint = PdEndpoint::for_worker(prefill.as_ref());
    let allows = |loads: &Loads, own_reservations| {
        bypass.state.local_prefill_revision(worker.as_ref()) == Some(revision)
            && load_allows(
                bypass.feature,
                loads,
                &prefill_endpoint,
                &endpoint,
                own_reservations,
                estimate,
            )
    };
    if let Some(commit) = commit {
        // There is one candidate: the rank whose local KV was checked. Reuse
        // stage selection for the endpoint eligibility and session binding.
        let context = PdRequestContext {
            worker_pairs: None,
            prefill_inputs: None,
            model_id: model,
            legacy_policy_input: None,
            prefill_cache: None,
            decode_cache: None,
            headers: None,
            affinity_key: commit.affinity_key.clone(),
            lora_name: request.get(NAMED_LORA_FIELD).and_then(Value::as_str),
            cache_salt: commit.cache_salt.as_deref(),
            prefill_pricing: None,
            decode_pricing: None,
            route: path,
            probe: false,
        };
        let decision = bypass.load.commit_decode(attempt, |loads| {
            if !allows(loads, 0) {
                return None;
            }
            select_decode_stage(
                loads,
                std::slice::from_ref(&worker),
                bypass.policy.as_ref(),
                &context,
            )
        })?;
        RouterMetrics::record_pd_decode_commit("same");
        if let Some(affinity) = decision.affinity_outcome {
            RouterMetrics::record_pd_affinity_decision("decode", affinity);
        }
        attempt.attach_decode_worker(worker.clone());
    } else if !attempt.check_reserved_decode(&endpoint, |loads| allows(loads, 1)) {
        return None;
    }
    Some(worker)
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
