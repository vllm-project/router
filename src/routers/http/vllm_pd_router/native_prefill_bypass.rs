//! Native P/D adapter for optional Prefill bypass.

use std::sync::Arc;
use std::time::Instant;

use axum::{
    body::Body,
    http::{HeaderMap, Method, StatusCode},
    response::{IntoResponse, Response},
};
use serde_json::Value;
use uuid::Uuid;

use super::super::{
    dp_utils,
    pd_kv_state::PdKvState,
    pd_router::select_decode_stage,
    pd_types::{error_chain, Loads, PdAttempt, PdEndpoint, PdLoadManager, PdRequestContext, Stage},
};
use super::prefill_bypass::{
    grpc_rejected, http_rejected, PdPrefillBypassConfig, PrefillBypass, PrefillEstimate,
    ERROR_HEADER,
};
use super::{
    grpc_decode_outcome, own_grpc_decode, record_grpc_decode_stream_failure, response_with_body,
    DecodeCommitInput, DecodePoolSource, VllmPDRouter, NAMED_LORA_FIELD,
};
use crate::backend::{EngineFrontend, PreparedRequest};
use crate::config::{KvConnector, PolicyConfig, RouterConfig};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::otel_http::{self, ClientRequestOptions};
use crate::policies::{kv_aware_ops, LoadBalancingPolicy, PromptHashes};
use crate::routers::{header_utils, RouterTrait};

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

pub(crate) fn install_prefill_bypass(
    router: &dyn RouterTrait,
    config: PdPrefillBypassConfig,
) -> Result<(), String> {
    if !config.is_enabled() {
        return Ok(());
    }
    router
        .as_any()
        .downcast_ref::<VllmPDRouter>()
        .ok_or_else(|| "prefill bypass requires a vLLM P/D router".to_string())?
        .configure_prefill_bypass(config)
}

type LiveDecode<'a> =
    dyn Fn(DecodePoolSource, Option<&str>, &Arc<dyn Worker>) -> Option<Arc<dyn Worker>> + Sync + 'a;

struct NativeBypass<'a> {
    feature: &'a PrefillBypass,
    state: &'a PdKvState,
    load: &'a PdLoadManager,
    policy: Arc<dyn LoadBalancingPolicy>,
    live_decode: &'a LiveDecode<'a>,
}

pub(super) struct SelectedPair<'a> {
    pub(super) prefill: &'a Arc<dyn Worker>,
    pub(super) decode: &'a Arc<dyn Worker>,
    pub(super) attempt: &'a PdAttempt,
    pub(super) commit: Option<&'a DecodeCommitInput>,
}

#[derive(Clone, Copy)]
enum Transport<'a> {
    Http,
    Grpc(&'a PreparedRequest),
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

enum DirectDecode {
    Responded(Response),
    Refused,
}

tokio::task_local! {
    // The split retry reuses native admission with bypass disabled only for that future.
    static SPLIT_RETRY: ();
}

impl VllmPDRouter {
    /// Install the optional policy before the router serves requests.
    pub fn configure_prefill_bypass(&self, config: PdPrefillBypassConfig) -> Result<(), String> {
        config.validate()?;
        if let Some(feature) = PrefillBypass::enabled(config) {
            self.prefill_bypass
                .set(feature)
                .map_err(|_| "prefill bypass is already configured".to_string())?;
        }
        Ok(())
    }

    pub(super) async fn try_http_prefill_bypass(
        &self,
        request: &Value,
        path: &str,
        headers: Option<&HeaderMap>,
        pair: SelectedPair<'_>,
    ) -> Option<Response> {
        let input = BeforePrefill {
            request,
            path,
            headers,
            pool: if self.use_discovery {
                DecodePoolSource::Discovery
            } else {
                DecodePoolSource::Registry
            },
            prompts: None,
            transport: Transport::Http,
        };
        match self.try_prefill_bypass(input, pair).await? {
            DirectDecode::Responded(response) => Some(response),
            DirectDecode::Refused => {
                let retry = self.route_transparent(headers, path, &Method::POST, request.clone());
                Some(SPLIT_RETRY.scope((), retry).await)
            }
        }
    }

    pub(super) async fn try_grpc_prefill_bypass(
        &self,
        request: &Value,
        path: &str,
        headers: Option<&HeaderMap>,
        (prepared, prompts): (&PreparedRequest, &Arc<[Vec<u32>]>),
        pair: SelectedPair<'_>,
    ) -> Option<Response> {
        let input = BeforePrefill {
            request,
            path,
            headers,
            pool: DecodePoolSource::Registry,
            prompts: Some(prompts),
            transport: Transport::Grpc(prepared),
        };
        match self.try_prefill_bypass(input, pair).await? {
            DirectDecode::Responded(response) => Some(response),
            DirectDecode::Refused => {
                let workers = match self.available_grpc_workers() {
                    Ok(workers) => workers,
                    Err(response) => return Some(*response),
                };
                let retry = Box::pin(self.route_grpc_prepared(
                    headers,
                    request.clone(),
                    path,
                    prepared.clone(),
                    workers,
                ));
                Some(SPLIT_RETRY.scope((), retry).await)
            }
        }
    }

    async fn try_prefill_bypass(
        &self,
        input: BeforePrefill<'_>,
        pair: SelectedPair<'_>,
    ) -> Option<DirectDecode> {
        let feature = self.prefill_bypass.get()?;
        if SPLIT_RETRY.try_with(|_| ()).is_ok()
            || self.kv_connector != KvConnector::Nixl
            || self.epd.is_some()
            || !PrefillBypass::eligible_request(input.path, input.request)
            || self
                .get_nixl_push_identity(pair.prefill.base_url(), pair.prefill.dp_rank())
                .is_some()
        {
            return None;
        }
        let live_decode = |pool, model: Option<&str>, selected: &Arc<dyn Worker>| {
            let mut candidates = self.live_decode_candidates(pool);
            let endpoint = PdEndpoint::for_worker(selected.as_ref());
            Self::retain_commit_candidates(&mut candidates, model, Some(&endpoint));
            candidates.into_iter().find(|worker| {
                matches!(pool, DecodePoolSource::Discovery) || Arc::ptr_eq(worker, selected)
            })
        };
        let bypass = NativeBypass {
            feature,
            state: self.pd_router.kv_state.as_deref()?,
            load: &self.pd_router.pd_load,
            policy: self.policy_registry.get_decode_policy(),
            live_decode: &live_decode,
        };
        let worker = bypass_target(&bypass, input, &pair).await?;
        let params = feature.transfer_params();
        let response = match input.transport {
            Transport::Http => {
                dispatch_http(
                    &self.pd_router.client,
                    input,
                    worker.as_ref(),
                    pair.attempt,
                    params,
                )
                .await
            }
            Transport::Grpc(prepared) => {
                dispatch_grpc(
                    &self.grpc_frontend,
                    prepared.clone(),
                    input.path,
                    worker.as_ref(),
                    pair.attempt,
                    params,
                )
                .await
            }
        };
        let (result, outcome) = match response {
            Some(response) => {
                let result = if response.status().is_success() {
                    "accepted"
                } else {
                    "failed"
                };
                (
                    result,
                    DirectDecode::Responded(pair.attempt.clone().track_response(response)),
                )
            }
            None => {
                pair.attempt.release();
                ("refused", DirectDecode::Refused)
            }
        };
        metrics::counter!("vllm_router_pd_prefill_bypass_total", "result" => result).increment(1);
        Some(outcome)
    }
}

#[derive(Clone, Copy)]
enum DecodeReservation {
    Deferred,
    Held,
}

fn load_allows(
    feature: &PrefillBypass,
    loads: &Loads,
    prefill: &PdEndpoint,
    decode: &PdEndpoint,
    reservation: DecodeReservation,
    estimate: PrefillEstimate,
) -> bool {
    let decode_load = loads
        .get(&(Stage::Decode, decode.clone()))
        .copied()
        .unwrap_or_default();
    // A held reservation is this request's own pending slot.
    let own_pending = usize::from(matches!(reservation, DecodeReservation::Held));
    let decode_idle = decode_load.active_decode_requests == 0
        && decode_load.pending_decode_requests == own_pending;
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
    Some((
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
    pair: &SelectedPair<'_>,
) -> Option<Arc<dyn Worker>> {
    let BeforePrefill {
        request,
        path,
        headers,
        ..
    } = input;
    let SelectedPair {
        prefill,
        decode,
        attempt,
        commit,
    } = pair;
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

    let model = request.get("model").and_then(Value::as_str);
    let worker = (bypass.live_decode)(input.pool, model, decode)?;
    let endpoint = PdEndpoint::for_worker(worker.as_ref());
    let prefill_endpoint = PdEndpoint::for_worker(prefill.as_ref());
    let allows = |loads: &Loads, reservation| {
        bypass.state.local_prefill_revision(worker.as_ref()) == Some(revision)
            && load_allows(
                bypass.feature,
                loads,
                &prefill_endpoint,
                &endpoint,
                reservation,
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
            if !allows(loads, DecodeReservation::Deferred) {
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
    } else if !attempt
        .check_reserved_decode(&endpoint, |loads| allows(loads, DecodeReservation::Held))
    {
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

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::routers::http::pd_types::{PdSelection, StageDecision};

    #[tokio::test]
    async fn split_retry_marker_is_local_to_its_future() {
        let (entered, in_retry) = tokio::sync::oneshot::channel();
        tokio::join!(
            SPLIT_RETRY.scope((), async {
                assert!(SPLIT_RETRY.try_with(|_| ()).is_ok());
                entered.send(()).unwrap();
                tokio::task::yield_now().await;
                assert!(SPLIT_RETRY.try_with(|_| ()).is_ok());
            }),
            async {
                in_retry.await.unwrap();
                assert!(SPLIT_RETRY.try_with(|_| ()).is_err());
            },
        );
        assert!(SPLIT_RETRY.try_with(|_| ()).is_err());
    }

    #[test]
    fn bypass_rejects_other_pending_and_active_decode_requests() {
        let endpoint = PdEndpoint {
            http_url: "http://worker".into(),
            dp_rank: None,
        };
        let decision = StageDecision {
            fallback_reason: None,
            credited_prefix_tokens: None,
            worker_index: 0,
            endpoint: endpoint.clone(),
            score: None,
            new_prefill_tokens: Some(8),
            affinity_outcome: None,
            affinity: None,
        };
        let select = || {
            Some(PdSelection {
                prefill: decision.clone(),
                decode: decision.clone(),
            })
        };
        let bypass = PrefillBypass::enabled(PdPrefillBypassConfig {
            enabled: true,
            ..Default::default()
        })
        .unwrap();
        let estimate = PrefillEstimate {
            prompt_tokens: 8,
            remaining_tokens: 1,
        };

        for reservation in [DecodeReservation::Deferred, DecodeReservation::Held] {
            let ledger = PdLoadManager::default();
            let (_, attempt) = ledger
                .admit_with(|_| select(), matches!(reservation, DecodeReservation::Held))
                .unwrap();
            let allowed = || {
                let check = |loads: &Loads| {
                    load_allows(&bypass, loads, &endpoint, &endpoint, reservation, estimate)
                };
                match reservation {
                    DecodeReservation::Deferred => {
                        // Inspect the commit snapshot without adding a reservation.
                        let mut allowed = false;
                        ledger.commit_decode(&attempt, |loads| {
                            allowed = check(loads);
                            None
                        });
                        allowed
                    }
                    DecodeReservation::Held => attempt.check_reserved_decode(&endpoint, check),
                }
            };
            assert!(allowed());
            let (_, other) = ledger.admit(|_| select()).unwrap();
            assert!(!allowed(), "another pending Decode must block bypass");
            other.begin_decode();
            assert!(!allowed(), "another active Decode must block bypass");
            other.release();
            assert!(allowed());
        }
    }
}
