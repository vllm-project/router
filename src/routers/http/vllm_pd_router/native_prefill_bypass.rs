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
    DecodeCommitInput, DecodePoolSource, PDRouterError, ServiceType, VllmPDRouter,
    NAMED_LORA_FIELD,
};
use crate::backend::PreparedRequest;
use crate::config::{KvConnector, PolicyConfig, RouterConfig};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::policies::{kv_aware_ops, LoadBalancingPolicy, PromptHashes};
use crate::routers::{header_utils, RouterTrait};

pub(crate) fn prepare_prefill_bypass(
    config: PdPrefillBypassConfig,
    router: &RouterConfig,
) -> Result<Option<PrefillBypass>, String> {
    let feature = PrefillBypass::from_config(config)?;
    if feature.is_some()
        && (!router.mode.is_pd_mode()
            || router.kv_connector != KvConnector::Nixl
            || !matches!(
                router.mode.get_decode_policy(&router.policy),
                PolicyConfig::KvAware { .. }
            ))
    {
        return Err("prefill bypass requires vLLM P/D with NIXL and kv_aware Decode".into());
    }
    Ok(feature)
}

pub(crate) fn install_prefill_bypass(
    router: &dyn RouterTrait,
    feature: Option<PrefillBypass>,
) -> Result<(), String> {
    let Some(feature) = feature else {
        return Ok(());
    };
    router
        .as_any()
        .downcast_ref::<VllmPDRouter>()
        .ok_or_else(|| "prefill bypass requires a vLLM P/D router".to_string())?
        .set_prefill_bypass(feature)
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

tokio::task_local! {
    // The split retry reuses native admission with bypass disabled only for that future.
    static SPLIT_RETRY: ();
}

impl VllmPDRouter {
    /// Install the optional policy before the router serves requests.
    pub fn configure_prefill_bypass(&self, config: PdPrefillBypassConfig) -> Result<(), String> {
        if let Some(feature) = PrefillBypass::from_config(config)? {
            self.set_prefill_bypass(feature)?;
        }
        Ok(())
    }

    fn set_prefill_bypass(&self, feature: PrefillBypass) -> Result<(), String> {
        self.prefill_bypass
            .set(feature)
            .map_err(|_| "prefill bypass is already configured".to_string())?;
        if self.use_discovery {
            if let Some(state) = &self.pd_router.kv_state {
                let registry = self.service_registry.clone();
                let connector = self.kv_connector;
                let dp_size = self.intra_node_data_parallel_size;
                state.track_discovered_decodes(move || {
                    Self::expand_discovery_candidates(
                        registry.get_instance_snapshots(ServiceType::Decode),
                        ServiceType::Decode,
                        connector,
                        dp_size,
                    )
                    .unwrap_or_default()
                });
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn process_native_http_pair(
        &self,
        request: Value,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        path: &str,
        headers: Option<&HeaderMap>,
        attempt: PdAttempt,
        commit: Option<&DecodeCommitInput>,
    ) -> Result<Response, PDRouterError> {
        let input = BeforePrefill {
            request: &request,
            path,
            headers,
            transport: Transport::Http,
        };
        let pair = SelectedPair {
            prefill: &prefill,
            decode: &decode,
            attempt: &attempt,
            commit,
        };
        if let Some(response) = self.try_prefill_bypass(input, pair).await {
            return Ok(response);
        }
        self.process_vllm_two_stage_request(
            request, prefill, decode, path, headers, attempt, None, commit,
        )
        .await
    }

    pub(super) async fn try_grpc_prefill_bypass(
        &self,
        request: &Value,
        path: &str,
        headers: Option<&HeaderMap>,
        prepared: &PreparedRequest,
        prompts: &Arc<[Vec<u32>]>,
        pair: SelectedPair<'_>,
    ) -> Option<Response> {
        self.try_prefill_bypass(
            BeforePrefill {
                request,
                path,
                headers,
                transport: Transport::Grpc { prepared, prompts },
            },
            pair,
        )
        .await
    }

    async fn try_prefill_bypass(
        &self,
        input: BeforePrefill<'_>,
        pair: SelectedPair<'_>,
    ) -> Option<Response> {
        let feature = self.prefill_bypass.get()?;
        let state = self.pd_router.kv_state.as_deref()?;
        let BeforePrefill {
            request,
            path,
            headers,
            transport,
        } = input;
        if SPLIT_RETRY.try_with(|_| ()).is_ok()
            || self.kv_connector != KvConnector::Nixl
            || self.epd.is_some()
            || !PrefillBypass::eligible_request(path, request)
            || self
                .get_nixl_push_identity(pair.prefill.base_url(), pair.prefill.dp_rank())
                .is_some()
        {
            return None;
        }

        let policy = self.policy_registry.get_decode_policy();
        let (estimate, revision) =
            prepare_estimate(state, pair.decode.as_ref(), policy.as_ref(), input).await?;
        if !feature.within_limit(estimate) {
            return None;
        }

        let pool = if self.use_discovery {
            DecodePoolSource::Discovery
        } else {
            DecodePoolSource::Registry
        };
        let model = request.get("model").and_then(Value::as_str);
        let endpoint = PdEndpoint::for_worker(pair.decode.as_ref());
        let mut candidates = self.live_decode_candidates(pool);
        Self::retain_commit_candidates(&mut candidates, model, Some(&endpoint));
        let worker = candidates
            .into_iter()
            .find(|worker| self.use_discovery || Arc::ptr_eq(worker, pair.decode))?;
        let prefill_endpoint = PdEndpoint::for_worker(pair.prefill.as_ref());
        let allows = |loads: &Loads, reservation| {
            state.local_prefill_revision(worker.as_ref()) == Some(revision)
                && load_allows(
                    feature,
                    loads,
                    &prefill_endpoint,
                    &endpoint,
                    reservation,
                    estimate,
                )
        };
        let attempt = pair.attempt;
        if let Some(commit) = pair.commit {
            commit_decode(
                &self.pd_router.pd_load,
                attempt,
                commit,
                &worker,
                policy.as_ref(),
                input,
                |loads| allows(loads, DecodeReservation::Deferred),
            )?;
        } else if !attempt
            .check_reserved_decode(&endpoint, |loads| allows(loads, DecodeReservation::Held))
        {
            return None;
        }

        let start = Instant::now();
        let params = feature.transfer_params();
        attempt.finish_prefill();
        attempt.begin_decode();
        let response = match transport {
            Transport::Http => {
                let url = worker.endpoint_url(path);
                let mut body = request.clone();
                body["kv_transfer_params"] = params;
                let response = self
                    .send_http_decode(
                        &url,
                        worker.dp_rank(),
                        &Uuid::new_v4().to_string(),
                        path,
                        headers,
                        &body,
                    )
                    .await;
                local_http_response(response, &url)
            }
            Transport::Grpc { prepared, .. } => {
                let response = self
                    .grpc_frontend
                    .dispatch_decode(
                        worker.url(),
                        prepared.clone(),
                        Uuid::new_v4().to_string(),
                        Some(params),
                    )
                    .await;
                if response
                    .extensions()
                    .get::<tonic::Status>()
                    .is_some_and(grpc_rejected)
                {
                    None
                } else {
                    if response.status().is_success() {
                        record_grpc_decode_stream_failure(
                            grpc_decode_outcome(&response),
                            worker.base_url(),
                        );
                    }
                    Some(own_grpc_decode(response, attempt))
                }
            }
        };
        let worker_url = worker.base_url();
        RouterMetrics::record_pd_decode_request(worker_url);
        let response = if let Some(response) = response {
            RouterMetrics::record_pd_request(path);
            RouterMetrics::record_pd_request_duration(path, start.elapsed());
            if response.status().is_success() {
                record_bypass("accepted");
            } else {
                RouterMetrics::record_pd_decode_error(worker_url);
                record_bypass("failed");
            }
            response
        } else {
            attempt.release();
            record_bypass("refused");
            match transport {
                Transport::Http => {
                    SPLIT_RETRY
                        .scope(
                            (),
                            self.route_transparent(headers, path, &Method::POST, request.clone()),
                        )
                        .await
                }
                Transport::Grpc { prepared, .. } => {
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
                    SPLIT_RETRY.scope((), retry).await
                }
            }
        };
        Some(response)
    }
}

fn record_bypass(result: &'static str) {
    metrics::counter!("vllm_router_pd_prefill_bypass_total", "result" => result).increment(1);
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
    input: BeforePrefill<'_>,
) -> Option<(PrefillEstimate, u64)> {
    let BeforePrefill {
        request,
        path,
        headers,
        transport,
    } = input;
    let prompts = match transport {
        Transport::Http => None,
        Transport::Grpc { prompts, .. } => Some(prompts),
    };
    let worker_revision = state.local_prefill_revision(worker)?;
    let policy = kv_aware_ops(policy)?;
    let bases = vec![worker.base_url().to_owned()];
    let verified = state.credit_by_base(&bases, request).await;
    if verified.get(worker.base_url()) != Some(&true) {
        return None;
    }
    let inputs = state.tokenize(bases, request, path, headers, prompts).await;
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

fn commit_decode(
    load: &PdLoadManager,
    attempt: &PdAttempt,
    commit: &DecodeCommitInput,
    worker: &Arc<dyn Worker>,
    policy: &dyn LoadBalancingPolicy,
    input: BeforePrefill<'_>,
    allows: impl FnOnce(&Loads) -> bool,
) -> Option<()> {
    // Only the rank whose local KV was checked can take this commitment.
    let context = PdRequestContext {
        worker_pairs: None,
        prefill_inputs: None,
        model_id: commit.model_id.as_deref(),
        legacy_policy_input: None,
        prefill_cache: None,
        decode_cache: None,
        headers: None,
        affinity_key: commit.affinity_key.clone(),
        lora_name: input.request.get(NAMED_LORA_FIELD).and_then(Value::as_str),
        cache_salt: commit.cache_salt.as_deref(),
        prefill_pricing: None,
        decode_pricing: None,
        route: input.path,
        probe: false,
    };
    let decision = load.commit_decode(attempt, |loads| {
        if !allows(loads) {
            return None;
        }
        select_decode_stage(loads, std::slice::from_ref(worker), policy, &context)
    })?;
    RouterMetrics::record_pd_decode_commit("same");
    if let Some(affinity) = decision.affinity_outcome {
        RouterMetrics::record_pd_affinity_decision("decode", affinity);
    }
    attempt.attach_decode_worker(worker.clone());
    Some(())
}

/// None is reserved for a typed refusal before generation.
fn local_http_response(result: reqwest::Result<reqwest::Response>, url: &str) -> Option<Response> {
    if result
        .as_ref()
        .is_ok_and(|response| http_rejected(response.status(), response.headers()))
    {
        return None;
    }
    Some(match result {
        Ok(response) => {
            let status = response.status();
            let mut headers = header_utils::preserve_response_headers(response.headers());
            headers.remove(ERROR_HEADER);
            response_with_body(status, &headers, Body::from_stream(response.bytes_stream()))
                .unwrap_or_else(|error| (StatusCode::BAD_GATEWAY, error).into_response())
        }
        Err(error) => {
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
        let bypass = PrefillBypass::from_config(PdPrefillBypassConfig {
            enabled: true,
            ..Default::default()
        })
        .unwrap()
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
