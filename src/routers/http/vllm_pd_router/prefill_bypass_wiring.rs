//! Assemble the native router dependencies for the Prefill bypass adapter.

use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
};
use serde_json::Value;

use super::native_prefill_bypass::{
    self, BeforePrefill, DecodeAdmission, NativeBypass, SplitDispatch, Transport,
};
use super::{DecodeCommitGuard, DecodeCommitInput, DecodePoolSource, GrpcPdWorkers, VllmPDRouter};
use crate::backend::PreparedRequest;
use crate::config::KvConnector;
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::policies::LoadBalancingPolicy;
use crate::routers::http::{
    pd_types::{PdAttempt, PdEndpoint},
    prefill_bypass::PrefillDispatch,
};

impl VllmPDRouter {
    fn native_prefill_bypass(&self) -> Option<NativeBypass<'_>> {
        Some(NativeBypass {
            feature: self.prefill_bypass.as_ref()?,
            state: self.pd_router.kv_state.as_deref()?,
            admission: self,
        })
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
        native_prefill_bypass::admit_for_dispatch(
            self.native_prefill_bypass(),
            BeforePrefill {
                request,
                path,
                headers,
                transport: Transport::Http {
                    client: &self.pd_router.client,
                    pool,
                },
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
        native_prefill_bypass::admit_for_dispatch(
            self.native_prefill_bypass(),
            BeforePrefill {
                request,
                path,
                headers,
                transport: Transport::Grpc {
                    frontend: &self.grpc_frontend,
                    prepared,
                    prompts,
                },
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

#[async_trait]
impl DecodeAdmission for VllmPDRouter {
    fn supports_prefill(&self, prefill: &dyn Worker) -> bool {
        self.kv_connector == KvConnector::Nixl
            && self.epd.is_none()
            && self
                .get_nixl_push_identity(prefill.base_url(), prefill.dp_rank())
                .is_none()
    }

    fn decode_policy(&self) -> Arc<dyn LoadBalancingPolicy> {
        self.policy_registry.get_decode_policy()
    }

    fn eligible_decode_candidates(
        &self,
        pool: DecodePoolSource,
        model: Option<&str>,
        endpoint: &PdEndpoint,
    ) -> Vec<Arc<dyn Worker>> {
        let mut candidates = self.live_decode_candidates(pool);
        Self::retain_commit_candidates(&mut candidates, model, Some(endpoint));
        candidates
    }

    async fn commit_provisional_decode(
        &self,
        attempt: &PdAttempt,
        commit: &DecodeCommitInput,
        request: &Value,
        headers: Option<&HeaderMap>,
        guard: &DecodeCommitGuard<'_>,
    ) -> Option<Arc<dyn Worker>> {
        self.select_and_commit_decode(attempt, commit, request, headers, true, Some(guard))
            .await
            .ok()
    }
}
