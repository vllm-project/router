//! P/D prefill KV discovery, refresh, and request-input preparation.

use super::pd_types::PdPrefillInput;
use crate::backend::preprocess::{
    http_cache_read_fallback, prepare_http_token_inputs, router_token_inputs,
};
use crate::core::Worker;
use crate::kv_index::discovery::CONTROL_TIMEOUT;
use crate::kv_index::{HashMode, KvIndexSupervisor};
use crate::policies::{kv_aware_ops, KvAwareOps, KvAwarePolicy, LoadBalancingPolicy};
use axum::http::HeaderMap;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

// Pool changes are noticed off the request path; the source supervisor
// owns descriptor cadence, qualification deadlines and recovery.
const REFRESH_TICK: Duration = Duration::from_secs(1);

/// The most recent P pool and effective policy seen on the request path.
/// The background refresh loop binds from it; recording it is a few Arc
/// clones, so requests never wait for control-plane probes.
#[derive(Clone, Default)]
struct RefreshInputs {
    workers: Arc<[Arc<dyn Worker>]>,
    policy: Option<Arc<dyn LoadBalancingPolicy>>,
}

pub(super) struct PdKvState {
    client: reqwest::Client,
    api_key: Option<String>,
    supervisor: KvIndexSupervisor,
    targets: Mutex<HashMap<String, u64>>,
    // The background refresh loop's latest view of the P pool and policy.
    refresh_inputs: Mutex<RefreshInputs>,
}

impl std::fmt::Debug for PdKvState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PdKvState")
            .field("supervisor", &self.supervisor)
            .finish()
    }
}

impl PdKvState {
    pub(super) async fn start(
        client: reqwest::Client,
        api_key: Option<String>,
        sources: HashMap<String, crate::config::types::PdKvEventSources>,
        workers: Vec<Arc<dyn Worker>>,
        policy: Arc<dyn LoadBalancingPolicy>,
    ) -> Arc<Self> {
        let mut state = Self::new_with_statics(client, sources);
        state.api_key = api_key;
        state.supervisor.set_api_key(state.api_key.clone());
        state.refresh_inputs = Mutex::new(RefreshInputs {
            workers: workers.iter().cloned().collect(),
            policy: Some(policy.clone()),
        });
        let state = Arc::new(state);
        // Bind during startup; unavailable metadata falls back to load until refresh.
        if let Some(kv) = kv_aware_ops(policy.as_ref()) {
            if !workers.is_empty()
                && tokio::time::timeout(CONTROL_TIMEOUT, state.bind(&workers, kv))
                    .await
                    .is_err()
            {
                tracing::warn!(
                    "Initial PD KV binding timed out; using no-credit fallback until refresh"
                );
            }
        }
        state.spawn_refresh();
        state
    }

    /// Test-only constructor; production always passes the configured
    /// `pd_kv_event_sources` via `new_with_statics`.
    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn new(client: reqwest::Client) -> Self {
        Self::new_with_statics(client, HashMap::new())
    }

    fn new_with_statics(
        client: reqwest::Client,
        static_sources: HashMap<String, crate::config::types::PdKvEventSources>,
    ) -> Self {
        let (supervisor, signals) =
            KvIndexSupervisor::new_pd(client.clone(), HashMode::Sha256, static_sources);
        drop(signals);
        Self {
            client,
            supervisor,
            api_key: None,
            targets: Mutex::new(HashMap::new()),
            refresh_inputs: Mutex::new(RefreshInputs::default()),
        }
    }

    /// Reconcile bindings off the request path. Requests only record their
    /// pool and policy; they never wait for control-plane probes, so one
    /// hung P endpoint can no longer delay admission. Until the first
    /// successful bind, requests see no binding and route with no KV credit.
    fn spawn_refresh(self: &Arc<Self>) {
        let weak = Arc::downgrade(self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(REFRESH_TICK);
            loop {
                interval.tick().await;
                let Some(state) = weak.upgrade() else { break };
                let inputs = state.refresh_inputs.lock().await.clone();
                let Some(policy) = inputs.policy else {
                    continue;
                };
                let Some(kv) = kv_aware_ops(policy.as_ref()) else {
                    continue;
                };
                state.bind(&inputs.workers, kv).await;
            }
        });
    }

    /// Refresh source-owned registrations, then attach the current pool to policy.
    async fn bind(&self, workers: &[Arc<dyn Worker>], policy: &dyn KvAwareOps) {
        // The request path never reads targets. Keep pool projection ordered
        // across source refresh without another route-owned lifecycle lock.
        let mut targets = self.targets.lock().await;
        let bases = workers
            .iter()
            .map(|worker| worker.base_url().to_owned())
            .collect();
        let registrations = self.supervisor.refresh_workers(&bases).await;
        targets.retain(|target, _| {
            if workers.iter().any(|worker| worker.url() == target) {
                return true;
            }
            policy.unregister_worker(target);
            false
        });
        for worker in workers {
            let registration = registrations.get(worker.base_url());
            let revision = registration.map_or(0, |registration| registration.revision);
            if targets.get(worker.url()) == Some(&revision) {
                continue;
            }
            policy.unregister_worker(worker.url());
            let info = registration.and_then(|registration| registration.info.as_ref());
            let provider = info
                .filter(|info| info.ranks.contains(&(worker.dp_rank().unwrap_or(0) as u32)))
                .and_then(|_| registration.and_then(|registration| registration.provider.clone()));
            // A pool member with no observations remains registered for load-only fallback.
            policy.register_worker(worker.as_ref(), info, provider);
            targets.insert(worker.url().to_owned(), revision);
        }
    }

    /// `router_prompts` are the router's own token ids for `grpc://` pools;
    /// HTTP pools pass `None` and render through each backend.
    pub(super) async fn prepare(
        &self,
        workers: &[Arc<dyn Worker>],
        policy: &Arc<dyn LoadBalancingPolicy>,
        request: &Value,
        route: &str,
        headers: Option<&HeaderMap>,
        router_prompts: Option<&Arc<[Vec<u32>]>>,
    ) -> HashMap<String, PdPrefillInput> {
        let started = Instant::now();
        *self.refresh_inputs.lock().await = RefreshInputs {
            workers: workers.iter().cloned().collect(),
            policy: Some(policy.clone()),
        };
        let bases: Vec<_> = workers
            .iter()
            .map(|worker| worker.base_url().to_owned())
            .collect();
        let allow_credit = self
            .supervisor
            .worker_model_eligibility(&bases, request.get("model").and_then(Value::as_str));
        let results = match router_prompts {
            Some(prompts) => router_token_inputs(bases, request, prompts),
            None => {
                prepare_http_token_inputs(
                    &self.client,
                    bases,
                    request,
                    route,
                    headers,
                    self.api_key.as_deref(),
                    Duration::from_secs(1),
                )
                .await
            }
        };
        let mut prepared = HashMap::new();
        for (base, result) in results {
            let fallback = http_cache_read_fallback(request).or_else(|| {
                KvAwarePolicy::credit_fallback(
                    request.get("cache_salt").and_then(Value::as_str),
                    allow_credit.get(&base).copied().unwrap_or(false),
                )
            });
            let (prompts, fallback_reason, source) = match result {
                Ok(input) => (Some(input.prompts), fallback, input.source),
                Err(reason) => (None, Some(reason), reason),
            };
            crate::metrics::RouterMetrics::record_kv_routing_input("pd", source);
            let input = PdPrefillInput {
                prompts,
                work_by_rank: HashMap::new(),
                allow_credit: fallback_reason.is_none(),
                fallback_reason,
            };
            metrics::gauge!("vllm_router_pd_identity_verified", "worker" => base.clone())
                .set(if input.allow_credit { 1.0 } else { 0.0 });
            prepared.insert(base, input);
        }
        if let Some(kv) = kv_aware_ops(policy.as_ref()) {
            let inputs: Vec<_> = workers
                .iter()
                .filter_map(|worker| {
                    let input = prepared.get(worker.base_url())?;
                    Some((
                        worker.as_ref(),
                        input.prompts.as_deref()?,
                        input.allow_credit,
                    ))
                })
                .collect();
            let work: Vec<_> = inputs
                .iter()
                .zip(kv.estimate_prefill_work(&inputs))
                .map(|((worker, _, _), estimate)| {
                    (worker.base_url().to_owned(), worker.dp_rank(), estimate)
                })
                .collect();
            for (base, rank, work) in work {
                prepared
                    .get_mut(&base)
                    .unwrap()
                    .work_by_rank
                    .insert(rank, work);
            }
        }
        metrics::histogram!("vllm_router_pd_kv_prepare_seconds")
            .record(started.elapsed().as_secs_f64());
        prepared
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod kv_input_tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};
    use crate::policies::{kv_aware_ops, KvAwarePolicy};
    use axum::extract::Request;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use reqwest::Client;
    use serde_json::json;

    #[tokio::test]
    async fn prompt_scoring_keeps_prefill_tokens_but_disables_pd_credit() {
        let state = PdKvState::new(Client::new());
        let policy: Arc<dyn LoadBalancingPolicy> = Arc::new(KvAwarePolicy::default());
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "http://worker".into(),
            WorkerType::Regular,
        ));
        let prompt: Vec<u32> = (0..65).collect();
        for (field, expected) in [
            (
                json!({"skip_reading_prefix_cache":true}),
                "skip_prefix_cache",
            ),
            (json!({"prompt_logprobs":0}), "unsupported_prompt_scoring"),
            (
                json!({"skip_reading_prefix_cache":false,"prompt_logprob_start":32}),
                "unsupported_prompt_scoring",
            ),
        ] {
            let mut body = json!({"model":"m","prompt":prompt});
            body.as_object_mut()
                .unwrap()
                .extend(field.as_object().unwrap().clone());
            let result = state
                .prepare(
                    std::slice::from_ref(&worker),
                    &policy,
                    &body,
                    "/v1/completions",
                    None,
                    None,
                )
                .await;
            let input = &result[worker.base_url()];
            assert_eq!(input.fallback_reason, Some(expected));
            assert!(!input.allow_credit);
            assert_eq!(input.prompts.as_ref().unwrap()[0].len(), 65);
            assert_eq!(input.work_by_rank[&worker.dp_rank()].new_prefill_tokens, 65);
        }
    }

    #[tokio::test]
    async fn unsupported_inputs_fall_back_without_control_or_tokenization_requests() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = calls.clone();
        let app = axum::Router::new().fallback(move || {
            let calls = observed.clone();
            async move {
                calls.fetch_add(1, Ordering::Relaxed);
                StatusCode::INTERNAL_SERVER_ERROR
            }
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let state = PdKvState::new(Client::new());
        let policy: Arc<dyn LoadBalancingPolicy> = Arc::new(KvAwarePolicy::default());
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(BasicWorker::new(
            base.clone(),
            WorkerType::Regular,
        ))];
        for (route, request) in [
            (
                "/v1/completions",
                json!({"model":"m","prompt_embeds":[[0.1,0.2]]}),
            ),
            (
                "/v1/chat/completions",
                json!({"model":"m","messages":[{"role":"user","content":"hello"}],"lora_name":"unsupported"}),
            ),
            (
                "/v1/responses",
                json!({"model":"m","input":"hello","previous_response_id":"resp_1"}),
            ),
        ] {
            let prepared = state
                .prepare(&workers, &policy, &request, route, None, None)
                .await;
            let input = &prepared[&base];
            assert!(input.prompts.is_none());
            assert!(input.work_by_rank.is_empty());
            assert!(input.fallback_reason.is_some());
        }
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        task.abort();
    }

    #[tokio::test]
    async fn http_preparation_uses_backend_tokens_and_forwards_auth() {
        use axum::{
            routing::{get, post},
            Json, Router,
        };
        for (configured_key, forwarded_auth, expected_auth) in [
            (None, None, None),
            (None, Some("Bearer caller"), Some("Bearer caller")),
            (
                Some("backend-key"),
                Some("Bearer caller"),
                Some("Bearer backend-key"),
            ),
        ] {
            let app = Router::new()
                .route(
                    "/get_server_info",
                    get(|| async {
                        Json(json!({"model_id":"m","block_size":16,"instance_id":"p"}))
                    }),
                )
                .route("/kv_event_sources", get(|| async { Json(json!({})) }))
                .route(
                    "/v1/models",
                    get(|| async { Json(json!({"data":[{"id":"m","parent":null}]})) }),
                )
                .route(
                    "/tokenize",
                    post(
                        move |headers: HeaderMap, Json(body): Json<Value>| async move {
                            assert_eq!(
                                headers
                                    .get("authorization")
                                    .map(|value| value.to_str().unwrap()),
                                expected_auth
                            );
                            assert_eq!(body["prompt"], "hello");
                            Json(json!({"tokens":[7,8,9],"count":3,"max_model_len":128}))
                        },
                    ),
                );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let base = format!("http://{}", listener.local_addr().unwrap());
            let task = tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            });
            let mut state = PdKvState::new(reqwest::Client::new());
            state.api_key = configured_key.map(str::to_owned);
            let policy: Arc<dyn LoadBalancingPolicy> = Arc::new(KvAwarePolicy::default());
            let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(BasicWorker::new(
                base.clone(),
                WorkerType::Regular,
            ))];
            let mut headers = HeaderMap::new();
            if let Some(auth) = forwarded_auth {
                headers.insert("authorization", auth.parse().unwrap());
            }
            state
                .bind(&workers, kv_aware_ops(policy.as_ref()).unwrap())
                .await;
            let inputs = state
                .prepare(
                    &workers,
                    &policy,
                    &json!({"model":"m","prompt":"hello"}),
                    "/v1/completions",
                    Some(&headers),
                    None,
                )
                .await;
            assert_eq!(
                inputs[&base].prompts.as_deref(),
                Some([vec![7, 8, 9]].as_slice())
            );
            let work = kv_aware_ops(policy.as_ref())
                .unwrap()
                .estimate_prefill_work(&[(workers[0].as_ref(), &[vec![7, 8, 9]], true)])[0];
            assert_eq!(work.new_prefill_tokens, 3);
            assert!(!work.placement_verified); // No event source: exact tokens, zero fabricated credit.
            task.abort();
        }
    }

    #[tokio::test]
    async fn discovered_and_static_sources_enable_real_prefix_credit() {
        use axum::{routing::get, Json, Router};
        use std::sync::atomic::{AtomicUsize, Ordering};
        for static_source in [false, true] {
            let probes = Arc::new(AtomicUsize::new(0));
            let info_probes = probes.clone();
            let source_probes = probes.clone();
            let publisher_context = zmq::Context::new();
            let publisher = publisher_context.socket(zmq::PUB).unwrap();
            publisher.bind("tcp://127.0.0.1:*").unwrap();
            let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
            let source_metadata = Arc::new(parking_lot::Mutex::new(json!({"0": {
                "publisher": "zmq", "endpoint": endpoint, "enable_kv_cache_events": true
            }})));
            let mut app = Router::new().route(
                "/v1/models",
                get(|| async { Json(json!({"data":[{"id":"m", "parent":null}]})) }),
            );
            if !static_source {
                app = app.route("/server_info", get(move || {
            let probes = info_probes.clone();
            async move { probes.fetch_add(1, Ordering::Relaxed);
                Json(json!({"vllm_config":{"model_config":{"model":"engine-model"},"cache_config":{"block_size":16}}}))
            }}))
            .route("/kv_event_sources", get({ let sources = source_metadata.clone(); move || {
                let sources = sources.lock().clone();
                let probes = source_probes.clone();
                async move { probes.fetch_add(1, Ordering::Relaxed); Json(sources) }
            }}));
            }
            let app = app.layer(axum::middleware::from_fn(
                |request: Request, next: axum::middleware::Next| async move {
                    if request
                        .headers()
                        .get("authorization")
                        .and_then(|value| value.to_str().ok())
                        != Some("Bearer pd-test-key")
                    {
                        return StatusCode::UNAUTHORIZED.into_response();
                    }
                    next.run(request).await
                },
            ));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let base = format!("http://{}", listener.local_addr().unwrap());
            let task = tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            });
            let sources = if static_source {
                std::iter::once((
                    base.clone(),
                    crate::config::types::PdKvEventSources {
                        endpoints: vec![endpoint],
                        replay_endpoints: None,
                        block_size: 16,
                        model: None,
                        hybrid: None,
                        compatible_decode_urls: Vec::new(),
                    },
                ))
                .collect()
            } else {
                HashMap::new()
            };
            let mut state = PdKvState::new_with_statics(reqwest::Client::new(), sources);
            state.api_key = Some("pd-test-key".to_owned());
            state.supervisor.set_api_key(state.api_key.clone());
            let policy: Arc<dyn LoadBalancingPolicy> = Arc::new(KvAwarePolicy::default());
            let ops = kv_aware_ops(policy.as_ref()).unwrap();
            let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(BasicWorker::new(
                base.clone(),
                WorkerType::Regular,
            ))];
            let tokens: Vec<u32> = (0..33).collect();
            let request = json!({"model":"m", "prompt":tokens});
            state.bind(&workers, ops).await;
            assert_eq!(
                probes.load(Ordering::Relaxed),
                if static_source { 0 } else { 2 }
            );
            let initial = state
                .prepare(&workers, &policy, &request, "/v1/completions", None, None)
                .await;
            assert_eq!(initial[&base].work_by_rank[&None].new_prefill_tokens, 33);
            assert!(initial[&base].work_by_rank[&None].placement_verified);
            let payload = rmp_serde::to_vec(&(
                1.0,
                vec![json!({
                    "type":"BlockStored", "block_hashes":[1,2], "parent_block_hash":null,
                    "token_ids":(0..32).collect::<Vec<u32>>(), "block_size":16,
                    "lora_id":null, "lora_name":null, "medium":"GPU"
                })],
                0_u32,
            ))
            .unwrap();
            let seq = 0_u64.to_be_bytes();
            tokio::time::timeout(Duration::from_secs(3), async {
                loop {
                    publisher
                        .send_multipart([b"".as_slice(), seq.as_slice(), payload.as_slice()], 0)
                        .unwrap();
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    let work = ops.estimate_prefill_work(&[(
                        workers[0].as_ref(),
                        std::slice::from_ref(&tokens),
                        true,
                    )])[0];
                    if work.new_prefill_tokens == 1 {
                        break;
                    }
                }
            })
            .await
            .expect("subscriber should ingest the warm prefix");
            let warm = state
                .prepare(&workers, &policy, &request, "/v1/completions", None, None)
                .await;
            assert_eq!(warm[&base].work_by_rank[&None].new_prefill_tokens, 1);
            assert!(warm[&base].work_by_rank[&None].placement_verified);
            drop(state);
            task.abort();
        }
    }
}
