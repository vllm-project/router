//! P/D KV source registration and request-input preparation.

use super::pd_types::{PdEndpoint, PdPrefillInput};
use crate::backend::preprocess::{
    http_cache_read_fallback, http_input_requires_render, prepare_http_token_inputs,
    router_token_inputs,
};
use crate::config::KvConnector;
use crate::core::{Worker, WorkerRegistry};
use crate::kv_index::discovery::CONTROL_TIMEOUT;
use crate::kv_index::KvIndexSupervisor;
use crate::policies::{
    kv_aware_ops, KvAwareOps, KvAwarePolicy, LoadBalancingPolicy, PdWorkPrices, PromptHashes,
};
use axum::http::HeaderMap;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

// The supervisor owns the refresh and recovery deadlines for each source.
const REFRESH_TICK: Duration = Duration::from_secs(1);

type StagePool<'a> = (&'a [Arc<dyn Worker>], Option<&'a dyn KvAwareOps>);

#[derive(PartialEq, Eq)]
struct TargetRegistration {
    revision: u64,
    decode: bool,
}

pub(super) struct PdKvState {
    client: reqwest::Client,
    api_key: Option<String>,
    supervisor: KvIndexSupervisor,
    registry: Option<Arc<WorkerRegistry>>,
    prefill_policy: Option<Arc<dyn LoadBalancingPolicy>>,
    decode_policy: Option<Arc<dyn LoadBalancingPolicy>>,
    // Serializes updates to the target bindings in both policies.
    targets: Mutex<HashMap<String, TargetRegistration>>,
    worker_pairs: Option<HashMap<String, HashSet<String>>>,
    allow_hybrid_layouts: bool,
    render_images: bool,
    nixl_decode_replay: bool,
    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    credit_gate: Mutex<()>,
}

impl std::fmt::Debug for PdKvState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PdKvState")
            .field("supervisor", &self.supervisor)
            .finish()
    }
}

impl PdKvState {
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn start(
        client: reqwest::Client,
        api_key: Option<String>,
        sources: HashMap<String, crate::config::types::PdKvEventSources>,
        connector: KvConnector,
        pairs: HashMap<String, Vec<String>>,
        registry: Arc<WorkerRegistry>,
        prefill_policy: Option<Arc<dyn LoadBalancingPolicy>>,
        decode_policy: Option<Arc<dyn LoadBalancingPolicy>>,
        strict_recovery: bool,
        allow_hybrid_layouts: bool,
        render_images: bool,
    ) -> Arc<Self> {
        let mut state = Self::new_with_statics(client, sources);
        state.worker_pairs = (!pairs.is_empty()).then(|| {
            pairs
                .into_iter()
                .map(|(prefill, decodes)| (prefill, decodes.into_iter().collect()))
                .collect()
        });
        state.api_key = api_key;
        state.allow_hybrid_layouts = allow_hybrid_layouts;
        state.render_images = render_images;
        state.nixl_decode_replay = connector == KvConnector::Nixl;
        state.supervisor.set_api_key(state.api_key.clone());
        state.supervisor.set_strict_recovery(strict_recovery);
        state.supervisor.set_pd_connector(connector);
        state.registry = Some(registry);
        state.prefill_policy = prefill_policy;
        state.decode_policy = decode_policy;
        let state = Arc::new(state);
        if tokio::time::timeout(CONTROL_TIMEOUT, state.bind_refresh())
            .await
            .is_err()
        {
            tracing::warn!(
                "Initial PD KV binding timed out; using no-credit fallback until refresh"
            );
        }
        state.spawn_refresh();
        state
    }

    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn new(client: reqwest::Client) -> Self {
        Self::new_with_statics(client, HashMap::new())
    }

    /// Pause credit preparation so a commit test can remove a Decode worker.
    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(super) async fn lock_bindings_for_test(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.credit_gate.lock().await
    }

    fn new_with_statics(
        client: reqwest::Client,
        static_sources: HashMap<String, crate::config::types::PdKvEventSources>,
    ) -> Self {
        let (supervisor, signals) = KvIndexSupervisor::new_pd(client.clone(), static_sources);
        drop(signals);
        Self {
            client,
            api_key: None,
            supervisor,
            registry: None,
            prefill_policy: None,
            decode_policy: None,
            targets: Mutex::new(HashMap::new()),
            worker_pairs: None,
            allow_hybrid_layouts: true,
            render_images: true,
            nixl_decode_replay: false,
            #[cfg(all(test, feature = "cpu", feature = "ready"))]
            credit_gate: Mutex::new(()),
        }
    }

    /// Match Prefill workers with their configured Decode workers.
    pub(super) fn allowed_worker_pairs(
        &self,
        prefill: &[Arc<dyn Worker>],
        decode: &[Arc<dyn Worker>],
    ) -> Option<HashSet<(PdEndpoint, PdEndpoint)>> {
        let configured = self.worker_pairs.as_ref()?;
        let mut allowed = HashSet::new();
        for p in prefill {
            for d in decode {
                if configured
                    .get(p.base_url())
                    .is_some_and(|decodes| decodes.contains(d.base_url()))
                {
                    allowed.insert((
                        PdEndpoint::for_worker(p.as_ref()),
                        PdEndpoint::for_worker(d.as_ref()),
                    ));
                }
            }
        }
        Some(allowed)
    }

    /// Refresh both stages together so one stage cannot retire a source
    /// still used by the other.
    fn spawn_refresh(self: &Arc<Self>) {
        let weak = Arc::downgrade(self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(REFRESH_TICK);
            loop {
                interval.tick().await;
                let Some(state) = weak.upgrade() else { break };
                state.bind_refresh().await;
            }
        });
    }

    async fn bind_refresh(&self) {
        let Some(registry) = &self.registry else {
            return;
        };
        let prefill_policy = self.prefill_policy.as_deref().and_then(kv_aware_ops);
        let decode_policy = self.decode_policy.as_deref().and_then(kv_aware_ops);
        let prefill = registry.get_prefill_workers();
        let decode = registry.get_decode_workers();
        self.bind_pools([
            (prefill.as_slice(), prefill_policy),
            (decode.as_slice(), decode_policy),
        ])
        .await;
    }

    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    #[cfg_attr(coverage_nightly, coverage(off))]
    async fn bind(&self, workers: &[Arc<dyn Worker>], policy: &dyn KvAwareOps) {
        self.bind_pools([(workers, Some(policy)), (&[], None)])
            .await;
    }

    async fn bind_pools(&self, pools: [StagePool<'_>; 2]) {
        let bases: HashSet<String> = pools
            .iter()
            .filter(|(_, policy)| policy.is_some())
            .flat_map(|(workers, _)| workers.iter().map(|worker| worker.base_url().to_owned()))
            .collect();
        let (prefill, prefill_policy) = pools[0];
        let prefill_bases = prefill
            .iter()
            .filter(|_| prefill_policy.is_some())
            .map(|worker| worker.base_url().to_owned())
            .collect();
        let plain_only = if self.allow_hybrid_layouts {
            HashSet::new()
        } else {
            bases.clone()
        };
        let registrations = self
            .supervisor
            .refresh_workers(&bases, &plain_only, &prefill_bases)
            .await;

        let mut targets = self.targets.lock().await;
        // A source probe may outlive the worker it started for.
        let is_current = |worker: &Arc<dyn Worker>| {
            self.registry.as_ref().is_none_or(|registry| {
                registry
                    .get_by_url(worker.url())
                    .is_some_and(|current| Arc::ptr_eq(worker, &current))
            })
        };
        targets.retain(|url, target| {
            let (workers, policy) = pools[usize::from(target.decode)];
            if let Some(policy) = policy {
                let still_registered = match &self.registry {
                    Some(registry) => registry.get_by_url(url).is_some_and(|worker| {
                        if target.decode {
                            worker.worker_type() == crate::core::WorkerType::Decode
                        } else {
                            matches!(
                                worker.worker_type(),
                                crate::core::WorkerType::Prefill { .. }
                            )
                        }
                    }),
                    None => workers.iter().any(|worker| worker.url() == url),
                };
                if still_registered {
                    return true;
                }
                policy.unregister_worker(url);
            }
            false
        });
        for (stage, (workers, policy)) in pools.into_iter().enumerate() {
            let Some(policy) = policy else { continue };
            for worker in workers.iter().filter(|worker| is_current(worker)) {
                let registration = registrations.get(worker.base_url());
                let target = TargetRegistration {
                    revision: registration.map_or(0, |registration| registration.revision),
                    decode: stage == 1,
                };
                if targets.get(worker.url()) == Some(&target) {
                    continue;
                }
                policy.unregister_worker(worker.url());
                let info = registration.and_then(|registration| registration.info.as_ref());
                let provider = info
                    .filter(|info| info.ranks.contains(&(worker.dp_rank().unwrap_or(0) as u32)))
                    .and_then(|_| {
                        registration.and_then(|registration| registration.provider.clone())
                    });
                policy.register_worker(worker.as_ref(), info, provider);
                targets.insert(worker.url().to_owned(), target);
            }
        }
    }

    /// Each remaining Decode candidate needs both an eligible model and a
    /// registered index for its rank before any of them receive a KV price.
    fn all_priceable(
        &self,
        workers: &[Arc<dyn Worker>],
        request: &Value,
        allow_credit: &HashMap<String, bool>,
    ) -> bool {
        KvAwarePolicy::all_candidates_priceable(
            http_cache_read_fallback(request).is_none(),
            request.get("cache_salt").and_then(Value::as_str),
            workers.iter().map(|worker| {
                allow_credit
                    .get(worker.base_url())
                    .copied()
                    .unwrap_or(false)
            }),
        ) && self.supervisor.workers_have_cache_bindings(
            workers
                .iter()
                .map(|worker| (worker.base_url(), worker.dp_rank().unwrap_or(0) as u32)),
        )
    }

    pub(super) async fn unregister_worker(&self, url: &str, policy: &dyn KvAwareOps) {
        let mut targets = self.targets.lock().await;
        policy.unregister_worker(url);
        targets.remove(url);
        let base = super::dp_utils::parse_worker_url(url).0;
        let base_still_live = self.registry.as_ref().is_some_and(|registry| {
            registry
                .get_prefill_workers()
                .iter()
                .chain(registry.get_decode_workers().iter())
                .any(|worker| worker.base_url() == base)
        });
        if !base_still_live {
            self.supervisor.on_worker_removed(&base);
        }
    }

    async fn credit_by_base(&self, bases: &[String], request: &Value) -> HashMap<String, bool> {
        #[cfg(all(test, feature = "cpu", feature = "ready"))]
        let _wait = self.credit_gate.lock().await;
        self.supervisor
            .worker_model_eligibility(bases, request.get("model").and_then(Value::as_str))
    }

    /// Prepare both stages for one request.
    ///
    /// Render both pools concurrently, then share the local hash cache and credit decision.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn prepare_stages(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        prefill_policy: &Arc<dyn LoadBalancingPolicy>,
        decode_workers: &[Arc<dyn Worker>],
        decode_policy: &Arc<dyn LoadBalancingPolicy>,
        request: &Value,
        route: &str,
        headers: Option<&HeaderMap>,
        router_prompts: Option<&Arc<[Vec<u32>]>>,
        defer_decode: bool,
    ) -> (
        HashMap<String, PdPrefillInput>,
        Option<PdWorkPrices>,
        Option<PdWorkPrices>,
    ) {
        let started = Instant::now();
        let bases = |workers: &[Arc<dyn Worker>]| -> Vec<String> {
            workers
                .iter()
                .map(|worker| worker.base_url().to_owned())
                .collect()
        };
        let prefill_bases = bases(prefill_workers);
        let decode_bases = bases(decode_workers);
        let mut all_bases = prefill_bases.clone();
        all_bases.extend(decode_bases.iter().cloned());
        let allow_credit = self.credit_by_base(&all_bases, request).await;
        // Decide whether any Decode candidate can be credited *before* rendering: a salted
        // request, an unverified identity or a pool that publishes no events all end with
        // the rendered result discarded, and the render is the slow part.
        // A deferred decision only picks a provisional worker here and re-prices at commit,
        // so rendering and pricing Decode now would be discarded - and could stall admission
        // on a slow backend for nothing.
        // A candidate can only be priced when its base is creditable. The Decode
        // price table needs an estimate for every candidate, so pricing a partial set
        // always falls back - after paying for the slowest render. Require the whole set.
        // Decode pricing also never renders on non-gRPC pools: only gRPC router_prompts
        // and direct token-id prompts price without a `/render` fan-out.
        let decode_expected = (!defer_decode || self.worker_pairs.is_some())
            && kv_aware_ops(decode_policy.as_ref()).is_some();
        let decode_requires_render =
            router_prompts.is_none() && http_input_requires_render(request, route);
        let decode_priced_here = decode_expected
            && !decode_requires_render
            && self.all_priceable(decode_workers, request, &allow_credit);
        let (prefill_results, decode_results) = tokio::join!(
            self.tokenize(prefill_bases, request, route, headers, router_prompts),
            async {
                if decode_priced_here {
                    self.tokenize(decode_bases, request, route, headers, router_prompts)
                        .await
                } else {
                    HashMap::new()
                }
            },
        );
        let prepared = Self::build_prefill_inputs(request, &allow_credit, prefill_results);
        let mut hashes = PromptHashes::new();
        let prefill_prices = kv_aware_ops(prefill_policy.as_ref())
            .and_then(|kv| Self::prefill_prices(kv, prefill_workers, &prepared, &mut hashes));
        let decode_prices = if decode_priced_here {
            let kv = kv_aware_ops(decode_policy.as_ref()).expect("priced decode policy");
            self.decode_prices_from_results(
                kv,
                decode_workers,
                &allow_credit,
                &decode_results,
                &mut hashes,
            )
        } else {
            if decode_expected {
                crate::metrics::RouterMetrics::record_kv_placement_lookup("decode_unbound");
            }
            None
        };

        drop(hashes);
        metrics::histogram!("vllm_router_pd_kv_prepare_seconds")
            .record(started.elapsed().as_secs_f64());
        (prepared, prefill_prices, decode_prices)
    }

    /// Render the request against each base, or reuse the router's own ids.
    async fn tokenize(
        &self,
        bases: Vec<String>,
        request: &Value,
        route: &str,
        headers: Option<&HeaderMap>,
        router_prompts: Option<&Arc<[Vec<u32>]>>,
    ) -> HashMap<String, Result<crate::backend::preprocess::HttpTokenInput, &'static str>> {
        match router_prompts {
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
                    self.render_images,
                )
                .await
            }
        }
    }

    /// Pure: rendered prompts plus credit eligibility become `PdPrefillInput`s.
    fn build_prefill_inputs(
        request: &Value,
        allow_credit: &HashMap<String, bool>,
        results: HashMap<String, Result<crate::backend::preprocess::HttpTokenInput, &'static str>>,
    ) -> HashMap<String, PdPrefillInput> {
        let mut prepared = HashMap::with_capacity(results.len());
        for (base, result) in results {
            let fallback = http_cache_read_fallback(request).or_else(|| {
                KvAwarePolicy::credit_fallback(
                    request.get("cache_salt").and_then(Value::as_str),
                    allow_credit.get(&base).copied().unwrap_or(false),
                )
            });
            let (prompts, image, fallback_reason, source) = match result {
                Ok(input) => (Some(input.prompts), input.image, fallback, input.source),
                Err(reason) => (None, None, Some(reason), reason),
            };
            crate::metrics::RouterMetrics::record_kv_routing_input("pd", source);
            let input = PdPrefillInput {
                prompts,
                image,
                allow_credit: fallback_reason.is_none(),
                fallback_reason,
            };
            metrics::gauge!("vllm_router_pd_identity_verified", "worker" => base.clone())
                .set(if input.allow_credit { 1.0 } else { 0.0 });
            prepared.insert(base, input);
        }
        prepared
    }

    fn prefill_prices<'tokens>(
        kv: &dyn KvAwareOps,
        workers: &[Arc<dyn Worker>],
        prepared: &'tokens HashMap<String, PdPrefillInput>,
        hashes: &mut PromptHashes<'tokens>,
    ) -> Option<PdWorkPrices> {
        let inputs: Vec<_> = workers
            .iter()
            .filter_map(|worker| {
                let input = prepared.get(worker.base_url())?;
                let prompts = input.prompts.as_deref()?;
                if let ([tokens], Some(image)) = (prompts, input.image.as_ref()) {
                    hashes.register_image(tokens, image);
                }
                Some((worker.as_ref(), prompts, input.allow_credit))
            })
            .collect();
        kv.prepare_prefill_prices(&inputs, hashes)
    }

    /// Pure: the Decode estimate over already-rendered prompts.
    ///
    /// Returns `None` when any candidate is unpriced, because one credited
    /// figure beside a full-price one would compare different units across the
    /// pool.
    fn decode_prices_from_results<'tokens>(
        &self,
        kv: &dyn KvAwareOps,
        workers: &[Arc<dyn Worker>],
        allow_credit: &HashMap<String, bool>,
        results: &'tokens HashMap<
            String,
            Result<crate::backend::preprocess::HttpTokenInput, &'static str>,
        >,
        hashes: &mut PromptHashes<'tokens>,
    ) -> Option<PdWorkPrices> {
        let inputs: Vec<_> = workers
            .iter()
            .filter_map(|worker| {
                let result = results.get(worker.base_url())?;
                match result {
                    Ok(input) => Some((
                        worker.as_ref(),
                        input.prompts.as_ref(),
                        allow_credit
                            .get(worker.base_url())
                            .copied()
                            .unwrap_or(false),
                    )),
                    Err(reason) => {
                        crate::metrics::RouterMetrics::record_kv_routing_input("pd_decode", reason);
                        None
                    }
                }
            })
            .collect();
        if inputs.len() != workers.len() {
            crate::metrics::RouterMetrics::record_kv_placement_lookup("decode_unpriced");
            return None;
        }
        kv.prepare_decode_prices(&inputs, hashes, self.nixl_decode_replay)
    }

    /// Marginal Decode context per candidate, priced against the KV index.
    /// `hashes` is the caller's cache: the admission path shares it with the
    /// Prefill estimate, while the deferred commit starts a fresh one because it
    /// prices against the index as it stands at commit time.
    pub(super) async fn prepare_decode_prices(
        &self,
        workers: &[Arc<dyn Worker>],
        policy: &Arc<dyn LoadBalancingPolicy>,
        request: &Value,
        route: &str,
        headers: Option<&HeaderMap>,
        router_prompts: Option<&Arc<[Vec<u32>]>>,
    ) -> Option<PdWorkPrices> {
        // The only caller is the deferred commit, which prices against the index as it
        // stands now, so this owns its cache. The admission path shares one cache across both
        // stages through `prepare_stages`.
        let kv = kv_aware_ops(policy.as_ref())?;
        // Same no-render rule as admission: the commit path must not pay a
        // second fan-out for ids that cannot have changed.
        if router_prompts.is_none() && http_input_requires_render(request, route) {
            crate::metrics::RouterMetrics::record_kv_placement_lookup("decode_unbound");
            return None;
        }
        let bases: Vec<_> = workers
            .iter()
            .map(|worker| worker.base_url().to_owned())
            .collect();
        let allow_credit = self.credit_by_base(&bases, request).await;
        if !self.all_priceable(workers, request, &allow_credit) {
            crate::metrics::RouterMetrics::record_kv_placement_lookup("decode_unbound");
            return None;
        }
        let results = self
            .tokenize(bases, request, route, headers, router_prompts)
            .await;
        self.decode_prices_from_results(
            kv,
            workers,
            &allow_credit,
            &results,
            &mut PromptHashes::new(),
        )
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod kv_input_tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};
    use crate::policies::{kv_aware_ops, KvAwarePolicy, PromptHashes};
    use axum::extract::Request;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use reqwest::Client;
    use serde_json::json;

    /// Exercise the same preparation entry point as the request path. Decode is
    /// deferred so these prefill-only tests need no Decode pool.
    async fn prepare_prefill(
        state: &PdKvState,
        workers: &[Arc<dyn Worker>],
        policy: &Arc<dyn LoadBalancingPolicy>,
        request: &Value,
        route: &str,
        headers: Option<&HeaderMap>,
    ) -> (HashMap<String, PdPrefillInput>, Option<PdWorkPrices>) {
        let (inputs, prices, _) = state
            .prepare_stages(
                workers,
                policy,
                &[],
                policy,
                request,
                route,
                headers,
                None,
                true,
            )
            .await;
        (inputs, prices)
    }

    #[test]
    fn prefill_prices_the_rendered_image() {
        use crate::kv_index::request_hash::ImagePrefix;
        use crate::kv_index::source::test_support::{
            full_binding, warm_test_image, TEST_IMAGE_HASH,
        };
        use crate::kv_index::{CacheKey, RoutableTarget};

        let policy = KvAwarePolicy::default();
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "http://image-worker".into(),
            WorkerType::Regular,
        ));
        let base = worker.base_url().to_owned();
        let target = RoutableTarget {
            instance_id: Arc::from(base.as_str()),
            dp_rank: 0,
        };
        let binding = Arc::new(full_binding(target.clone(), 64));
        binding.begin_empty(0);
        warm_test_image(&binding);
        policy.register_target_with_cache_identity(
            &base,
            target,
            CacheKey {
                model: Arc::from("m"),
                block_size: 64,
            },
            binding,
        );
        let workers = [worker];
        let prepared = HashMap::from([(
            base,
            PdPrefillInput {
                prompts: Some(Arc::from([(0..129).collect()])),
                image: ImagePrefix::new(TEST_IMAGE_HASH, 60, 8, 129),
                allow_credit: true,
                fallback_reason: None,
            },
        )]);
        let prefill =
            PdKvState::prefill_prices(&policy, &workers, &prepared, &mut PromptHashes::default())
                .unwrap();
        let validated = prefill.validate(&workers);
        assert_eq!(
            validated
                .prefill_work(workers[0].url())
                .unwrap()
                .new_prefill_tokens,
            1
        );
    }

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
            let (result, prices) = prepare_prefill(
                &state,
                std::slice::from_ref(&worker),
                &policy,
                &body,
                "/v1/completions",
                None,
            )
            .await;
            let input = &result[worker.base_url()];
            assert_eq!(input.fallback_reason, Some(expected));
            assert!(!input.allow_credit);
            assert_eq!(input.prompts.as_ref().unwrap()[0].len(), 65);
            let prices = prices.unwrap();
            let work = prices.validate(std::slice::from_ref(&worker));
            assert_eq!(
                work.prefill_work(worker.url()).unwrap().new_prefill_tokens,
                65
            );
        }
    }

    #[tokio::test]
    async fn configured_worker_pairs_restrict_hybrid_and_preserve_full_workers() {
        let mut state = PdKvState::new(Client::new());
        state.worker_pairs = Some(HashMap::from([
            ("http://p0".into(), HashSet::from(["http://d0".into()])),
            ("http://p1".into(), HashSet::from(["http://d1".into()])),
        ]));
        let workers = |prefix: &str| -> Vec<Arc<dyn Worker>> {
            (0..2)
                .map(|i| {
                    Arc::new(BasicWorker::new(
                        format!("http://{prefix}{i}"),
                        WorkerType::Regular,
                    )) as Arc<dyn Worker>
                })
                .collect()
        };
        let p = workers("p");
        let d = workers("d");
        let pairs = state.allowed_worker_pairs(&p, &d).unwrap();
        assert_eq!(
            pairs,
            HashSet::from([
                (
                    PdEndpoint::for_worker(p[0].as_ref()),
                    PdEndpoint::for_worker(d[0].as_ref())
                ),
                (
                    PdEndpoint::for_worker(p[1].as_ref()),
                    PdEndpoint::for_worker(d[1].as_ref())
                )
            ])
        );
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
            let (prepared, prices) =
                prepare_prefill(&state, &workers, &policy, &request, route, None).await;
            let input = &prepared[&base];
            assert!(input.prompts.is_none());
            assert!(prices
                .unwrap()
                .validate(&workers)
                .prefill_work(workers[0].url())
                .is_none());
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
            let (inputs, _) = prepare_prefill(
                &state,
                &workers,
                &policy,
                &json!({"model":"m","prompt":"hello"}),
                "/v1/completions",
                Some(&headers),
            )
            .await;
            assert_eq!(
                inputs[&base].prompts.as_deref(),
                Some([vec![7, 8, 9]].as_slice())
            );
            let work = kv_aware_ops(policy.as_ref())
                .unwrap()
                .estimate_prefill_work(
                    &[(workers[0].as_ref(), &[vec![7, 8, 9]], true)],
                    &mut PromptHashes::new(),
                )[0];
            assert_eq!(work.new_prefill_tokens, 3);
            assert!(!work.placement_verified); // No event source: exact tokens, zero fabricated credit.
            task.abort();
        }
    }

    struct TestPublisher {
        live: zmq::Socket,
        replay: zmq::Socket,
        topic: &'static str,
    }

    impl TestPublisher {
        fn new(topic: &'static str) -> Self {
            let context = zmq::Context::new();
            let bound = |kind| {
                let socket = context.socket(kind).unwrap();
                socket.set_linger(0).unwrap();
                socket.bind("tcp://127.0.0.1:*").unwrap();
                socket
            };
            Self {
                live: bound(zmq::XPUB),
                replay: bound(zmq::ROUTER),
                topic,
            }
        }

        fn endpoints(&self) -> (String, String) {
            (
                self.live.get_last_endpoint().unwrap().unwrap(),
                self.replay.get_last_endpoint().unwrap().unwrap(),
            )
        }

        fn layout(&self, model: &str) -> Value {
            json!({
                "schema_version": 1,
                "adapters": [],
                "ranks": [{
                    "rank": 0, "model": model, "weight_version": "default",
                    "publisher_topic": self.topic, "instance_topic": true,
                    "hash_algorithm": "sha256_cbor", "full_hashes": true, "hash_seed": "0",
                    "hash_block_size": 16, "hash_lookahead_tokens": 0, "speculative_tokens": 0,
                    "prefix_caching": true, "declared_groups": 1, "transfer": null,
                    "groups": [{
                        "group_idx": 0, "kind": "full_attention", "spec_type": "FullAttentionSpec",
                        "block_size": 16, "prefix_cacheable": true, "prefix_replay_tokens": 0
                    }]
                }]
            })
        }

        fn static_source(&self, model: &str) -> crate::config::PdKvEventSources {
            let (live, replay) = self.endpoints();
            crate::config::PdKvEventSources {
                endpoints: vec![live],
                replay_endpoints: Some(vec![replay]),
                block_size: 16,
                model: Some(model.into()),
                hybrid: Some(serde_json::from_value(self.layout(model)).unwrap()),
            }
        }

        async fn replay_empty_and_publish(&self, events: &[crate::kv_index::KVEvent]) {
            async fn receive(socket: &zmq::Socket) -> Vec<Vec<u8>> {
                tokio::time::timeout(Duration::from_secs(3), async {
                    loop {
                        match socket.recv_multipart(zmq::DONTWAIT) {
                            Ok(frames) => return frames,
                            Err(zmq::Error::EAGAIN) => {
                                tokio::time::sleep(Duration::from_millis(2)).await
                            }
                            Err(error) => panic!("KV test publisher receive: {error}"),
                        }
                    }
                })
                .await
                .expect("subscriber did not connect")
            }

            let subscribed = receive(&self.live).await;
            assert_eq!(
                subscribed[0],
                [b"\x01".as_slice(), self.topic.as_bytes()].concat()
            );
            let request = receive(&self.replay).await;
            assert_eq!(request.last().unwrap(), &0_u64.to_be_bytes());
            self.replay
                .send_multipart(
                    [
                        request[0].as_slice(),
                        &[],
                        &[],
                        &(-1_i64).to_be_bytes(),
                        &[],
                    ],
                    0,
                )
                .unwrap();
            let payload =
                crate::kv_index::source::test_support::event_payload(1.0, events, Some(0));
            self.live
                .send_multipart([self.topic.as_bytes(), &0_i64.to_be_bytes(), &payload], 0)
                .unwrap();
        }
    }

    #[tokio::test]
    async fn discovered_and_static_sources_enable_real_prefix_credit() {
        use crate::kv_index::source::test_support::full_store;
        use crate::kv_index::KVEvent;
        use axum::{routing::get, Json, Router};
        use std::sync::atomic::{AtomicUsize, Ordering};

        for static_source in [false, true] {
            let publisher = TestPublisher::new("kv/boot");
            let (endpoint, replay_endpoint) = publisher.endpoints();
            let probes = Arc::new(AtomicUsize::new(0));
            let mut app = Router::new().route(
                "/v1/models",
                get(|| async { Json(json!({"data":[{"id":"m","parent":null}]})) }),
            );
            if !static_source {
                let info_probes = probes.clone();
                let source_probes = probes.clone();
                let layout_probes = probes.clone();
                let sources = json!({"0": {
                    "publisher": "zmq", "endpoint": endpoint, "replay_endpoint": replay_endpoint,
                    "topic": publisher.topic, "enable_kv_cache_events": true
                }});
                let layout = publisher.layout("engine-model");
                app = app
                    .route("/server_info", get(move || {
                        let probes = info_probes.clone();
                        async move {
                            probes.fetch_add(1, Ordering::Relaxed);
                            Json(json!({"vllm_config":{
                                "model_config":{"model":"engine-model"}, "cache_config":{"block_size":16}
                            }}))
                        }
                    }))
                    .route("/kv_event_sources", get(move || {
                        let (probes, sources) = (source_probes.clone(), sources.clone());
                        async move {
                            probes.fetch_add(1, Ordering::Relaxed);
                            Json(sources)
                        }
                    }))
                    .route("/kv_routing_capabilities", get(move || {
                        let (probes, layout) = (layout_probes.clone(), layout.clone());
                        async move {
                            probes.fetch_add(1, Ordering::Relaxed);
                            Json(layout)
                        }
                    }));
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
            let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let sources = if static_source {
                HashMap::from([(base.clone(), publisher.static_source("m"))])
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
                if static_source { 0 } else { 3 }
            );

            let initial =
                prepare_prefill(&state, &workers, &policy, &request, "/v1/completions", None).await;
            let initial_prices = initial.1.unwrap();
            let validated = initial_prices.validate(&workers);
            let work = validated.prefill_work(workers[0].url()).unwrap();
            assert_eq!(work.new_prefill_tokens, 33);
            assert!(work.placement_verified);

            publisher
                .replay_empty_and_publish(&[KVEvent::BlockStored(full_store(
                    &tokens[..32],
                    16,
                    None,
                ))])
                .await;
            tokio::time::timeout(Duration::from_secs(3), async {
                loop {
                    let work = ops.estimate_prefill_work(
                        &[(workers[0].as_ref(), std::slice::from_ref(&tokens), true)],
                        &mut PromptHashes::new(),
                    )[0];
                    if work.new_prefill_tokens == 1 {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(2)).await;
                }
            })
            .await
            .expect("subscriber should ingest the warm prefix");

            let warm =
                prepare_prefill(&state, &workers, &policy, &request, "/v1/completions", None).await;
            let warm_prices = warm.1.unwrap();
            let validated = warm_prices.validate(&workers);
            let work = validated.prefill_work(workers[0].url()).unwrap();
            assert_eq!(work.new_prefill_tokens, 1);
            assert!(work.placement_verified);
            drop(validated);
            drop(state);
            task.abort();
        }
    }

    async fn models_server() -> (String, tokio::task::JoinHandle<()>) {
        use axum::{routing::get, Json, Router};
        let models = || async { Json(json!({"data":[{"id":"m","parent":null}]})) };
        let app = Router::new()
            .route("/p/v1/models", get(models))
            .route("/d/v1/models", get(models));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (base, task)
    }

    #[tokio::test]
    async fn worker_removal_does_not_wait_for_source_discovery() {
        for replace in [false, true] {
            check_worker_removal_during_probe(replace).await;
        }
    }

    async fn check_worker_removal_during_probe(replace: bool) {
        use axum::{routing::get, Json, Router};
        use tokio::sync::Notify;

        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let app = Router::new().route(
            "/get_server_info",
            get({
                let (entered, release) = (entered.clone(), release.clone());
                move || {
                    let (entered, release) = (entered.clone(), release.clone());
                    async move {
                        entered.notify_one();
                        release.notified().await;
                        Json(json!({"model_id": "m", "kv_block_size": 16}))
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let registry = Arc::new(WorkerRegistry::new());
        registry.register(Arc::new(BasicWorker::new(url.clone(), WorkerType::Decode)));
        let policy = Arc::new(KvAwarePolicy::default());
        let mut state = PdKvState::new(Client::new());
        state.registry = Some(registry.clone());
        state.decode_policy = Some(policy.clone());
        let state = Arc::new(state);
        let refresh = {
            let state = state.clone();
            tokio::spawn(async move { state.bind_refresh().await })
        };
        tokio::time::timeout(Duration::from_secs(1), entered.notified())
            .await
            .expect("source probe did not start");
        registry.remove_by_url(&url).unwrap();
        tokio::time::timeout(
            Duration::from_millis(500),
            state.unregister_worker(&url, policy.as_ref()),
        )
        .await
        .expect("worker removal must finish while the source probe is blocked");
        if replace {
            let worker = Arc::new(BasicWorker::new(url.clone(), WorkerType::Decode));
            registry.register(worker.clone());
            policy.register_worker(worker.as_ref(), None, None);
            state.targets.lock().await.insert(
                url.clone(),
                TargetRegistration {
                    revision: 1,
                    decode: true,
                },
            );
        }
        release.notify_one();
        refresh.await.unwrap();
        assert_eq!(
            state
                .targets
                .lock()
                .await
                .get(&url)
                .map(|binding| binding.revision),
            replace.then_some(1),
        );
        server.abort();
    }

    #[tokio::test]
    async fn start_registers_and_prices_both_pools_before_any_request() {
        // Lazy request-path binding would leave a burst unpriced for one tick.
        let (base, server) = models_server().await;
        let p_base = format!("{base}/p");
        let d_base = format!("{base}/d");
        let prefill_publisher = TestPublisher::new("kv/prefill");
        let decode_publisher = TestPublisher::new("kv/decode");
        let prefill_policy: Arc<dyn LoadBalancingPolicy> = Arc::new(KvAwarePolicy::default());
        let decode_policy: Arc<dyn LoadBalancingPolicy> = Arc::new(KvAwarePolicy::default());
        let prefill: Vec<Arc<dyn Worker>> = vec![Arc::new(BasicWorker::new(
            p_base.clone(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        ))];
        let decode: Vec<Arc<dyn Worker>> = vec![Arc::new(BasicWorker::new(
            d_base.clone(),
            WorkerType::Decode,
        ))];
        let registry = Arc::new(crate::core::WorkerRegistry::new());
        for worker in prefill.iter().chain(decode.iter()) {
            registry.register(worker.clone());
        }
        let state = PdKvState::start(
            reqwest::Client::new(),
            None,
            HashMap::from([
                (p_base.clone(), prefill_publisher.static_source("m")),
                (d_base.clone(), decode_publisher.static_source("m")),
            ]),
            KvConnector::Nixl,
            HashMap::new(),
            registry.clone(),
            Some(prefill_policy.clone()),
            Some(decode_policy.clone()),
            false,
            true,
            true,
        )
        .await;
        tokio::join!(
            prefill_publisher.replay_empty_and_publish(&[]),
            decode_publisher.replay_empty_and_publish(&[]),
        );
        let request = json!({"model":"m","prompt":[1,2,3]});
        let credit = state
            .credit_by_base(&[p_base.clone(), d_base.clone()], &request)
            .await;
        assert_eq!(credit.get(&p_base), Some(&true));
        assert_eq!(credit.get(&d_base), Some(&true));
        assert!(state.all_priceable(&prefill, &request, &credit));
        let unbound: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "http://unbound".to_owned(),
            WorkerType::Decode,
        ));
        let mixed = [decode[0].clone(), unbound.clone()];
        let mut mixed_credit = credit.clone();
        mixed_credit.insert(unbound.base_url().to_owned(), true);
        assert!(!state.all_priceable(&mixed, &request, &mixed_credit));
        let prompts: Arc<[Vec<u32>]> = Arc::from(vec![vec![1, 2, 3]]);
        assert!(
            state
                .prepare_decode_prices(
                    &decode,
                    &decode_policy,
                    &request,
                    "/v1/completions",
                    None,
                    Some(&prompts),
                )
                .await
                .is_some(),
            "Decode must be priced before the first request"
        );
        // HTTP prices Decode only from free ids: a token-id prompt prices, a string falls back.
        assert!(
            state
                .prepare_decode_prices(
                    &decode,
                    &decode_policy,
                    &request,
                    "/v1/completions",
                    None,
                    None,
                )
                .await
                .is_some(),
            "an HTTP direct token-id prompt prices without rendering"
        );
        for (extra, eligible) in [
            (json!({}), true),
            (json!({"cache_salt": "tenant"}), false),
            (json!({"skip_reading_prefix_cache": true}), false),
        ] {
            let mut request = request.clone();
            request
                .as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            let (_, _, admission) = state
                .prepare_stages(
                    &prefill,
                    &prefill_policy,
                    &decode,
                    &decode_policy,
                    &request,
                    "/v1/completions",
                    None,
                    Some(&prompts),
                    false,
                )
                .await;
            let commit = state
                .prepare_decode_prices(
                    &decode,
                    &decode_policy,
                    &request,
                    "/v1/completions",
                    None,
                    Some(&prompts),
                )
                .await;
            assert_eq!(admission.is_some(), eligible, "admission: {extra}");
            assert_eq!(commit.is_some(), eligible, "commit: {extra}");
        }
        assert!(
            state
                .prepare_decode_prices(
                    &decode,
                    &decode_policy,
                    &json!({"model":"m","prompt":"hello"}),
                    "/v1/completions",
                    None,
                    None,
                )
                .await
                .is_none(),
            "an HTTP string body must not render Decode for pricing"
        );
        // The same URL changes role before the next refresh. The cached
        // signature matches, but the owner changed, so the Decode policy must
        // be registered rather than skipped.
        let prefill_kv = kv_aware_ops(prefill_policy.as_ref()).unwrap();
        let decode_kv = kv_aware_ops(decode_policy.as_ref()).unwrap();
        let old_prefill = prefill[0].clone();
        registry.remove_by_url(old_prefill.url());
        let decode_role: Arc<dyn Worker> =
            Arc::new(BasicWorker::new(p_base.clone(), WorkerType::Decode));
        registry.register(decode_role.clone());
        state.bind_refresh().await;
        assert!(
            decode_kv.estimate_decode_work(
                &[(decode_role.as_ref(), &[vec![1, 2, 3]], true)],
                &mut PromptHashes::new(),
            )[0]
            .additional_tokens
            .is_some(),
            "a role change must register the new owning policy"
        );
        assert_eq!(
            prefill_kv.estimate_decode_work(
                &[(decode_role.as_ref(), &[vec![1, 2, 3]], true)],
                &mut PromptHashes::new(),
            )[0]
            .additional_tokens,
            None,
            "the old Prefill policy target must be removed"
        );
        // Refresh removes a deleted Decode worker from its owning policy.
        let removed = decode[0].clone();
        registry.remove_by_url(removed.url());
        state.bind_refresh().await;
        assert_eq!(
            decode_kv.estimate_decode_work(
                &[(removed.as_ref(), &[vec![1, 2, 3]], true)],
                &mut PromptHashes::new(),
            )[0]
            .additional_tokens,
            None,
            "Prefill refresh must unregister the Decode target through Decode policy"
        );
        registry.register(removed.clone());
        state.bind_refresh().await;
        // A stale refresh snapshot must not revive the removed worker.
        registry.remove_by_url(removed.url());
        state.unregister_worker(removed.url(), decode_kv).await;
        state.bind_refresh().await;
        assert!(
            state
                .prepare_decode_prices(
                    &decode,
                    &decode_policy,
                    &request,
                    "/v1/completions",
                    None,
                    Some(&prompts),
                )
                .await
                .is_none(),
            "a stale refresh must not re-register a removed Decode worker"
        );
        registry.register(decode[0].clone());
        state.bind_refresh().await;
        assert!(
            state
                .prepare_decode_prices(
                    &decode,
                    &decode_policy,
                    &request,
                    "/v1/completions",
                    None,
                    Some(&prompts),
                )
                .await
                .is_some(),
            "re-added Decode worker must be registered and priceable again"
        );
        server.abort();
    }
}
