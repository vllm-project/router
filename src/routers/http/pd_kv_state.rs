//! P/D prefill KV discovery, refresh, and request-input preparation.

use super::pd_types::PdPrefillInput;
use crate::backend::preprocess::{prepare_http_token_inputs, router_token_inputs};
use crate::core::Worker;
use crate::kv_index::discovery::{worker_get, WorkerDiscovery};
use crate::kv_index::{HashMode, KvIndexSupervisor, WorkerKvInfo};
use crate::policies::{kv_aware_ops, KvAwareOps, KvAwarePolicy, LoadBalancingPolicy, PromptHashes};
use axum::http::HeaderMap;
use futures_util::{stream, StreamExt};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

// HTTP prefill inputs and event bindings are prepared before admission.
const CONTROL_TIMEOUT: Duration = Duration::from_secs(3);
const REFRESH_INTERVAL: Duration = Duration::from_secs(30);
/// Background reconcile cadence. bind() itself throttles per-base probes
/// (REFRESH_INTERVAL when healthy, CONTROL_TIMEOUT while a binding is
/// missing), so the tick only bounds how quickly pool changes and recovery
/// are noticed.
const REFRESH_TICK: Duration = Duration::from_secs(1);

#[derive(Clone)]
struct Binding {
    signature: String,
    checked: Instant,
    info: Option<WorkerKvInfo>,
    base_models: HashSet<String>,
    probe_failures: u8,
}

/// Metadata and served models from one successful refresh probe.
struct Discovery {
    signature: String,
    base_models: HashSet<String>,
    worker: WorkerDiscovery,
}

/// The most recent P and D pools and effective policies seen on the request
/// path. The background refresh loop binds from them; recording them is a few
/// Arc clones, so requests never wait for control-plane probes.
#[derive(Clone, Default)]
struct RefreshInputs {
    prefill: Arc<[Arc<dyn Worker>]>,
    decode: Arc<[Arc<dyn Worker>]>,
    prefill_policy: Option<Arc<dyn LoadBalancingPolicy>>,
    decode_policy: Option<Arc<dyn LoadBalancingPolicy>>,
}

/// Consecutive failed probes before a previously healthy binding is treated as
/// gone. One dropped control-plane round trip must not wipe a P engine's KV
/// index; ~MAX_PROBE_FAILURES refresh intervals of unreachability may.
const MAX_PROBE_FAILURES: u8 = 3;

/// What bind() should do with a base after a refresh probe.
#[derive(Debug, PartialEq, Eq)]
enum ProbeOutcome {
    /// Same healthy incarnation: only the check timestamp and models move.
    Unchanged,
    /// Probe failed: keep whatever binding exists (possibly none) and retry.
    Keep,
    /// New or changed incarnation: tear down and re-add the subscriber.
    Rebuild,
    /// Healthy binding unreachable for too long: drop it and the index.
    Evict,
}

fn probe_outcome(existing: Option<&Binding>, signature: Option<&str>) -> ProbeOutcome {
    match (existing, signature) {
        (Some(binding), Some(signature))
            if binding.info.is_some() && binding.signature == signature =>
        {
            ProbeOutcome::Unchanged
        }
        // A failed probe never evicts a healthy binding or wipes its index;
        // retry after the normal refresh interval instead.
        (Some(binding), None) => {
            if binding.info.is_some() && binding.probe_failures + 1 >= MAX_PROBE_FAILURES {
                ProbeOutcome::Evict
            } else {
                ProbeOutcome::Keep
            }
        }
        // New or changed incarnation, and first success for an untracked base.
        (_, Some(_)) => ProbeOutcome::Rebuild,
        // Untracked base whose probe failed: record an empty binding so the
        // short CONTROL_TIMEOUT retry cadence applies.
        (None, None) => ProbeOutcome::Keep,
    }
}

/// A worker pool plus the policy that ranks it, when that policy is KV-aware.
type PoolInput = (Vec<Arc<dyn Worker>>, Option<Arc<dyn LoadBalancingPolicy>>);

pub(super) struct PdKvState {
    client: reqwest::Client,
    api_key: Option<String>,
    supervisor: KvIndexSupervisor,
    bindings: Mutex<HashMap<String, Binding>>,
    targets: Mutex<HashMap<String, String>>,
    // Serializes bind() runs. Discovery and subscriber handshakes run outside
    // the map locks, so a slow P engine can only delay its own refresh —
    // never the request path's short binding reads.
    bind_lock: Mutex<()>,
    // The background refresh loop's latest view of the P and D pools.
    refresh_inputs: Mutex<RefreshInputs>,
    // Base URLs configured to publish KV events. A pool with no entry has
    // nothing to subscribe to, and recording it would only churn the policy's
    // registration epoch — which invalidates live session bindings.
    published: HashSet<String>,
    signals: tokio::task::JoinHandle<()>,
}

impl std::fmt::Debug for PdKvState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PdKvState")
            .field("supervisor", &self.supervisor)
            .finish()
    }
}

impl PdKvState {
    /// `prefill` and `decode` are the pools this state should bind, each paired
    /// with the policy that ranks it (only when that policy is KV-aware).
    ///
    /// Both are registered here rather than on the first request: bindings are
    /// created by the refresh loop, so a pool recorded from the request path has
    /// no binding until the next tick, and every request admitted before that
    /// tick is priced without the credit the configuration asked for.
    pub(super) async fn start(
        client: reqwest::Client,
        api_key: Option<String>,
        sources: HashMap<String, crate::config::types::PdKvEventSources>,
        prefill: PoolInput,
        decode: PoolInput,
    ) -> Arc<Self> {
        let mut state = Self::new_with_statics(client, sources);
        state.api_key = api_key;
        state.supervisor.set_api_key(state.api_key.clone());
        let prefill_workers = prefill.0.clone();
        let decode_workers = decode.0.clone();
        state.refresh_inputs = Mutex::new(RefreshInputs {
            prefill: prefill.0.iter().cloned().collect(),
            prefill_policy: prefill.1,
            decode: decode.0.iter().cloned().collect(),
            decode_policy: decode.1,
        });
        let state = Arc::new(state);
        // Bind during startup; unavailable metadata falls back to load until
        // refresh. Both pools are bound here so a burst that arrives before the
        // first refresh tick is still priced with the credit it configured.
        let (prefill_policy, decode_policy) = {
            let inputs = state.refresh_inputs.lock().await;
            (inputs.prefill_policy.clone(), inputs.decode_policy.clone())
        };
        for (pool, policy) in [
            (&prefill_workers, prefill_policy),
            (&decode_workers, decode_policy),
        ] {
            let Some(policy) = policy else { continue };
            let Some(kv) = kv_aware_ops(policy.as_ref()) else {
                continue;
            };
            if pool.is_empty() {
                continue;
            }
            if tokio::time::timeout(CONTROL_TIMEOUT, state.bind(pool, kv))
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
    fn new(client: reqwest::Client) -> Self {
        Self::new_with_statics(client, HashMap::new())
    }

    fn new_with_statics(
        client: reqwest::Client,
        static_sources: HashMap<String, crate::config::types::PdKvEventSources>,
    ) -> Self {
        // Kept before the map is moved into the supervisor: the request path
        // asks whether a pool is configured to publish before it records that
        // pool for binding.
        let published: HashSet<String> = static_sources.keys().cloned().collect();
        let (supervisor, mut signals) =
            KvIndexSupervisor::new_pd(client.clone(), HashMode::Sha256, static_sources);
        Self {
            client,
            supervisor,
            api_key: None,
            bindings: Mutex::new(HashMap::new()),
            targets: Mutex::new(HashMap::new()),
            bind_lock: Mutex::new(()),
            refresh_inputs: Mutex::new(RefreshInputs::default()),
            published,
            // Subscribers enforce trust by suspending/clearing affected sources.
            signals: tokio::spawn(async move { while signals.recv().await.is_some() {} }),
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
                // One subscriber per configured base, whatever role it serves:
                // a Decode engine that publishes KV events is bound exactly like
                // a Prefill one, and its blocks become the credit its own
                // marginal-context estimate needs.
                for (workers, policy) in [
                    (&inputs.prefill, inputs.prefill_policy.as_ref()),
                    (&inputs.decode, inputs.decode_policy.as_ref()),
                ] {
                    let Some(policy) = policy else { continue };
                    let Some(kv) = kv_aware_ops(policy.as_ref()) else {
                        continue;
                    };
                    if !workers.is_empty() {
                        state.bind(&workers[..], kv).await;
                    }
                }
            }
        });
    }

    /// Reconcile the pool of one role, including incarnation and publisher
    /// changes. Binds serialize on `bind_lock`; refresh probes and subscriber
    /// handshakes run outside every map lock so an unresponsive engine can
    /// only delay its own refresh, never the request path. Never hold the
    /// admission load lock.
    async fn bind(&self, workers: &[Arc<dyn Worker>], policy: &dyn KvAwareOps) {
        let _bind_guard = self.bind_lock.lock().await;
        // Bindings outlive the pool that installed them: another role's pool
        // keeps its base registered, so removal is decided by every live pool
        // rather than by this one.
        let current_bases: HashSet<String> = {
            let inputs = self.refresh_inputs.lock().await;
            inputs
                .prefill
                .iter()
                .chain(inputs.decode.iter())
                .map(|worker| worker.base_url().to_owned())
                .collect()
        };
        let refresh: Vec<String> = {
            let mut bindings = self.bindings.lock().await;
            bindings.retain(|base, _| {
                if current_bases.contains(base) {
                    return true;
                }
                self.supervisor.on_worker_removed(base);
                false
            });
            let targets: HashSet<String> = workers
                .iter()
                .map(|worker| worker.base_url().to_owned())
                .collect();
            targets
                .iter()
                .filter(|base| {
                    bindings.get(*base).is_none_or(|binding| {
                        binding.checked.elapsed()
                            >= if binding.info.is_some() {
                                REFRESH_INTERVAL
                            } else {
                                CONTROL_TIMEOUT
                            }
                    })
                })
                .cloned()
                .collect()
        };
        {
            let mut targets = self.targets.lock().await;
            // Every live pool, not just this one: one role's bind must not
            // unregister the other role's targets.
            let current: HashSet<_> = {
                let inputs = self.refresh_inputs.lock().await;
                inputs
                    .prefill
                    .iter()
                    .chain(inputs.decode.iter())
                    .map(|worker| worker.url().to_owned())
                    .collect()
            };
            targets.retain(|target, _| {
                if current.contains(target) {
                    return true;
                }
                policy.unregister_worker(target);
                false
            });
        }
        // Probe refresh candidates without holding any map lock.
        let probes: Vec<(String, Option<Discovery>)> =
            stream::iter(refresh.into_iter().map(|base| async move {
                let discovery = tokio::time::timeout(CONTROL_TIMEOUT, self.probe_base(&base))
                    .await
                    .ok()
                    .and_then(Result::ok);
                (base, discovery)
            }))
            .buffer_unordered(8)
            .collect()
            .await;
        for (base, discovery) in probes {
            let existing = {
                let bindings = self.bindings.lock().await;
                bindings.get(&base).cloned()
            };
            let outcome = probe_outcome(
                existing.as_ref(),
                discovery
                    .as_ref()
                    .map(|snapshot| snapshot.signature.as_str()),
            );
            match outcome {
                ProbeOutcome::Unchanged => {
                    let mut bindings = self.bindings.lock().await;
                    if let Some(binding) = bindings.get_mut(&base) {
                        binding.checked = Instant::now();
                        binding.probe_failures = 0;
                        binding.base_models = discovery
                            .map(|snapshot| snapshot.base_models)
                            .unwrap_or_default();
                    }
                }
                ProbeOutcome::Keep | ProbeOutcome::Evict => {
                    let mut bindings = self.bindings.lock().await;
                    if outcome == ProbeOutcome::Evict {
                        bindings.remove(&base);
                    }
                    bindings
                        .entry(base.clone())
                        .and_modify(|binding| {
                            binding.checked = Instant::now();
                            binding.probe_failures = binding.probe_failures.saturating_add(1);
                        })
                        .or_insert_with(|| Binding {
                            signature: String::new(),
                            checked: Instant::now(),
                            info: None,
                            base_models: HashSet::new(),
                            probe_failures: 0,
                        });
                    drop(bindings);
                    if outcome == ProbeOutcome::Evict {
                        self.supervisor.on_worker_removed(&base);
                    }
                }
                ProbeOutcome::Rebuild => {
                    {
                        let mut bindings = self.bindings.lock().await;
                        bindings.remove(&base);
                    }
                    // The subscriber handshake runs without the map locks; a
                    // concurrent prepare() sees no binding and denies credit
                    // until the rebuild completes.
                    let snapshot = discovery.expect("successful discovery probe");
                    let info = self
                        .supervisor
                        .install_worker(&base, snapshot.worker)
                        .ok()
                        .filter(|info| info.cache_key.is_some() && !info.ranks.is_empty());
                    let mut bindings = self.bindings.lock().await;
                    bindings.insert(
                        base,
                        Binding {
                            signature: snapshot.signature,
                            checked: Instant::now(),
                            info,
                            base_models: snapshot.base_models,
                            probe_failures: 0,
                        },
                    );
                }
            }
        }
        let bindings = self.bindings.lock().await;
        let mut targets = self.targets.lock().await;
        for worker in workers {
            let binding = bindings.get(worker.base_url());
            let signature = binding
                .map(|binding| format!("{}:{:?}", binding.signature, binding.info))
                .unwrap_or_default();
            if targets.get(worker.url()) == Some(&signature) {
                continue;
            }
            policy.unregister_worker(worker.url());
            let info = binding.and_then(|binding| binding.info.as_ref());
            // Keep a load-only registration when discovery has no info: the
            // occupancy fallback (affinity off, cache_salt/tools, health
            // probes) shares this table, and KvAwarePolicy::prepare_selection
            // drops every unregistered worker once any target is registered.
            let matches = info.and_then(|info| {
                let rank = worker.dp_rank().unwrap_or(0) as u32;
                info.ranks
                    .contains(&rank)
                    .then(|| self.supervisor.match_provider_for_worker(worker.base_url()))
                    .flatten()
            });
            policy.register_worker(worker.as_ref(), info, matches);
            targets.insert(worker.url().to_owned(), signature);
        }
    }

    /// One refresh probe: control-plane discovery plus the served base models.
    /// `Ok` carries `(signature, base_models)`; any failure is an `Err`.
    /// Statically configured workers skip the control-plane queries and
    /// derive the signature from their configuration.
    async fn probe_base(&self, base: &str) -> Result<Discovery, String> {
        let worker = self.supervisor.discover(base).await;
        let info = worker.server_info.as_ref().map_err(Clone::clone)?;
        let sources = worker.sources.as_ref().map_err(Clone::clone)?;
        let mut sources: Vec<_> = sources.iter().collect();
        sources.sort_by(|left, right| left.0.cmp(right.0));
        let signature = format!("{info:?}:{sources:?}");
        // gRPC workers serve no HTTP /v1/models; Control.GetModelInfo names
        // the served model, including for statically configured sources.
        if crate::backend::is_grpc_url(base) {
            let base_models = crate::backend::control::get_grpc_model_info(base, CONTROL_TIMEOUT)
                .await
                .ok()
                .map(|info| info.model_id)
                .filter(|model| !model.trim().is_empty())
                .into_iter()
                .collect();
            return Ok(Discovery {
                signature,
                base_models,
                worker,
            });
        }
        let models_request = worker_get(
            &self.client,
            &format!("{}/v1/models", base.trim_end_matches('/')),
            self.api_key.as_deref(),
        );
        let models = match models_request.send().await {
            Ok(response) if response.status().is_success() => response.json::<Value>().await.ok(),
            _ => None,
        };
        let base_models: HashSet<String> = models
            .as_ref()
            .and_then(|models| models.get("data"))
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|model| model.get("parent").is_none_or(Value::is_null))
            .filter_map(|model| model.get("id").and_then(Value::as_str).map(str::to_owned))
            .collect();
        Ok(Discovery {
            signature,
            base_models,
            worker,
        })
    }

    /// Whether a binding's cache identity is verified for `model`, which is what
    /// lets its blocks be credited. Model/block-size declarations and a
    /// discovered cache identity are both accepted; a binding with neither gets
    /// no credit, so its candidate is priced at full occupancy instead.
    fn credit_allowed(binding: &Binding, model: Option<&str>) -> bool {
        model.is_some_and(|model| {
            binding.base_models.contains(model)
                || binding
                    .info
                    .as_ref()
                    .and_then(|info| info.cache_key.as_ref())
                    .is_some_and(|key| key.model.as_ref() == model)
        })
    }

    /// Whether any of `workers` is configured to publish KV events.
    ///
    /// Binding a pool that publishes nothing is not merely wasteful: the bind
    /// path re-registers each target, which starts a new registration epoch and
    /// makes every live session binding look stale. So the Decode pool is
    /// recorded for binding only when it has something to subscribe to.
    pub(super) fn publishes_for(&self, workers: &[Arc<dyn Worker>]) -> bool {
        workers
            .iter()
            .any(|worker| self.published.contains(worker.base_url()))
    }

    /// Record the Decode pool the request path saw, so the refresh loop binds
    /// its KV event subscriptions too. A Decode engine that enables prefix
    /// caching holds blocks of its own; without them the Router cannot price
    /// what a candidate must newly allocate.
    pub(super) async fn record_decode_pool(
        &self,
        workers: &[Arc<dyn Worker>],
        policy: &Arc<dyn LoadBalancingPolicy>,
    ) {
        let mut inputs = self.refresh_inputs.lock().await;
        inputs.decode = workers.iter().cloned().collect();
        inputs.decode_policy = Some(policy.clone());
    }

    #[cfg(test)]
    pub(super) async fn pool_sizes(&self) -> (usize, usize) {
        let inputs = self.refresh_inputs.lock().await;
        (inputs.prefill.len(), inputs.decode.len())
    }

    /// One snapshot of which bases may be credited, keyed by base URL.
    async fn credit_by_base(&self, request: &Value) -> HashMap<String, bool> {
        let bindings = self.bindings.lock().await;
        let model = request.get("model").and_then(Value::as_str);
        bindings
            .iter()
            .map(|(base, binding)| (base.clone(), Self::credit_allowed(binding, model)))
            .collect()
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
            let fallback = KvAwarePolicy::credit_fallback(
                request.get("cache_salt").and_then(Value::as_str),
                allow_credit.get(&base).copied().unwrap_or(false),
            );
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
        prepared
    }

    fn attach_prefill_work(
        kv: &dyn KvAwareOps,
        workers: &[Arc<dyn Worker>],
        prepared: &mut HashMap<String, PdPrefillInput>,
        hashes: &mut PromptHashes,
    ) {
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
            .zip(kv.estimate_prefill_work(&inputs, hashes))
            .map(|((worker, _, _), estimate)| {
                (worker.base_url().to_owned(), worker.dp_rank(), estimate)
            })
            .collect();
        for (base, rank, work) in work {
            if let Some(entry) = prepared.get_mut(&base) {
                entry.work_by_rank.insert(rank, work);
            }
        }
    }

    /// Pure: the Decode estimate over already-rendered prompts.
    ///
    /// Returns `None` when any candidate is unpriced, because one credited
    /// figure beside a full-price one would compare different units across the
    /// pool.
    fn decode_context_from_results(
        kv: &dyn KvAwareOps,
        workers: &[Arc<dyn Worker>],
        allow_credit: &HashMap<String, bool>,
        results: &HashMap<String, Result<crate::backend::preprocess::HttpTokenInput, &'static str>>,
        sequences: u32,
        hashes: &mut PromptHashes,
    ) -> Option<HashMap<String, u64>> {
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
        let work = kv.estimate_decode_work(&inputs, sequences, hashes);
        let mut context = HashMap::with_capacity(work.len());
        for ((worker, _, _), estimate) in inputs.iter().zip(work) {
            context.insert(worker.url().to_owned(), estimate.additional_tokens?);
        }
        Some(context)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn prepare_decode_context(
        &self,
        workers: &[Arc<dyn Worker>],
        policy: &Arc<dyn LoadBalancingPolicy>,
        request: &Value,
        route: &str,
        headers: Option<&HeaderMap>,
        router_prompts: Option<&Arc<[Vec<u32>]>>,
        sequences: u32,
    ) -> Option<HashMap<String, u64>> {
        let kv = kv_aware_ops(policy.as_ref())?;
        let allow_credit = self.credit_by_base(request).await;
        // A salted request must not be credited with unsalted blocks: the same rule
        // Prefill applies to its own estimate, so both stages price the same prefix.
        let cache_salt = request.get("cache_salt").and_then(Value::as_str);
        // Nothing to price unless some Decode base can be credited. Checking
        // first keeps the render fan-out and the hash walk off the request path
        // for every deployment whose Decode pool publishes no KV events.
        //
        // `credit_fallback` returns the reason credit is *not* allowed, or `None`
        // when it is: the guard must therefore look for a worker with no fallback.
        if !workers.iter().any(|worker| {
            KvAwarePolicy::credit_fallback(
                cache_salt,
                allow_credit
                    .get(worker.base_url())
                    .copied()
                    .unwrap_or(false),
            )
            .is_none()
        }) {
            crate::metrics::RouterMetrics::record_kv_placement_lookup("decode_unbound");
            return None;
        }
        let bases = workers
            .iter()
            .map(|worker| worker.base_url().to_owned())
            .collect();
        let results = self
            .tokenize(bases, request, route, headers, router_prompts)
            .await;
        Self::decode_context_from_results(
            kv,
            workers,
            &allow_credit,
            &results,
            sequences,
            &mut PromptHashes::new(),
        )
    }

    /// Prefill inputs and, when the Decode pool can be priced, its marginal
    /// context - both computed from one hash cache.
    ///
    /// This is the KV-aware-prefill path. The other shape, a count-based
    /// Prefill with a KV-aware Decode, goes through [`Self::prepare`] plus
    /// [`Self::prepare_decode_context`] instead: the Prefill stage has no work
    /// estimate to share there, so there is nothing to co-schedule.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn prepare_pair(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        prefill_policy: &Arc<dyn LoadBalancingPolicy>,
        decode_workers: &[Arc<dyn Worker>],
        decode_policy: &Arc<dyn LoadBalancingPolicy>,
        request: &Value,
        route: &str,
        headers: Option<&HeaderMap>,
        router_prompts: Option<&Arc<[Vec<u32>]>>,
        sequences: u32,
    ) -> (
        HashMap<String, PdPrefillInput>,
        Option<HashMap<String, u64>>,
    ) {
        let started = Instant::now();
        {
            let mut inputs = self.refresh_inputs.lock().await;
            inputs.prefill = prefill_workers.iter().cloned().collect();
            inputs.prefill_policy = Some(prefill_policy.clone());
        }
        let allow_credit = self.credit_by_base(request).await;
        let decode_priced = kv_aware_ops(decode_policy.as_ref()).is_some()
            && decode_workers.iter().any(|worker| {
                allow_credit
                    .get(worker.base_url())
                    .copied()
                    .unwrap_or(false)
            });
        if !decode_priced {
            crate::metrics::RouterMetrics::record_kv_placement_lookup("decode_unbound");
        }
        let prefill_bases: Vec<_> = prefill_workers
            .iter()
            .map(|worker| worker.base_url().to_owned())
            .collect();
        let decode_bases: Vec<_> = decode_workers
            .iter()
            .map(|worker| worker.base_url().to_owned())
            .collect();
        // Renders run together: they are network work over disjoint bases, and
        // serialising them would add a whole fan-out to request latency. They
        // compute no hashes, so nothing is shared across a mutable borrow here.
        let (prefill_results, decode_results) = tokio::join!(
            self.tokenize(prefill_bases, request, route, headers, router_prompts),
            async {
                if decode_priced {
                    self.tokenize(decode_bases, request, route, headers, router_prompts)
                        .await
                } else {
                    HashMap::new()
                }
            },
        );
        // One hash cache for both stages: the Decode estimate walks the same
        // prompt with the same block size, so hashing it twice would be pure
        // duplication, linear in prompt length, on the admission path.
        let mut hashes = PromptHashes::new();
        let mut prepared = Self::build_prefill_inputs(request, &allow_credit, prefill_results);
        if let Some(kv) = kv_aware_ops(prefill_policy.as_ref()) {
            Self::attach_prefill_work(kv, prefill_workers, &mut prepared, &mut hashes);
        }
        let decode_context = if decode_priced {
            kv_aware_ops(decode_policy.as_ref()).and_then(|kv| {
                Self::decode_context_from_results(
                    kv,
                    decode_workers,
                    &allow_credit,
                    &decode_results,
                    sequences,
                    &mut hashes,
                )
            })
        } else {
            None
        };
        metrics::histogram!("vllm_router_pd_kv_prepare_seconds")
            .record(started.elapsed().as_secs_f64());
        (prepared, decode_context)
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
        {
            let mut inputs = self.refresh_inputs.lock().await;
            inputs.prefill = workers.iter().cloned().collect();
            inputs.prefill_policy = Some(policy.clone());
        }
        let allow_credit = self.credit_by_base(request).await;
        let bases = workers
            .iter()
            .map(|worker| worker.base_url().to_owned())
            .collect();
        let results = self
            .tokenize(bases, request, route, headers, router_prompts)
            .await;
        let mut prepared = Self::build_prefill_inputs(request, &allow_credit, results);
        if let Some(kv) = kv_aware_ops(policy.as_ref()) {
            Self::attach_prefill_work(kv, workers, &mut prepared, &mut PromptHashes::new());
        }
        metrics::histogram!("vllm_router_pd_kv_prepare_seconds")
            .record(started.elapsed().as_secs_f64());
        prepared
    }
}

impl Drop for PdKvState {
    fn drop(&mut self) {
        self.supervisor.shutdown();
        self.signals.abort();
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod kv_input_tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};
    use crate::policies::{kv_aware_ops, KvAwarePolicy, PromptHashes};
    use axum::extract::Request;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use reqwest::Client;
    use serde_json::json;

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
                .estimate_prefill_work(
                    &[(workers[0].as_ref(), &[vec![7, 8, 9]], true)],
                    &mut PromptHashes::new(),
                )[0];
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
                    let work = ops.estimate_prefill_work(
                        &[(workers[0].as_ref(), std::slice::from_ref(&tokens), true)],
                        &mut PromptHashes::new(),
                    )[0];
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
            if !static_source {
                use crate::kv_index::{MatchQuery, StorageTier};

                let provider = state.supervisor.match_provider_for_worker(&base).unwrap();
                let query = MatchQuery {
                    group_idx: 0,
                    local_hashes: crate::kv_index::local_block_hash_strings(&tokens, 16, None),
                    tiers_of_interest: vec![StorageTier::Device],
                };
                assert_eq!(provider.find_tiered_matches(&query).len(), 1);
                let valid = source_metadata.lock()["0"].clone();
                *source_metadata.lock() = json!({"not-a-rank": valid});
                state.bindings.lock().await.get_mut(&base).unwrap().checked =
                    Instant::now() - REFRESH_INTERVAL;
                state.bind(&workers, ops).await;
                // Installation rejects the complete new snapshot before
                // retiring the old subscriber, but the changed PD binding
                // itself falls back without granting that old credit.
                assert!(state.supervisor.match_provider_for_worker(&base).is_some());
                assert_eq!(provider.find_tiered_matches(&query).len(), 1);
                assert!(state.bindings.lock().await[&base].info.is_none());
                let fallback = state
                    .prepare(&workers, &policy, &request, "/v1/completions", None, None)
                    .await;
                assert_eq!(fallback[&base].work_by_rank[&None].new_prefill_tokens, 33);
                assert!(!fallback[&base].work_by_rank[&None].placement_verified);
                assert_eq!(probes.load(Ordering::Relaxed), 4);
            }
            drop(state);
            task.abort();
        }
    }

    fn healthy_binding(signature: &str, probe_failures: u8) -> Binding {
        Binding {
            signature: signature.to_owned(),
            checked: Instant::now(),
            info: Some(WorkerKvInfo::default()),
            base_models: HashSet::new(),
            probe_failures,
        }
    }

    #[test]
    fn probe_outcomes_cover_identity_changes_and_failure_threshold() {
        let healthy = healthy_binding("sig", 0);
        let failing = healthy_binding("sig", MAX_PROBE_FAILURES - 1);
        let unverified = Binding {
            info: None,
            ..failing.clone()
        };
        for (binding, signature, expected) in [
            (Some(&healthy), Some("sig"), ProbeOutcome::Unchanged),
            (Some(&healthy), Some("other"), ProbeOutcome::Rebuild),
            (Some(&unverified), Some("sig"), ProbeOutcome::Rebuild),
            (Some(&failing), None, ProbeOutcome::Evict),
            (Some(&unverified), None, ProbeOutcome::Keep),
            (None, Some("sig"), ProbeOutcome::Rebuild),
            (None, None, ProbeOutcome::Keep),
        ] {
            assert_eq!(probe_outcome(binding, signature), expected);
        }
        for failures in 0..MAX_PROBE_FAILURES - 1 {
            let binding = healthy_binding("sig", failures);
            assert_eq!(probe_outcome(Some(&binding), None), ProbeOutcome::Keep);
        }
    }

    #[tokio::test]
    async fn start_registers_the_decode_pool_before_any_request() {
        // A Decode pool recorded lazily from the request path has no binding for
        // one refresh tick, and a burst admitted inside that tick loses the
        // credit its configuration asked for. Both pools must be present from
        // construction.
        let policy: Arc<dyn LoadBalancingPolicy> = Arc::new(KvAwarePolicy::default());
        let prefill: Vec<Arc<dyn Worker>> = vec![Arc::new(BasicWorker::new(
            "http://p0".to_owned(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        ))];
        let decode: Vec<Arc<dyn Worker>> = vec![Arc::new(BasicWorker::new(
            "http://d0".to_owned(),
            WorkerType::Decode,
        ))];
        let state = PdKvState::start(
            reqwest::Client::new(),
            None,
            HashMap::new(),
            (prefill, Some(policy.clone())),
            (decode, Some(policy)),
        )
        .await;
        assert_eq!(state.pool_sizes().await, (1, 1));
    }

    #[tokio::test]
    async fn failed_refresh_probes_keep_healthy_binding_until_consecutive_limit() {
        use axum::{routing::get, Json, Router};
        use std::sync::atomic::{AtomicBool, Ordering};

        // Rewind the binding's check timestamp so the next bind must re-probe.
        async fn rewind(state: &PdKvState, base: &str) {
            let mut bindings = state.bindings.lock().await;
            if let Some(binding) = bindings.get_mut(base) {
                binding.checked = Instant::now() - REFRESH_INTERVAL - CONTROL_TIMEOUT;
            }
        }

        let healthy = Arc::new(AtomicBool::new(true));
        let publisher_context = zmq::Context::new();
        let publisher = publisher_context.socket(zmq::PUB).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
        let app = Router::new()
            .route(
                "/server_info",
                get({
                    let healthy = healthy.clone();
                    move || {
                        let healthy = healthy.clone();
                        async move {
                            if healthy.load(Ordering::Relaxed) {
                                (
                                    StatusCode::OK,
                                    Json(json!({"vllm_config":{"model_config":{"model":"m"},"cache_config":{"block_size":16}}})),
                                )
                            } else {
                                (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({})))
                            }
                        }
                    }
                }),
            )
            .route(
                "/v1/models",
                get(|| async { Json(json!({"data":[{"id":"m","parent":null}]})) }),
            )
            .route("/kv_event_sources", get(move || {
                let endpoint = endpoint.clone();
                async move {
                    Json(json!({"0":{"publisher":"zmq","endpoint":endpoint,"enable_kv_cache_events":true}}))
                }
            }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let state = PdKvState::new(reqwest::Client::new());
        let policy: Arc<dyn LoadBalancingPolicy> = Arc::new(KvAwarePolicy::default());
        let ops = kv_aware_ops(policy.as_ref()).unwrap();
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(BasicWorker::new(
            base.clone(),
            WorkerType::Regular,
        ))];
        let request = json!({"model":"m","prompt":(0..8).collect::<Vec<u32>>()});

        // First bind: discovery succeeds and installs a binding with info.
        state.bind(&workers, ops).await;
        let inputs = state
            .prepare(&workers, &policy, &request, "/v1/completions", None, None)
            .await;
        assert!(inputs[&base].allow_credit);
        {
            let bindings = state.bindings.lock().await;
            let binding = bindings.get(&base).expect("binding after first discovery");
            assert!(binding.info.is_some());
            assert_eq!(binding.probe_failures, 0);
        }

        // Transient failures keep the healthy binding and its credit authority.
        healthy.store(false, Ordering::Relaxed);
        for expected_failures in 1..MAX_PROBE_FAILURES {
            rewind(&state, &base).await;
            state.bind(&workers, ops).await;
            let inputs = state
                .prepare(&workers, &policy, &request, "/v1/completions", None, None)
                .await;
            assert!(
                inputs[&base].allow_credit,
                "retained binding keeps credit authority"
            );
            let bindings = state.bindings.lock().await;
            let binding = bindings
                .get(&base)
                .expect("binding retained after transient failure");
            assert!(binding.info.is_some(), "index identity must survive");
            assert_eq!(binding.probe_failures, expected_failures);
        }

        // The consecutive-failure limit replaces the binding with an empty one
        // and tears the subscriber down.
        rewind(&state, &base).await;
        state.bind(&workers, ops).await;
        let inputs = state
            .prepare(&workers, &policy, &request, "/v1/completions", None, None)
            .await;
        assert!(!inputs[&base].allow_credit);
        let bindings = state.bindings.lock().await;
        let binding = bindings
            .get(&base)
            .expect("empty binding re-recorded for retry cadence");
        assert!(binding.info.is_none());
        assert_eq!(binding.probe_failures, 0);

        task.abort();
    }
}
