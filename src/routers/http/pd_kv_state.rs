//! P/D prefill KV discovery, refresh, and request-input preparation.

use super::pd_types::PdPrefillInput;
use crate::backend::preprocess::{
    http_input_requires_render, prepare_http_token_inputs, router_token_inputs,
};
use crate::core::worker::worker_get;
use crate::core::Worker;
use crate::kv_index::discovery::WorkerDiscovery;
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

/// One policy registration owned by this state, tagged with the role that
/// installed it so a shared target table can unregister through the right
/// policy.
struct TargetRegistration {
    signature: String,
    decode: bool,
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

pub(super) struct PdKvState {
    client: reqwest::Client,
    api_key: Option<String>,
    supervisor: KvIndexSupervisor,
    registry: Option<Arc<crate::core::WorkerRegistry>>,
    prefill_policy: Option<Arc<dyn LoadBalancingPolicy>>,
    decode_policy: Option<Arc<dyn LoadBalancingPolicy>>,
    bindings: Mutex<HashMap<String, Binding>>,
    targets: Mutex<HashMap<String, TargetRegistration>>,
    // Serializes bind() runs. Discovery and subscriber handshakes run outside
    // the map locks, so a slow P engine can only delay its own refresh —
    // never the request path's short binding reads.
    bind_lock: Mutex<()>,
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
    /// `prefill_policy` and `decode_policy` rank the stages that publish KV.
    /// Worker membership comes from `registry`; startup binds through the same
    /// entry point as refresh so the two cannot diverge.
    pub(super) async fn start(
        client: reqwest::Client,
        api_key: Option<String>,
        sources: HashMap<String, crate::config::types::PdKvEventSources>,
        registry: Arc<crate::core::WorkerRegistry>,
        prefill_policy: Option<Arc<dyn LoadBalancingPolicy>>,
        decode_policy: Option<Arc<dyn LoadBalancingPolicy>>,
    ) -> Arc<Self> {
        let mut state = Self::new_with_statics(client, sources);
        state.api_key = api_key;
        state.supervisor.set_api_key(state.api_key.clone());
        state.registry = Some(registry);
        state.prefill_policy = prefill_policy;
        state.decode_policy = decode_policy;
        let state = Arc::new(state);
        for decode in [false, true] {
            if tokio::time::timeout(CONTROL_TIMEOUT, state.bind_refresh(decode))
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
        let (supervisor, mut signals) =
            KvIndexSupervisor::new_pd(client.clone(), HashMode::Sha256, static_sources);
        Self {
            client,
            supervisor,
            api_key: None,
            registry: None,
            prefill_policy: None,
            decode_policy: None,
            bindings: Mutex::new(HashMap::new()),
            targets: Mutex::new(HashMap::new()),
            bind_lock: Mutex::new(()),
            // Subscribers enforce trust by suspending/clearing affected sources.
            signals: tokio::spawn(async move { while signals.recv().await.is_some() {} }),
        }
    }

    /// Reconcile bindings off the request path. Requests never wait for
    /// control-plane probes, so one hung P endpoint cannot delay admission.
    /// Until the first successful bind, requests see no binding and route
    /// without KV credit.
    fn spawn_refresh(self: &Arc<Self>) {
        let weak = Arc::downgrade(self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(REFRESH_TICK);
            loop {
                interval.tick().await;
                let Some(state) = weak.upgrade() else { break };
                // Each role reads its current pool under bind_lock. A snapshot
                // taken before the lock could re-register a worker that was
                // removed while this tick waited.
                state.bind_refresh(false).await;
                state.bind_refresh(true).await;
            }
        });
    }

    /// Reconcile the pool of one role, including incarnation and publisher
    /// changes. Binds serialize on `bind_lock`; refresh probes and subscriber
    /// handshakes run outside every map lock so an unresponsive engine can
    /// only delay its own refresh, never the request path. Never hold the
    /// admission load lock.
    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    #[cfg_attr(coverage_nightly, coverage(off))]
    async fn bind(&self, workers: &[Arc<dyn Worker>], policy: &dyn KvAwareOps) {
        let _bind_guard = self.bind_lock.lock().await;
        let decode = workers
            .first()
            .is_some_and(|worker| matches!(worker.worker_type(), crate::core::WorkerType::Decode));
        self.bind_locked(workers, policy, decode).await;
    }

    fn policy(&self, decode: bool) -> Option<&Arc<dyn LoadBalancingPolicy>> {
        if decode {
            self.decode_policy.as_ref()
        } else {
            self.prefill_policy.as_ref()
        }
    }

    async fn bind_refresh(&self, decode: bool) {
        let _bind_guard = self.bind_lock.lock().await;
        let Some(registry) = self.registry.clone() else {
            return;
        };
        let workers = if decode {
            registry.get_decode_workers()
        } else {
            registry.get_prefill_workers()
        };
        let Some(policy) = self.policy(decode) else {
            return;
        };
        let Some(kv) = kv_aware_ops(policy.as_ref()) else {
            return;
        };
        self.bind_locked(&workers, kv, decode).await;
    }

    fn live_workers(&self, fallback: &[Arc<dyn Worker>]) -> Vec<Arc<dyn Worker>> {
        match self.registry.as_ref() {
            Some(registry) => {
                let mut workers = registry.get_prefill_workers();
                workers.extend(registry.get_decode_workers());
                workers
            }
            None => fallback.to_vec(),
        }
    }

    async fn bind_locked(
        &self,
        workers: &[Arc<dyn Worker>],
        policy: &dyn KvAwareOps,
        decode: bool,
    ) {
        // Bindings outlive the pool that installed them: another role's pool
        // keeps its base registered, so removal is decided by every live pool
        // rather than by this one.
        let live = self.live_workers(workers);
        let current_bases: HashSet<String> = live
            .iter()
            .map(|worker| worker.base_url().to_owned())
            .collect();
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
        let mut removed = Vec::new();
        {
            let mut targets = self.targets.lock().await;
            // Every live pool, not just this one: one role's bind must not
            // unregister the other role's targets.
            let current: HashSet<_> = live.iter().map(|worker| worker.url().to_owned()).collect();
            targets.retain(|target, registration| {
                if current.contains(target) {
                    return true;
                }
                removed.push((target.clone(), registration.decode));
                false
            });
        }
        for (target, decode) in removed {
            if let Some(policy) = self.policy(decode) {
                if let Some(kv) = kv_aware_ops(policy.as_ref()) {
                    kv.unregister_worker(&target);
                }
            }
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
            let previous = targets.get(worker.url());
            let same = previous.is_some_and(|registration| {
                registration.signature == signature && registration.decode == decode
            });
            if same {
                continue;
            }
            let old_decode = previous
                .filter(|registration| registration.decode != decode)
                .map(|registration| registration.decode);
            if let Some(old_decode) = old_decode {
                if let Some(policy) = self.policy(old_decode) {
                    if let Some(kv) = kv_aware_ops(policy.as_ref()) {
                        kv.unregister_worker(worker.url());
                    }
                }
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
            targets.insert(
                worker.url().to_owned(),
                TargetRegistration { signature, decode },
            );
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

    /// Whether credit may be used for one base.
    ///
    /// `credit_fallback` returns the *reason* credit is refused, or `None` when it is
    /// allowed. Both preparation paths call this instead of testing the option themselves,
    /// because one of them had the polarity backwards and refused every eligible request.
    fn credit_eligible(cache_salt: Option<&str>, model_verified: bool) -> bool {
        KvAwarePolicy::credit_fallback(cache_salt, model_verified).is_none()
    }

    /// Whether **every** candidate can actually be priced.
    ///
    /// Two conditions, because they are checked in different places: the base must be
    /// allowed to claim credit (salt and model identity) *and* its binding must carry a
    /// cache key. A base whose model is merely advertised passes the first check but has no
    /// index to match against, and pricing then falls back after the render has been paid
    /// for. `decode_context_from_results` needs an estimate for every candidate, so the
    /// caller must require the whole set rather than any member.
    async fn all_priceable(
        &self,
        workers: &[Arc<dyn Worker>],
        cache_salt: Option<&str>,
        allow_credit: &HashMap<String, bool>,
    ) -> bool {
        if workers.is_empty() {
            return false;
        }
        let bindings = self.bindings.lock().await;
        workers.iter().all(|worker| {
            let base = worker.base_url();
            if !Self::credit_eligible(cache_salt, allow_credit.get(base).copied().unwrap_or(false))
            {
                return false;
            }
            let Some(info) = bindings.get(base).and_then(|binding| binding.info.as_ref()) else {
                return false;
            };
            if info.cache_key.is_none() {
                return false;
            }
            // Registering a policy target also needs this rank in the source list, and an
            // unranked worker counts as rank 0 there - the same convention here, or a source
            // covering only rank 1 would pass this check and still fall back after render.
            let rank = worker.dp_rank().unwrap_or(0) as u32;
            if !info.ranks.is_empty() && !info.ranks.contains(&rank) {
                return false;
            }
            true
        })
    }

    /// Remove a worker from the policy and this state's registration cache, so
    /// a re-added URL cannot reuse a stale signature.
    pub(super) async fn unregister_worker(&self, url: &str, policy: &dyn KvAwareOps) {
        let _guard = self.bind_lock.lock().await;
        policy.unregister_worker(url);
        self.targets.lock().await.remove(url);
        let base = super::dp_utils::parse_worker_url(url).0;
        let base_still_live = self.registry.as_ref().is_some_and(|registry| {
            registry
                .get_prefill_workers()
                .iter()
                .chain(registry.get_decode_workers().iter())
                .any(|worker| worker.base_url() == base)
        });
        if !base_still_live {
            self.bindings.lock().await.remove(&base);
            self.supervisor.on_worker_removed(&base);
        }
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

    /// Prepare both stages for one request.
    ///
    /// Rendering runs for both pools at once, because it is the only network work here;
    /// estimation is local and walks the rendered prompts once, sharing one hash cache and
    /// one credit decision. Awaiting one pool and then the other made two slow backends add
    /// up in admission latency.
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
        Option<HashMap<String, u64>>,
    ) {
        let started = Instant::now();
        let allow_credit = self.credit_by_base(request).await;
        let cache_salt = request.get("cache_salt").and_then(Value::as_str);
        let bases = |workers: &[Arc<dyn Worker>]| -> Vec<String> {
            workers
                .iter()
                .map(|worker| worker.base_url().to_owned())
                .collect()
        };
        // Decide whether any Decode candidate can be credited *before* rendering: a salted
        // request, an unverified identity or a pool that publishes no events all end with
        // the rendered result discarded, and the render is the slow part.
        // A deferred decision only picks a provisional worker here and re-prices at commit,
        // so rendering and pricing Decode now would be discarded - and could stall admission
        // on a slow backend for nothing.
        // A candidate can only be priced when its base is creditable. `decode_context_from_
        // results` needs an estimate for every candidate, so pricing with a partial set
        // always falls back - after paying for the slowest render. Require the whole set.
        // Decode pricing also must not fan out backend renders: without router
        // token ids a string/chat body would `/render` every Decode base on the
        // admission hot path for a load-balancing heuristic. Fall back to the
        // load ladder there; marginal pricing then applies only where the ids
        // are already free (gRPC/router_prompts, or a direct token-id prompt).
        let decode_expected = !defer_decode && kv_aware_ops(decode_policy.as_ref()).is_some();
        let decode_requires_render =
            router_prompts.is_none() && http_input_requires_render(request, route);
        let decode_priced_here = decode_expected
            && !decode_requires_render
            && self
                .all_priceable(decode_workers, cache_salt, &allow_credit)
                .await;
        let (prefill_results, decode_results) = tokio::join!(
            self.tokenize(
                bases(prefill_workers),
                request,
                route,
                headers,
                router_prompts
            ),
            async {
                if decode_priced_here {
                    self.tokenize(
                        bases(decode_workers),
                        request,
                        route,
                        headers,
                        router_prompts,
                    )
                    .await
                } else {
                    HashMap::new()
                }
            },
        );
        // One hash cache for the request, shared by both stages. Prefix credit
        // requires an empty LoRA adapter: `http_cache_identity` refuses any
        // other request, so both estimates hash the base model's blocks.
        let mut hashes = PromptHashes::new();
        let mut prepared = Self::build_prefill_inputs(request, &allow_credit, prefill_results);
        if let Some(kv) = kv_aware_ops(prefill_policy.as_ref()) {
            Self::attach_prefill_work(kv, prefill_workers, &mut prepared, &mut hashes);
        }
        // One eligibility decision for both stages, so the two cannot diverge.

        let decode_context = if decode_priced_here {
            let kv = kv_aware_ops(decode_policy.as_ref()).expect("priced decode policy");
            Self::decode_context_from_results(
                kv,
                decode_workers,
                &allow_credit,
                &decode_results,
                &mut hashes,
            )
        } else {
            // Only a request that meant to price Decode and could not is "unbound". A
            // deferred decision skips pricing here by design and prices at commit, so
            // counting it would flag a normal path in the diagnostic metric.
            if decode_expected {
                crate::metrics::RouterMetrics::record_kv_placement_lookup("decode_unbound");
            }
            None
        };
        metrics::histogram!("vllm_router_pd_kv_prepare_seconds")
            .record(started.elapsed().as_secs_f64());
        (prepared, decode_context)
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
        let work = kv.estimate_decode_work(&inputs, hashes);
        let mut context = HashMap::with_capacity(work.len());
        for ((worker, _, _), estimate) in inputs.iter().zip(work) {
            context.insert(worker.url().to_owned(), estimate.additional_tokens?);
        }
        Some(context)
    }

    /// Marginal Decode context per candidate, priced against the KV index.
    /// `hashes` is the caller's cache: the admission path shares it with the
    /// Prefill estimate, while the deferred commit starts a fresh one because it
    /// prices against the index as it stands at commit time.
    pub(super) async fn prepare_decode_context(
        &self,
        workers: &[Arc<dyn Worker>],
        policy: &Arc<dyn LoadBalancingPolicy>,
        request: &Value,
        route: &str,
        headers: Option<&HeaderMap>,
        router_prompts: Option<&Arc<[Vec<u32>]>>,
    ) -> Option<HashMap<String, u64>> {
        // The only caller is the deferred commit, which prices against the index as it
        // stands now, so this owns its cache. The admission path shares one cache across both
        // stages through `prepare_stages`.
        let hashes = &mut PromptHashes::new();
        let kv = kv_aware_ops(policy.as_ref())?;
        // Decode pricing must not fan out backend renders. Without router token
        // ids a string/chat body would re-render the whole pool for ids that
        // cannot have changed, so the pool falls back to the load ladder instead
        // of paying a second full render fan-out on the commit path.
        if router_prompts.is_none() && http_input_requires_render(request, route) {
            crate::metrics::RouterMetrics::record_kv_placement_lookup("decode_unbound");
            return None;
        }
        let allow_credit = self.credit_by_base(request).await;
        // A salted request must not be credited with unsalted blocks: the same rule
        // Prefill applies to its own estimate, so both stages price the same prefix.
        let cache_salt = request.get("cache_salt").and_then(Value::as_str);
        // Nothing to price unless some Decode base can be credited. Checking
        // first keeps the render fan-out and the hash walk off the request path
        // for every deployment whose Decode pool publishes no KV events.
        // Same rule as the admission path: rendering a partial set always ends in a
        // fallback, and the render is the slow part.
        if !self.all_priceable(workers, cache_salt, &allow_credit).await {
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
        Self::decode_context_from_results(kv, workers, &allow_credit, &results, hashes)
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

    fn decode_worker(base: &str) -> Arc<dyn Worker> {
        Arc::new(BasicWorker::new(base.to_string(), WorkerType::Decode))
    }

    fn bound(info: Option<WorkerKvInfo>) -> Binding {
        Binding {
            signature: "s".to_string(),
            checked: Instant::now(),
            info,
            base_models: HashSet::from(["m".to_string()]),
            probe_failures: 0,
        }
    }

    /// One unpriceable candidate must not silence credit for the rest **if it is filtered
    /// out first**, and `all_priceable` must reject a set that still contains it. This is the
    /// pair of behaviours that kept a healthy-but-mismatched worker from disabling credit.
    #[tokio::test]
    async fn pricing_requires_every_remaining_candidate_to_be_bound() {
        let state = PdKvState::new(Client::new());
        let bound_worker = decode_worker("http://d0");
        let unbound_worker = decode_worker("http://d1");
        {
            let mut bindings = state.bindings.lock().await;
            bindings.insert(
                bound_worker.base_url().to_string(),
                bound(Some(WorkerKvInfo {
                    instance_id: "i".to_string(),
                    block_size: Some(16),
                    ranks: vec![0],
                    cache_key: Some(crate::kv_index::discovery::CacheKey {
                        model: "m".into(),
                        hash_mode: crate::kv_index::HashMode::Sha256,
                        block_size: 16,
                    }),
                })),
            );
        }
        let allow = HashMap::from([
            (bound_worker.base_url().to_string(), true),
            (unbound_worker.base_url().to_string(), true),
        ]);

        assert!(
            state
                .all_priceable(std::slice::from_ref(&bound_worker), None, &allow)
                .await,
            "a bound candidate is priceable"
        );
        assert!(
            !state
                .all_priceable(&[bound_worker, unbound_worker], None, &allow)
                .await,
            "a candidate without a cache key still makes the whole set unpriceable"
        );
    }

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
    ) -> HashMap<String, PdPrefillInput> {
        state
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
            .await
            .0
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
            let prepared = prepare_prefill(&state, &workers, &policy, &request, route, None).await;
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
            let inputs = prepare_prefill(
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
            let initial =
                prepare_prefill(&state, &workers, &policy, &request, "/v1/completions", None).await;
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
            let warm =
                prepare_prefill(&state, &workers, &policy, &request, "/v1/completions", None).await;
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
                let fallback =
                    prepare_prefill(&state, &workers, &policy, &request, "/v1/completions", None)
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
    async fn start_registers_and_prices_both_pools_before_any_request() {
        // Lazy request-path binding would leave a burst unpriced for one tick.
        let (base, server) = models_server().await;
        let p_base = format!("{base}/p");
        let d_base = format!("{base}/d");
        let sources = |port: u16| crate::config::types::PdKvEventSources {
            endpoints: vec![format!("tcp://127.0.0.1:{port}")],
            replay_endpoints: None,
            block_size: 16,
            model: Some("m".to_owned()),
        };
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
                (p_base.clone(), sources(5557)),
                (d_base.clone(), sources(5558)),
            ]),
            registry.clone(),
            Some(prefill_policy.clone()),
            Some(decode_policy.clone()),
        )
        .await;
        let credit = state.credit_by_base(&json!({"model":"m"})).await;
        assert_eq!(credit.get(&p_base), Some(&true));
        assert_eq!(credit.get(&d_base), Some(&true));
        assert!(state.all_priceable(&prefill, None, &credit).await);
        let prompts: Arc<[Vec<u32>]> = Arc::from(vec![vec![1, 2, 3]]);
        let request = json!({"model":"m","prompt":[1,2,3]});
        assert!(
            state
                .prepare_decode_context(
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
        // HTTP pools price Decode only when the ids are already free: a direct
        // token-id prompt needs no render and still prices, while a string body
        // would render and therefore falls back without a network call.
        assert!(
            state
                .prepare_decode_context(
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
        assert!(
            state
                .prepare_decode_context(
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
        state.bind_refresh(true).await;
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
        // Prefill refresh removes a deleted Decode worker through the Decode
        // policy, not its own.
        let removed = decode[0].clone();
        registry.remove_by_url(removed.url());
        state.bind_refresh(false).await;
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
        state.bind_refresh(true).await;
        // A stale refresh snapshot must not revive the removed worker.
        registry.remove_by_url(removed.url());
        state.unregister_worker(removed.url(), decode_kv).await;
        state.bind_refresh(true).await;
        assert!(
            state
                .prepare_decode_context(
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
        state.bind_refresh(true).await;
        assert!(
            state
                .prepare_decode_context(
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
        let inputs =
            prepare_prefill(&state, &workers, &policy, &request, "/v1/completions", None).await;
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
            let inputs =
                prepare_prefill(&state, &workers, &policy, &request, "/v1/completions", None).await;
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
        let inputs =
            prepare_prefill(&state, &workers, &policy, &request, "/v1/completions", None).await;
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
