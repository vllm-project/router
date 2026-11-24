//! Worker trust, source retirement and replay recovery.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tracing::info;

use super::metrics_meta::EngineCacheConfigs;
use super::probe::{probe_bound_sources, probe_engine_cache_configs, WorkerDescriptor};
use super::reconcile::{
    observed_sources_match, probe_keeps_trust, probe_renews_trust, runtime_hash_mismatch,
    BoundSources, StaticHybridMatch,
};
use super::{CacheKey, InstalledWorkers, KvIndexSupervisor, WorkerKvInfo, WorkerSubs};
use crate::kv_index::subscriber::EventSource;
use crate::kv_index::{
    CacheCapabilities, EventStream, EventSubscriberHandle, IngestionSignal, KvBlockIndexer,
    SourceBinding,
};

pub(crate) const CONTROL_TIMEOUT: Duration = Duration::from_secs(3);
pub(super) const REFRESH_INTERVAL: Duration = Duration::from_secs(30);
pub(super) const MAX_PROBE_FAILURES: u8 = 3;
const LAYOUT_FRESHNESS: Duration = Duration::from_secs(5);
const HASH_PROBE_BUDGET: Duration = Duration::from_secs(2);

/// A read-only registration. Providers and policy bindings share its revision.
#[derive(Clone)]
pub(crate) struct WorkerRegistration {
    pub(crate) revision: u64,
    pub(crate) info: Option<WorkerKvInfo>,
    pub(crate) provider: Option<Arc<dyn crate::policies::TieredMatchProvider>>,
}

/// Control observations for the same owner used by its event subscribers.
/// Descriptor failures and layout freshness have distinct clocks and evidence.
pub(super) struct WorkerLifecycle {
    pub(super) active: Arc<AtomicBool>,
    layout: bool,
    pub(super) state: Mutex<LifecycleState>,
}

pub(super) struct LifecycleState {
    pub(super) revision: u64,
    pub(super) info: Option<WorkerKvInfo>,
    pub(super) base_models: HashSet<String>,
    pub(super) descriptor: Option<WorkerDescriptor>,
    pub(super) hash_fences: BoundSources,
    pub(super) hash_checked: Instant,
    pub(super) subscriptions: Vec<Subscription>,
    // None schedules immediate descriptor discovery after a layout change.
    pub(super) checked: Option<Instant>,
    pub(super) probe_failures: u8,
    pub(super) trusted: bool,
    pub(super) verified_at: Instant,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum SourceAction {
    Keep,
    Recover,
    Rebuild,
    Stop,
}

impl WorkerLifecycle {
    pub(super) fn new(
        active: Arc<AtomicBool>,
        layout: bool,
        descriptor: Option<WorkerDescriptor>,
    ) -> Self {
        Self {
            active,
            layout,
            state: Mutex::new(LifecycleState {
                revision: crate::kv_index::next_owner_incarnation(),
                info: None,
                base_models: HashSet::new(),
                descriptor,
                hash_fences: BoundSources::new(),
                hash_checked: Instant::now(),
                subscriptions: Vec::new(),
                checked: Some(Instant::now()),
                probe_failures: 0,
                trusted: true,
                verified_at: Instant::now(),
            }),
        }
    }

    pub(super) fn trusted_at(&self, state: &LifecycleState, now: impl FnOnce() -> Instant) -> bool {
        self.active.load(Ordering::Acquire)
            && state.trusted
            && (!self.layout
                || now().saturating_duration_since(state.verified_at) <= LAYOUT_FRESHNESS)
    }
    fn trusted(&self, state: &LifecycleState) -> bool {
        self.trusted_at(state, Instant::now)
    }

    pub(super) fn read_rank<T>(
        &self,
        rank: &SourceBinding,
        mut first_trust_check: Option<Instant>,
        now: impl Fn() -> Instant,
        mut query: impl FnMut() -> Vec<T>,
    ) -> Vec<T> {
        for _ in 0..2 {
            let generation = rank.control.generation();
            let initial = {
                let state = self.state.lock();
                if generation == 0
                    || state.hash_fences.contains_key(&rank.target.dp_rank)
                    || !self.trusted_at(&state, || first_trust_check.take().unwrap_or_else(&now))
                {
                    return Vec::new();
                }
                state.verified_at
            };
            let hits = query();
            if hits.is_empty() {
                return hits;
            }
            let state = self.state.lock();
            let end = now();
            if state.hash_fences.contains_key(&rank.target.dp_rank)
                || !self.trusted_at(&state, || end)
            {
                return Vec::new();
            }
            // The same source and the original grant must cover the observation.
            // Renewal alone cannot certify pages observed during an expiry gap.
            if rank.control.generation() == generation
                && (!self.layout || end.saturating_duration_since(initial) <= LAYOUT_FRESHNESS)
            {
                return hits;
            }
        }
        Vec::new()
    }
    pub(super) fn expire_if_needed(&self, state: &mut LifecycleState) -> bool {
        self.layout
            && state.verified_at.elapsed() > LAYOUT_FRESHNESS
            && std::mem::replace(&mut state.trusted, false)
    }

    pub(super) fn refresh_due(&self) -> bool {
        let state = self.state.lock();
        state.checked.is_none_or(|checked| {
            let interval = if self.all_publishers_fenced(&state)
                || (state.info.is_some() && (!self.layout || self.trusted(&state)))
            {
                REFRESH_INTERVAL
            } else {
                CONTROL_TIMEOUT
            };
            checked.elapsed() >= interval
        })
    }

    pub(super) fn layout_rebind_due(&self) -> bool {
        if !self.layout {
            return false;
        }
        let state = self.state.lock();
        !state.trusted
            && !self.all_publishers_fenced(&state)
            && state
                .checked
                .is_none_or(|checked| checked.elapsed() >= CONTROL_TIMEOUT)
    }

    fn all_publishers_fenced(&self, state: &LifecycleState) -> bool {
        self.active.load(Ordering::Acquire)
            && state.subscriptions.is_empty()
            && !state.hash_fences.is_empty()
    }

    /// Descriptor probes run every 30s; three failures retire an installed owner.
    pub(super) fn observe_descriptor(
        &self,
        worker: &str,
        descriptor: Option<&WorkerDescriptor>,
        configs: Option<&EngineCacheConfigs>,
    ) -> SourceAction {
        let mut closing = Vec::new();
        let mut state = self.state.lock();
        state.checked = Some(Instant::now());
        state.fence_incompatible_ranks(worker, configs, &mut closing);
        let keep = descriptor.is_some_and(|descriptor| {
            if self.all_publishers_fenced(&state) {
                return true;
            }
            // The layout monitor owns replay recovery for surviving ranks.
            let has_fence = self.active.load(Ordering::Acquire) && !state.hash_fences.is_empty();
            state.info.is_some()
                && (!self.layout || self.trusted(&state) || has_fence)
                && state.descriptor.as_ref().is_some_and(|previous| {
                    previous.matches_unfenced(descriptor, &state.hash_fences)
                })
        });
        match descriptor {
            Some(_) if keep => {
                state.probe_failures = 0;
                SourceAction::Keep
            }
            Some(_) => {
                state.info = None;
                state.base_models.clear();
                state.trusted = false;
                state.revision = crate::kv_index::next_owner_incarnation();
                SourceAction::Rebuild
            }
            None if state.info.is_some() => {
                state.probe_failures = state.probe_failures.saturating_add(1);
                if state.probe_failures < MAX_PROBE_FAILURES {
                    SourceAction::Keep
                } else {
                    state.info = None;
                    state.base_models.clear();
                    state.trusted = false;
                    state.probe_failures = 0;
                    state.revision = crate::kv_index::next_owner_incarnation();
                    SourceAction::Stop
                }
            }
            None => SourceAction::Keep,
        }
    }

    /// A fast layout probe cannot advance the descriptor failure counter.
    pub(super) fn observe_layout(
        &self,
        state: &mut LifecycleState,
        outcome: &StaticHybridMatch,
    ) -> SourceAction {
        if !self.active.load(Ordering::Acquire) || !probe_keeps_trust(outcome) {
            self.stop_layout(state);
            return SourceAction::Stop;
        }
        if probe_renews_trust(outcome) {
            state.verified_at = Instant::now();
            if !state.trusted {
                return SourceAction::Recover;
            }
        }
        SourceAction::Keep
    }

    fn stop_layout(&self, state: &mut LifecycleState) {
        if self.active.swap(false, Ordering::AcqRel) {
            state.checked = None;
        }
        state.trusted = false;
        state.base_models.clear();
        if state.info.take().is_some() {
            state.revision = crate::kv_index::next_owner_incarnation();
        }
    }

    pub(super) fn finish_layout_recovery(
        &self,
        state: &mut LifecycleState,
        recovered: bool,
    ) -> bool {
        if recovered && self.active.load(Ordering::Acquire) {
            state.trusted = true;
            return true;
        }
        self.stop_layout(state);
        false
    }
}

impl LifecycleState {
    fn fence_incompatible_ranks(
        &mut self,
        worker: &str,
        configs: Option<&EngineCacheConfigs>,
        closing: &mut Vec<EventSubscriberHandle>,
    ) {
        let Some(capability) = self.descriptor.as_ref().and_then(WorkerDescriptor::layout) else {
            return;
        };
        for (_, _, handle) in self.subscriptions.extract_if(.., |(source, _, _)| {
            let rank = source.dp_rank;
            if !runtime_hash_mismatch(configs, capability, worker, rank) {
                return false;
            }
            self.hash_fences.insert(rank, (&*source).into());
            if let Some(info) = &mut self.info {
                info.ranks.retain(|&active| active != rank);
            }
            true
        }) {
            closing.extend(handle);
        }
    }
}

pub(super) struct LayoutWorkerProvider {
    /// Rank lookups share lifecycle evidence but never another rank's pages.
    pub(super) indexes: HashMap<u32, Arc<SourceBinding>, foldhash::fast::RandomState>,
    pub(super) lifecycle: Arc<WorkerLifecycle>,
}

impl crate::policies::TieredMatchProvider for LayoutWorkerProvider {
    fn shared_prefix_lookup(&self, rank: u32) -> Option<crate::kv_index::SharedPrefixLookup> {
        self.indexes.get(&rank)?.shared_prefix_lookup(rank)
    }

    fn target_scoped(&self) -> bool {
        true
    }

    fn nixl_decode_credit(&self, rank: u32, prompt_len: usize, reused: u64) -> u64 {
        self.indexes
            .get(&rank)
            .map_or(reused, |index| index.nixl_decode_credit(prompt_len, reused))
    }

    fn matches(
        &self,
        tokens: &[u32],
        block_size: u32,
        lora: Option<&str>,
        dp_rank: u32,
    ) -> Vec<crate::kv_index::TierMatch> {
        self.matches_with_cache(
            tokens,
            block_size,
            lora,
            dp_rank,
            &mut crate::kv_index::RequestHashCache::for_single_target(),
        )
    }

    fn matches_with_cache<'a>(
        &self,
        tokens: &'a [u32],
        block_size: u32,
        lora: Option<&str>,
        dp_rank: u32,
        cache: &mut crate::kv_index::RequestHashCache<'a>,
    ) -> Vec<crate::kv_index::TierMatch> {
        let Some(index) = self.indexes.get(&dp_rank) else {
            return Vec::new();
        };
        let first_trust_check = self
            .lifecycle
            .layout
            .then(|| *cache.first_trust_check.get_or_insert_with(Instant::now));
        self.lifecycle
            .read_rank(index, first_trust_check, Instant::now, || {
                crate::policies::TieredMatchProvider::matches_with_cache(
                    index.as_ref(),
                    tokens,
                    block_size,
                    lora,
                    dp_rank,
                    cache,
                )
            })
    }
}

pub(super) type Subscription = (
    EventSource,
    Arc<SourceBinding>,
    Option<EventSubscriberHandle>,
);

impl KvIndexSupervisor {
    pub(super) fn monitor_hybrid_source(
        &self,
        lifecycle: &Arc<WorkerLifecycle>,
        worker_url: &str,
        capability: CacheCapabilities,
        assumed_plain_full: bool,
    ) {
        let lifecycle = Arc::downgrade(lifecycle);
        let probe_ctx = self.ctx.clone();
        let client = self.client.clone();
        let signals = Some(self.signal_tx.clone());
        let worker = worker_url.to_owned();
        let api_key = self.api_key.clone();
        let static_config = self.resolved_static_source(worker_url);
        let installed = Arc::downgrade(&self.installed);
        let strict_recovery = self.strict_recovery;
        // P/D already checks the labels during its steady-state descriptor refresh.
        let periodic_hash_probe = !self.http_pd;
        tokio::spawn(async move {
            let invalidate = |targets: &mut [Subscription], closing: &mut Vec<_>| {
                for (source, _, handle) in targets {
                    if let Some(handle) = handle.take() {
                        closing.push(handle);
                        crate::kv_index::subscriber::emit(
                            &signals,
                            IngestionSignal::Invalidated {
                                source: source.source.clone(),
                                dp_rank: source.dp_rank,
                                reason: "cache_contract",
                            },
                        );
                    }
                }
            };
            loop {
                tokio::time::sleep(Duration::from_secs(2)).await;
                let Some(lifecycle) = lifecycle.upgrade() else {
                    break;
                };
                let hash_due = {
                    let mut closing = Vec::new();
                    let mut state = lifecycle.state.lock();
                    if !lifecycle.active.load(Ordering::Acquire)
                        || state.info.is_none()
                        || state.subscriptions.is_empty()
                    {
                        break;
                    }
                    if lifecycle.expire_if_needed(&mut state) {
                        invalidate(&mut state.subscriptions, &mut closing);
                    }
                    periodic_hash_probe && state.hash_checked.elapsed() >= REFRESH_INTERVAL
                };
                let probe_hash = || {
                    probe_engine_cache_configs(
                        &client,
                        super::split_worker_rank(&worker).0,
                        api_key.as_deref(),
                        HASH_PROBE_BUDGET,
                    )
                };
                let (observed, mut configs) = tokio::join!(
                    probe_bound_sources(
                        &client,
                        &worker,
                        api_key.as_deref(),
                        static_config.is_some()
                    ),
                    async {
                        if hash_due {
                            probe_hash().await
                        } else {
                            None
                        }
                    },
                );
                let matches_active = |state: &LifecycleState| {
                    observed_sources_match(
                        &worker,
                        state.subscriptions.iter().map(|(source, _, _)| source),
                        &state.hash_fences,
                        (static_config.is_none() && !assumed_plain_full).then_some(&capability),
                        &observed,
                    )
                };
                // Check for a hash mismatch before retiring the layout on a source change.
                let probe_changed_source = !hash_due && {
                    let state = lifecycle.state.lock();
                    !state.subscriptions.is_empty()
                        && matches_active(&state) == StaticHybridMatch::Differs
                };
                if probe_changed_source {
                    configs = probe_hash().await;
                }
                // Drop retired streams after releasing the lifecycle lock.
                let mut closing = Vec::new();
                let mut state = lifecycle.state.lock();
                if !lifecycle.active.load(Ordering::Acquire)
                    || state.info.is_none()
                    || state.subscriptions.is_empty()
                {
                    break;
                }
                // A probe may cross the deadline while it is in flight.
                if lifecycle.expire_if_needed(&mut state) {
                    invalidate(&mut state.subscriptions, &mut closing);
                }
                if hash_due || probe_changed_source {
                    state.hash_checked = Instant::now();
                }
                state.fence_incompatible_ranks(&worker, configs.as_ref(), &mut closing);
                if state.subscriptions.is_empty() {
                    break;
                }
                let outcome = matches_active(&state);
                match lifecycle.observe_layout(&mut state, &outcome) {
                    SourceAction::Stop => {}
                    SourceAction::Recover => {
                        let mut recovered = true;
                        for (ev, index, current) in &mut state.subscriptions {
                            let stream = Arc::new(
                                EventStream::new(index.clone())
                                    .with_strict_recovery(strict_recovery),
                            );
                            match super::super::events::event_stream::spawn_zmq_with_signals(
                                ev.clone(),
                                stream,
                                probe_ctx.clone(),
                                signals.clone(),
                            ) {
                                Ok(handle) => closing.extend(current.replace(handle)),
                                Err(_) => recovered = false,
                            }
                        }
                        // Retirement can race the awaited probe and the new
                        // subscriptions. Never make it active again.
                        if lifecycle.finish_layout_recovery(&mut state, recovered) {
                            continue;
                        }
                    }
                    SourceAction::Keep => continue,
                    SourceAction::Rebuild => {
                        unreachable!("layout probe cannot install a descriptor")
                    }
                }
                invalidate(&mut state.subscriptions, &mut closing);
                drop(state);
                drop(closing);
                if let Some(installed) = installed.upgrade() {
                    installed.retire_if_current(&worker, &lifecycle);
                }
                break;
            }
        });
    }
}

impl KvIndexSupervisor {
    /// Shut down this worker's subscribers, clear its residency, release its
    /// identity ref.
    pub fn on_worker_removed(&self, worker_url: &str) {
        if let Some(observers) = &self.observers {
            observers.lock().remove(worker_url);
            return;
        }
        let mut workers = self.installed.workers.lock();
        if let Some(lifecycle) = self.lifecycles.lock().remove(worker_url) {
            lifecycle.active.store(false, Ordering::Release);
        }
        if let Some(subs) = workers.remove(worker_url) {
            self.installed.retire_worker(subs);
            info!("kv_index {}: removed subscribers", worker_url);
        }
    }

    /// Shut down every subscriber and drop all indexers.
    pub fn shutdown(&self) {
        if let Some(observers) = &self.observers {
            observers.lock().clear();
            return;
        }
        let mut workers = self.installed.workers.lock();
        for (_url, subs) in workers.drain() {
            self.installed.retire_worker(subs);
        }
        for (_, lifecycle) in self.lifecycles.lock().drain() {
            lifecycle.active.store(false, Ordering::Release);
        }
        drop(workers);
        info!("kv_index supervisor shut down");
    }
}

impl InstalledWorkers {
    pub(super) fn retire_if_current(&self, worker_url: &str, lifecycle: &Arc<WorkerLifecycle>) {
        lifecycle.active.store(false, Ordering::Release);
        let mut workers = self.workers.lock();
        if workers
            .get(worker_url)
            .is_none_or(|worker| !Arc::ptr_eq(&worker.provider.lifecycle, lifecycle))
        {
            return;
        }
        if let Some(subs) = workers.remove(worker_url) {
            self.retire_worker(subs);
        }
    }

    pub(super) fn retire_worker(&self, subs: WorkerSubs) {
        subs.active.store(false, Ordering::Release);
        let subscriptions = std::mem::take(&mut subs.provider.lifecycle.state.lock().subscriptions);
        drop(subscriptions);
        for live in &subs.live_sources {
            let _guard = live.lease.gate.lock();
            subs.indexer.retire_source(&subs.source_id, live.rank);
        }
        self.release_identity(&subs.key);
    }

    /// Ref-count an identity; on first acquisition create its indexer.
    pub(super) fn acquire_identity(&self, key: &CacheKey) -> Arc<KvBlockIndexer> {
        let mut refs = self.identity_refs.lock();
        let count = refs.entry(key.clone()).or_insert(0);
        if *count == 0 {
            self.indexers
                .insert(key.clone(), Arc::new(KvBlockIndexer::new()));
        }
        *count += 1;
        self.indexers.get(key).expect("just inserted").clone()
    }

    /// Decrement an identity's ref-count; on last release drop its indexer.
    pub(super) fn release_identity(&self, key: &CacheKey) {
        let mut refs = self.identity_refs.lock();
        if let Some(c) = refs.get_mut(key) {
            *c = c.saturating_sub(1);
            if *c == 0 {
                refs.remove(key);
                self.indexers.remove(key);
            }
        }
    }
}
