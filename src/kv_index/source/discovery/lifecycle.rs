//! Worker trust, source retirement and replay recovery.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tracing::info;

use super::probe::WorkerDescriptor;
use super::reconcile::{
    native_full_sources_match, probe_keeps_trust, probe_renews_trust, static_hybrid_sources_match,
    BoundSource, BoundSources, StaticHybridMatch,
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
            if !self.trusted_at(&state, || end) {
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
    pub(super) fn expire_if_needed(&self) -> bool {
        let mut state = self.state.lock();
        self.layout
            && state.verified_at.elapsed() > LAYOUT_FRESHNESS
            && std::mem::replace(&mut state.trusted, false)
    }

    pub(super) fn refresh_due(&self) -> bool {
        let state = self.state.lock();
        state.checked.is_none_or(|checked| {
            let interval = if state.info.is_some() && (!self.layout || self.trusted(&state)) {
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
            && state
                .checked
                .is_none_or(|checked| checked.elapsed() >= CONTROL_TIMEOUT)
    }

    /// Descriptor probes run every 30s; three failures retire an installed owner.
    pub(super) fn observe_descriptor(&self, descriptor: Option<&WorkerDescriptor>) -> SourceAction {
        let mut state = self.state.lock();
        state.checked = Some(Instant::now());
        match descriptor {
            Some(descriptor)
                if state.info.is_some()
                    && state.descriptor.as_ref() == Some(descriptor)
                    && (!self.layout || self.trusted(&state)) =>
            {
                state.probe_failures = 0;
                SourceAction::Keep
            }
            Some(_) => {
                state.info = None;
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
    pub(super) fn observe_layout(&self, outcome: &StaticHybridMatch) -> SourceAction {
        let mut state = self.state.lock();
        if !self.active.load(Ordering::Acquire) || !probe_keeps_trust(outcome) {
            if self.active.swap(false, Ordering::AcqRel) {
                state.checked = None;
            }
            state.trusted = false;
            state.base_models.clear();
            if state.info.take().is_some() {
                state.revision = crate::kv_index::next_owner_incarnation();
            }
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
}

pub(super) struct LayoutWorkerProvider {
    /// Rank lookups share lifecycle evidence but never another rank's pages.
    pub(super) indexes: HashMap<u32, Arc<SourceBinding>, foldhash::fast::RandomState>,
    pub(super) lifecycle: Arc<WorkerLifecycle>,
}

impl crate::policies::TieredMatchProvider for LayoutWorkerProvider {
    fn target_scoped(&self) -> bool {
        true
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

impl KvIndexSupervisor {
    pub(super) fn monitor_hybrid_source(
        &self,
        provider: &Arc<LayoutWorkerProvider>,
        worker_url: &str,
        mut handles: Vec<Arc<EventSubscriberHandle>>,
        targets: Vec<(EventSource, Arc<SourceBinding>)>,
        capability: CacheCapabilities,
        assumed_plain_full: bool,
    ) {
        // Revalidate only publishers with an installed subscriber.
        let bound: BoundSources = targets
            .iter()
            .map(|(source, _)| {
                (
                    source.dp_rank,
                    BoundSource {
                        endpoint: source.pub_endpoint.clone(),
                        replay_endpoint: source.replay_endpoint.clone(),
                        topic: source.topic.clone(),
                    },
                )
            })
            .collect();
        let provider = Arc::downgrade(provider);
        let probe_ctx = self.ctx.clone();
        let client = self.client.clone();
        let signals = Some(self.signal_tx.clone());
        let worker = worker_url.to_owned();
        let api_key = self.api_key.clone();
        let static_config = self.resolved_static_source(worker_url);
        let installed = Arc::downgrade(&self.installed);
        let strict_recovery = self.strict_recovery;
        tokio::spawn(async move {
            let invalidate = |handles: &[Arc<EventSubscriberHandle>]| {
                for handle in handles {
                    handle.shutdown();
                }
                for (source, _) in &targets {
                    crate::kv_index::subscriber::emit(
                        &signals,
                        IngestionSignal::Invalidated {
                            source: source.source.clone(),
                            dp_rank: source.dp_rank,
                            reason: "cache_contract",
                        },
                    );
                }
            };
            loop {
                tokio::time::sleep(Duration::from_secs(2)).await;
                let Some(provider) = provider.upgrade() else {
                    break;
                };
                if !provider.lifecycle.active.load(Ordering::Acquire) {
                    break;
                }
                if provider.lifecycle.expire_if_needed() {
                    invalidate(&handles);
                }
                let outcome = if let Some(config) = &static_config {
                    static_hybrid_sources_match(
                        &client,
                        &worker,
                        api_key.as_deref(),
                        config,
                        Some(&bound),
                    )
                    .await
                } else {
                    native_full_sources_match(
                        &client,
                        &worker,
                        api_key.as_deref(),
                        (!assumed_plain_full).then_some(&capability),
                        &bound,
                    )
                    .await
                };
                if !provider.lifecycle.active.load(Ordering::Acquire) {
                    for handle in &handles {
                        handle.shutdown();
                    }
                    break;
                }
                // A probe may cross the deadline while it is in flight.
                if provider.lifecycle.expire_if_needed() {
                    invalidate(&handles);
                }
                match provider.lifecycle.observe_layout(&outcome) {
                    SourceAction::Stop => {
                        invalidate(&handles);
                        if let Some(installed) = installed.upgrade() {
                            installed.retire_if_current(&worker, &provider.lifecycle);
                        }
                        break;
                    }
                    SourceAction::Recover => {
                        handles.clear();
                        let mut recovered = true;
                        for (ev, index) in &targets {
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
                                Ok(handle) => handles.push(Arc::new(handle)),
                                Err(_) => recovered = false,
                            }
                        }
                        // Retirement can race the awaited probe and the new
                        // subscriptions. Never make it active again.
                        if !provider.lifecycle.active.load(Ordering::Acquire) || !recovered {
                            for handle in &handles {
                                handle.shutdown();
                            }
                            if !provider.lifecycle.active.load(Ordering::Acquire) {
                                break;
                            }
                        } else {
                            provider.lifecycle.state.lock().trusted = true;
                        }
                    }
                    SourceAction::Keep => {}
                    SourceAction::Rebuild => {
                        unreachable!("layout probe cannot install a descriptor")
                    }
                }
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
        self.lifecycles.lock().remove(worker_url);
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
        self.lifecycles.lock().clear();
        drop(workers);
        info!("kv_index supervisor shut down");
    }
}

impl InstalledWorkers {
    pub(super) fn retire_if_current(&self, worker_url: &str, lifecycle: &Arc<WorkerLifecycle>) {
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
        for handle in &subs.handles {
            handle.shutdown();
        }
        for live in &subs.live_sources {
            let _guard = live.lease.strict_rank.lock();
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
    fn release_identity(&self, key: &CacheKey) {
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
