//! Exact KV-placement and live-load aware worker selection.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use http::HeaderMap;
use lru::LruCache;
use parking_lot::Mutex;
use rand::Rng;
use serde_json::Value as JsonValue;
use sha2::{Digest as _, Sha256};

use super::{get_healthy_worker_indices, LoadBalancingPolicy, RequestHeaders};
use crate::core::Worker;
use crate::kv_index::{CacheKey, MatchQuery, RoutableTarget, StorageTier, TierMatch, WorkerKvInfo};
use crate::request_identity::{context_string, resolve_stable_identity};

/// Read-only placement port owned by its policy consumer.
pub trait TieredMatchProvider: Send + Sync {
    fn find_tiered_matches(&self, query: &MatchQuery) -> Vec<TierMatch>;
}

/// Exact prompt sequences rendered by each HTTP backend. DP ranks share their
/// base URL's input; separate prompts in a batch never form one prefix chain.
pub type WorkerTokenInputs = HashMap<String, Arc<[Vec<u32>]>>;

#[derive(Clone, Copy)]
pub enum KvRequestTokens<'a> {
    Shared(Option<&'a [u32]>),
    PerWorker(&'a WorkerTokenInputs),
}

impl<'a> KvRequestTokens<'a> {
    fn prompts_for(self, worker: &dyn Worker) -> Vec<&'a [u32]> {
        match self {
            Self::Shared(ids) => ids.into_iter().collect(),
            Self::PerWorker(inputs) => inputs
                .get(worker.base_url())
                .into_iter()
                .flat_map(|prompts| prompts.iter().map(Vec::as_slice))
                .collect(),
        }
    }
}

/// Optional operations used by adapters with tokenized requests and KV discovery.
pub trait KvAwareOps: Send + Sync {
    fn register_worker(
        &self,
        worker: &dyn Worker,
        info: Option<&WorkerKvInfo>,
        matches: Option<Arc<dyn TieredMatchProvider>>,
    );
    fn unregister_worker(&self, worker_url: &str);
    /// Select and reserve atomically: the returned receipt already carries the
    /// request slot and worker-load reservation made in the policy's short
    /// critical section, plus any affinity binding reference. Workers in
    /// `excluded_workers` already failed an attempt of this request and are
    /// pick constraints only: a still-valid binding to one is set aside, not
    /// invalidated.
    fn select_worker_with_receipt(
        &self,
        workers: &[Arc<dyn Worker>],
        token_ids: Option<&[u32]>,
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
        affinity_key: Option<&AffinityKey>,
        excluded_workers: &[String],
    ) -> Option<KvSelectionReceipt>;

    fn select_worker_with_inputs(
        &self,
        workers: &[Arc<dyn Worker>],
        tokens: KvRequestTokens<'_>,
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
        affinity_key: Option<&AffinityKey>,
        excluded_workers: &[String],
    ) -> Option<KvSelectionReceipt>;
}

/// Resolve the optional capability at the policy boundary. Keep concrete policy
/// discovery here so adapters and LoadBalancingPolicy stay independent of it.
pub fn kv_aware_ops(policy: &dyn LoadBalancingPolicy) -> Option<&dyn KvAwareOps> {
    policy
        .as_any()
        .downcast_ref::<KvAwarePolicy>()
        .map(|policy| policy as &dyn KvAwareOps)
}

/// Shared request accounting handle. The adapter signals output and completion;
/// concurrent body/producer cleanup updates the counters only once.
#[derive(Clone)]
pub struct KvRequestSlot {
    manager: Arc<dyn SlotManager>,
    target: RoutableTarget,
    prefill_tokens: u64,
    /// Affinity binding reference acquired with this reservation, if any.
    affinity: Option<Arc<AffinityRecord>>,
    // 0 = prefill, 1 = decode, 2 = freed.
    state: Arc<AtomicU8>,
}

impl KvRequestSlot {
    pub(crate) fn new(
        manager: Arc<dyn SlotManager>,
        target: RoutableTarget,
        prefill_tokens: u64,
        affinity: Option<Arc<AffinityRecord>>,
    ) -> Self {
        Self {
            manager,
            target,
            prefill_tokens,
            affinity,
            state: Arc::new(AtomicU8::new(0)),
        }
    }

    pub fn begin_decode(&self) {
        if self
            .state
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.manager
                .begin_decode(&self.target, &self.target, self.prefill_tokens);
        }
    }

    pub fn release(&self) {
        self.begin_decode();
        if self
            .state
            .compare_exchange(1, 2, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.manager.free(&self.target);
            if let Some(record) = &self.affinity {
                record.release_ref();
            }
        }
    }
}

impl std::fmt::Debug for KvRequestSlot {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("KvRequestSlot")
            .field("target", &self.target)
            .field("prefill_tokens", &self.prefill_tokens)
            .field("state", &self.state.load(Ordering::Relaxed))
            .finish()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LoadSample {
    pub active_prefill_tokens: u64,
    pub active_requests: usize,
}

pub trait LoadProvider: Send + Sync {
    fn sample(&self, target: &RoutableTarget) -> LoadSample;
}

pub trait SlotManager: LoadProvider {
    fn begin_prefill(&self, target: &RoutableTarget, prefill_tokens: u64);
    fn begin_decode(
        &self,
        prefill_target: &RoutableTarget,
        decode_target: &RoutableTarget,
        prefill_tokens: u64,
    );
    fn free(&self, target: &RoutableTarget);

    /// Drop idle state for a target that is no longer registered. Implementers
    /// may keep active state until its matching request cleanup arrives.
    fn unregister_target(&self, _target: &RoutableTarget) {}
}

#[derive(Debug, Default)]
struct SlotCounters {
    active_prefill_tokens: AtomicU64,
    active_requests: AtomicUsize,
}

/// Router-local, bracket-based load view keyed at the index's `(worker, rank)`
/// granularity. Saturating decrements make duplicate cleanup harmless.
#[derive(Debug, Default)]
pub struct InMemorySlotManager {
    slots: DashMap<RoutableTarget, Arc<SlotCounters>>,
}

impl InMemorySlotManager {
    pub fn new() -> Self {
        Self::default()
    }

    fn update_or_insert(&self, target: &RoutableTarget, update: impl FnOnce(&SlotCounters)) {
        let slot = self
            .slots
            .entry(target.clone())
            .or_insert_with(|| Arc::new(SlotCounters::default()));
        update(&slot);
    }

    fn prune_if_idle(&self, target: &RoutableTarget) {
        if let dashmap::mapref::entry::Entry::Occupied(entry) = self.slots.entry(target.clone()) {
            let slot = entry.get();
            if slot.active_requests.load(Ordering::Acquire) == 0
                && slot.active_prefill_tokens.load(Ordering::Acquire) == 0
            {
                // The occupied entry keeps the shard locked between checking
                // the counters and removal, so a concurrent begin cannot be
                // detached from the map.
                entry.remove();
            }
        }
    }
}

fn atomic_saturating_sub_u64(value: &AtomicU64, amount: u64) {
    let _ = value.try_update(Ordering::AcqRel, Ordering::Acquire, |current| {
        Some(current.saturating_sub(amount))
    });
}

fn atomic_saturating_sub_usize(value: &AtomicUsize, amount: usize) {
    let _ = value.try_update(Ordering::AcqRel, Ordering::Acquire, |current| {
        Some(current.saturating_sub(amount))
    });
}

impl LoadProvider for InMemorySlotManager {
    fn sample(&self, target: &RoutableTarget) -> LoadSample {
        let Some(slot) = self.slots.get(target) else {
            return LoadSample::default();
        };
        LoadSample {
            active_prefill_tokens: slot.active_prefill_tokens.load(Ordering::Acquire),
            active_requests: slot.active_requests.load(Ordering::Acquire),
        }
    }
}

impl SlotManager for InMemorySlotManager {
    fn begin_prefill(&self, target: &RoutableTarget, prefill_tokens: u64) {
        self.update_or_insert(target, |slot| {
            slot.active_prefill_tokens
                .fetch_add(prefill_tokens, Ordering::AcqRel);
            slot.active_requests.fetch_add(1, Ordering::AcqRel);
        });
    }

    fn begin_decode(
        &self,
        prefill_target: &RoutableTarget,
        decode_target: &RoutableTarget,
        prefill_tokens: u64,
    ) {
        if prefill_target == decode_target {
            self.update_or_insert(prefill_target, |prefill| {
                atomic_saturating_sub_u64(&prefill.active_prefill_tokens, prefill_tokens);
            });
        } else {
            self.update_or_insert(prefill_target, |prefill| {
                atomic_saturating_sub_u64(&prefill.active_prefill_tokens, prefill_tokens);
                atomic_saturating_sub_usize(&prefill.active_requests, 1);
            });
            self.update_or_insert(decode_target, |decode| {
                decode.active_requests.fetch_add(1, Ordering::AcqRel);
            });
            self.prune_if_idle(prefill_target);
        }
    }

    fn free(&self, target: &RoutableTarget) {
        if let Some(slot) = self.slots.get(target) {
            atomic_saturating_sub_usize(&slot.active_requests, 1);
        }
        self.prune_if_idle(target);
    }

    fn unregister_target(&self, target: &RoutableTarget) {
        self.prune_if_idle(target);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransferCredits {
    pub device: f64,
    pub host_pinned: f64,
    pub external: f64,
    pub disk: f64,
}

impl Default for TransferCredits {
    fn default() -> Self {
        Self {
            device: 1.0,
            host_pinned: 0.8,
            external: 0.6,
            disk: 0.3,
        }
    }
}

impl TransferCredits {
    fn for_tier(self, tier: StorageTier) -> f64 {
        match tier {
            StorageTier::Device => self.device,
            StorageTier::HostPinned => self.host_pinned,
            StorageTier::External => self.external,
            StorageTier::Disk => self.disk,
        }
    }

    fn valid(self) -> bool {
        self.device == 1.0
            && [self.host_pinned, self.external, self.disk]
                .into_iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(&value))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostParams {
    pub router_temperature: f64,
    pub transfer_credits: TransferCredits,
    pub block_size: u32,
}

impl Default for CostParams {
    fn default() -> Self {
        Self {
            router_temperature: 0.0,
            transfer_credits: TransferCredits::default(),
            block_size: 16,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CandidateView<'a> {
    pub target: &'a RoutableTarget,
    pub matches: &'a [TierMatch],
    pub load: LoadSample,
    pub input_tokens: usize,
    /// Sum of independently matched batch prompts, when this is a batch.
    pub prefill_tokens: Option<u64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CostComponents {
    pub ptoken: u64,
    /// Current active request count (max of the slot-manager sample and worker load).
    pub batchsize: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CostDecision {
    pub target: RoutableTarget,
    pub cost: f64,
    pub components: CostComponents,
}

/// Result of an admitted KV-aware selection. Worker load is already reserved;
/// the router owns its release together with the optional request slot.
#[derive(Debug, Clone)]
pub struct KvSelectionReceipt {
    pub worker_index: usize,
    pub new_prefill_tokens: u64,
    pub decision: CostDecision,
    pub slot: Option<KvRequestSlot>,
}

/// Fixed bound of the affinity table shared by every policy instance.
const KV_AFFINITY_MAX_RECORDS: usize = 10_000;

const AFFINITY_DIGEST_DOMAIN: &str = "kv_affinity.v1";

/// Stable identity extracted at request entry for KV session affinity.
///
/// Reuses the shared stable identity parsing (canonical context, agent_hint,
/// framework headers) with a generic body `session_id` fallback, without
/// Program scheduling hints or constraints. The digest scope keeps identity
/// kind, model, LoRA, and cache salt distinct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffinityKey {
    model_scope: String,
    source: &'static str,
    identity: String,
}

impl AffinityKey {
    pub fn from_request(
        headers: Option<&HeaderMap>,
        request: Option<&JsonValue>,
        model_scope: &str,
    ) -> Option<Self> {
        match resolve_stable_identity(headers, request) {
            Ok(Some(identity)) => Some(Self {
                model_scope: model_scope.to_string(),
                source: identity.source.as_str(),
                identity: identity.program_id,
            }),
            Ok(None) => body_session_identity(request).map(|identity| Self {
                model_scope: model_scope.to_string(),
                source: "body_session",
                identity,
            }),
            Err(_) => None,
        }
    }

    fn scoped_digest(&self, lora_name: Option<&str>, cache_salt: Option<&str>) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for field in [
            AFFINITY_DIGEST_DOMAIN.as_bytes(),
            self.model_scope.as_bytes(),
            self.source.as_bytes(),
            self.identity.as_bytes(),
            lora_name.unwrap_or_default().as_bytes(),
            cache_salt.unwrap_or_default().as_bytes(),
        ] {
            hasher.update((field.len() as u64).to_le_bytes());
            hasher.update(field);
        }
        hasher.finalize().into()
    }
}

fn body_session_identity(request: Option<&JsonValue>) -> Option<String> {
    let object = request?.as_object()?;
    if let Some(params) = object
        .get("session_params")
        .filter(|value| !value.is_null())
    {
        if let Some(session_id) = context_string(params.as_object()?, &["session_id"]).ok()? {
            return Some(session_id);
        }
    }
    context_string(object, &["session_id"]).ok().flatten()
}

/// One affinity binding. Guards hold direct `Arc` references so a record
/// replaced mid-flight still releases its own in-flight count.
pub(crate) struct AffinityRecord {
    target: RoutableTarget,
    generation: u64,
    state: Mutex<AffinityState>,
    ttl: Option<Duration>,
}

struct AffinityState {
    in_flight: usize,
    idle_deadline: Option<Instant>,
}

impl AffinityRecord {
    fn release_ref(&self) {
        let mut state = self.state.lock();
        state.in_flight -= 1;
        if state.in_flight == 0 {
            state.idle_deadline = self.ttl.map(|ttl| Instant::now() + ttl);
        }
    }

    fn is_expired(&self, now: Instant) -> bool {
        let state = self.state.lock();
        state.in_flight == 0 && state.idle_deadline.is_some_and(|deadline| deadline <= now)
    }
}

/// Bounded `digest -> target` affinity table shared by the router's KV-aware
/// policies. Idle records are reclaimed LRU at capacity; when all records are
/// in flight, new keys bypass binding and still route normally.
#[derive(Debug)]
pub(crate) struct KvAffinityTable {
    records: Mutex<LruCache<[u8; 32], Arc<AffinityRecord>>>,
    ttl: Option<Duration>,
}

impl Default for KvAffinityTable {
    fn default() -> Self {
        Self::new()
    }
}

impl KvAffinityTable {
    fn new() -> Self {
        Self::with_ttl(None)
    }

    pub(crate) fn with_ttl(ttl: Option<Duration>) -> Self {
        Self {
            records: Mutex::new(LruCache::new(
                NonZeroUsize::new(KV_AFFINITY_MAX_RECORDS).expect("capacity is positive"),
            )),
            ttl,
        }
    }

    #[cfg(test)]
    fn get(&self, digest: &[u8; 32]) -> Option<Arc<AffinityRecord>> {
        self.records.lock().peek(digest).cloned()
    }

    /// Acquire a binding, incrementing its in-flight count.
    fn acquire(&self, digest: &[u8; 32]) -> Option<Arc<AffinityRecord>> {
        let mut records = self.records.lock();
        if records
            .peek(digest)
            .is_some_and(|record| record.is_expired(Instant::now()))
        {
            records.pop(digest);
            return None;
        }
        let record = records.get(digest)?.clone();
        record.state.lock().in_flight += 1;
        Some(record)
    }

    /// Establish (or replace) a binding with one in-flight reference held by
    /// the caller. `None` means all capacity is in use.
    fn bind(
        &self,
        digest: [u8; 32],
        target: RoutableTarget,
        generation: u64,
    ) -> Option<Arc<AffinityRecord>> {
        let mut records = self.records.lock();
        if !records.contains(&digest)
            && records.len() >= records.cap().get()
            && !self.evict_lru_idle(&mut records)
        {
            return None;
        }
        let record = Arc::new(AffinityRecord {
            target,
            generation,
            state: Mutex::new(AffinityState {
                in_flight: 1,
                idle_deadline: None,
            }),
            ttl: self.ttl,
        });
        records.put(digest, record.clone());
        Some(record)
    }

    fn remove_target(&self, target: &RoutableTarget) {
        let mut records = self.records.lock();
        let stale: Vec<[u8; 32]> = records
            .iter()
            .filter(|(_, record)| record.target == *target)
            .map(|(digest, _)| *digest)
            .collect();
        for digest in stale {
            records.pop(&digest);
        }
    }

    pub(crate) fn clear(&self) {
        self.records.lock().clear();
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.records.lock().len()
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn evict_lru_idle(&self, records: &mut LruCache<[u8; 32], Arc<AffinityRecord>>) -> bool {
        let lru_idle = records
            .iter()
            .filter(|(_, record)| record.state.lock().in_flight == 0)
            .map(|(digest, _)| *digest)
            .next_back();
        lru_idle.is_some_and(|digest| records.pop(&digest).is_some())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionError {
    InvalidParams,
    CostLengthMismatch,
    NonFiniteCost,
}

pub trait Scorer: Send + Sync + std::fmt::Debug {
    fn score(
        &self,
        candidates: &[CandidateView<'_>],
        params: &CostParams,
        costs: &mut [f64],
    ) -> Result<(), SelectionError>;
}

pub trait Picker: Send + Sync + std::fmt::Debug {
    fn pick(&self, costs: &[f64]) -> Option<usize>;
}

/// Device prefix plus one external continuation, matching vLLM's two-segment
/// reuse rule. The return value is in tokens and can be fractional because a
/// transfer penalty is a credit ratio.
fn credited_tokens(candidate: &CandidateView<'_>, params: &CostParams) -> f64 {
    let device_blocks = candidate
        .matches
        .iter()
        .filter(|hit| hit.tier == StorageTier::Device)
        .flat_map(|hit| hit.segments.iter().copied())
        .filter(|(start, _)| *start == 0)
        .map(|(_, end)| end)
        .max()
        .unwrap_or(0)
        // vLLM must recompute the last prompt token to obtain logits. Local
        // GPU prefix reuse is block-aligned, so never credit its final block
        // when that block ends at the prompt boundary.
        .min(
            candidate
                .input_tokens
                .saturating_sub(1)
                .checked_div(params.block_size as usize)
                .unwrap_or(0)
                .min(u32::MAX as usize) as u32,
        );

    let mut best_continuation = (0_u32, 0.0_f64);
    for hit in candidate
        .matches
        .iter()
        .filter(|hit| hit.tier != StorageTier::Device)
    {
        for &(start, end) in &hit.segments {
            if start <= device_blocks && device_blocks < end {
                let blocks = end - device_blocks;
                let credit = params.transfer_credits.for_tier(hit.tier);
                if blocks > best_continuation.0
                    || (blocks == best_continuation.0 && credit > best_continuation.1)
                {
                    best_continuation = (blocks, credit);
                }
            }
        }
    }

    let credited_blocks =
        f64::from(device_blocks) + f64::from(best_continuation.0) * best_continuation.1;
    // External reuse may have a different alignment, but it cannot eliminate
    // the last-token logits work either.
    (credited_blocks * f64::from(params.block_size))
        .min(candidate.input_tokens.saturating_sub(1) as f64)
}

fn request_prefill_tokens(candidate: &CandidateView<'_>, params: &CostParams) -> u64 {
    if let Some(tokens) = candidate.prefill_tokens {
        return tokens;
    }
    // Round remaining work up: a fractional transfer credit never erases a
    // whole token of compute that was not fully credited.
    ((candidate.input_tokens as f64 - credited_tokens(candidate, params)).max(0.0)).ceil() as u64
}

pub fn cost_components(candidate: &CandidateView<'_>, params: &CostParams) -> CostComponents {
    let new_prefill_tokens = request_prefill_tokens(candidate, params);
    CostComponents {
        ptoken: new_prefill_tokens.saturating_add(candidate.load.active_prefill_tokens),
        // Keep the observed count for tie-breaking. The scorer adds one to
        // account for the request being selected, which is not active yet.
        batchsize: candidate.load.active_requests,
    }
}

#[derive(Debug, Default)]
pub struct MultiplicativeScorer;

impl Scorer for MultiplicativeScorer {
    fn score(
        &self,
        candidates: &[CandidateView<'_>],
        params: &CostParams,
        costs: &mut [f64],
    ) -> Result<(), SelectionError> {
        if candidates.len() != costs.len() {
            return Err(SelectionError::CostLengthMismatch);
        }
        if params.block_size == 0
            || !params.router_temperature.is_finite()
            || params.router_temperature < 0.0
            || !params.transfer_credits.valid()
        {
            return Err(SelectionError::InvalidParams);
        }
        for (candidate, cost) in candidates.iter().zip(costs.iter_mut()) {
            let components = cost_components(candidate, params);
            let projected_batchsize = components.batchsize.saturating_add(1);
            let contribution = components.ptoken as f64 * projected_batchsize as f64;
            if !contribution.is_finite() {
                return Err(SelectionError::NonFiniteCost);
            }
            *cost += contribution;
            if !cost.is_finite() {
                return Err(SelectionError::NonFiniteCost);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TemperaturePicker {
    temperature: f64,
}

impl TemperaturePicker {
    pub fn new(temperature: f64) -> Self {
        Self { temperature }
    }
}

impl Picker for TemperaturePicker {
    fn pick(&self, costs: &[f64]) -> Option<usize> {
        if costs.is_empty() || costs.iter().any(|cost| !cost.is_finite()) {
            return None;
        }
        if self.temperature <= 0.0 || !self.temperature.is_finite() {
            return costs
                .iter()
                .enumerate()
                .min_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, _)| index);
        }

        let min = costs.iter().copied().reduce(f64::min)?;
        let weights: Vec<f64> = costs
            .iter()
            .map(|cost| (-(cost - min) / self.temperature).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return costs
                .iter()
                .enumerate()
                .min_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, _)| index);
        }
        let mut draw = rand::rng().random::<f64>() * total;
        for (index, weight) in weights.into_iter().enumerate() {
            if draw < weight {
                return Some(index);
            }
            draw -= weight;
        }
        Some(costs.len() - 1)
    }
}

#[derive(Debug, Default)]
struct EmptyMatchProvider;

impl TieredMatchProvider for EmptyMatchProvider {
    fn find_tiered_matches(&self, _query: &MatchQuery) -> Vec<TierMatch> {
        Vec::new()
    }
}

#[derive(Clone)]
struct TargetBinding {
    target: RoutableTarget,
    generation: u64,
    cache_key: Option<CacheKey>,
    matches: Arc<dyn TieredMatchProvider>,
    block_size: u32,
}

/// One prepared selection snapshot: KV queries fill `grouped` outside every
/// lock; the policy's short critical section then samples load, scores, and
/// reserves against this plan.
struct SelectionPlan {
    healthy: Vec<usize>,
    bindings: Vec<TargetBinding>,
    grouped: Vec<Vec<TierMatch>>,
    input_tokens: Vec<usize>,
    prefill_tokens: Vec<Option<u64>>,
    has_tokens: bool,
}

/// Registered `kv_aware` policy. Discovery binds each worker target to the
/// placement provider for its cache identity.
pub struct KvAwarePolicy {
    matches: Arc<dyn TieredMatchProvider>,
    load: Arc<dyn LoadProvider>,
    slot_manager: Option<Arc<dyn SlotManager>>,
    scorers: Vec<Arc<dyn Scorer>>,
    picker: Arc<dyn Picker>,
    params: CostParams,
    targets: DashMap<String, TargetBinding>,
    /// Cross-request affinity bindings, shared across the router's policies.
    affinity: Arc<KvAffinityTable>,
    /// Serializes load sampling, selection, and reservation.
    selection_lock: Mutex<()>,
    /// Every registration gets a new generation, including load-only targets.
    next_generation: AtomicU64,
}

impl std::fmt::Debug for KvAwarePolicy {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("KvAwarePolicy")
            .field("scorers", &self.scorers.len())
            .field("params", &self.params)
            .field("registered_targets", &self.targets.len())
            .finish()
    }
}

impl KvAwarePolicy {
    pub fn new(params: CostParams) -> Self {
        let slots = Arc::new(InMemorySlotManager::new());
        let mut policy = Self::with_components(
            Arc::new(EmptyMatchProvider),
            slots.clone(),
            vec![Arc::new(MultiplicativeScorer)],
            Arc::new(TemperaturePicker::new(params.router_temperature)),
            params,
        );
        policy.slot_manager = Some(slots);
        policy
    }

    /// Share one affinity table across per-model policy instances.
    pub(crate) fn with_shared_affinity(mut self, table: Arc<KvAffinityTable>) -> Self {
        self.affinity = table;
        self
    }

    pub fn with_components(
        matches: Arc<dyn TieredMatchProvider>,
        load: Arc<dyn LoadProvider>,
        scorers: Vec<Arc<dyn Scorer>>,
        picker: Arc<dyn Picker>,
        params: CostParams,
    ) -> Self {
        Self {
            matches,
            load,
            slot_manager: None,
            scorers,
            picker,
            params,
            targets: DashMap::new(),
            affinity: Arc::new(KvAffinityTable::new()),
            selection_lock: Mutex::new(()),
            next_generation: AtomicU64::new(1),
        }
    }

    pub fn register_target(&self, worker_url: impl Into<String>, target: RoutableTarget) {
        let _guard = self.selection_lock.lock();
        self.targets.insert(
            worker_url.into(),
            TargetBinding {
                target,
                generation: self.next_generation.fetch_add(1, Ordering::AcqRel),
                cache_key: None,
                matches: self.matches.clone(),
                block_size: self.params.block_size,
            },
        );
    }

    pub fn register_target_with_cache_identity(
        &self,
        worker_url: impl Into<String>,
        target: RoutableTarget,
        cache_key: CacheKey,
        matches: Arc<dyn TieredMatchProvider>,
    ) {
        let _guard = self.selection_lock.lock();
        let block_size = cache_key.block_size;
        self.targets.insert(
            worker_url.into(),
            TargetBinding {
                target,
                generation: self.next_generation.fetch_add(1, Ordering::AcqRel),
                cache_key: Some(cache_key),
                matches,
                block_size,
            },
        );
    }

    pub fn unregister_target(&self, worker_url: &str) {
        let _guard = self.selection_lock.lock();
        if let Some((_, binding)) = self.targets.remove(worker_url) {
            if let Some(slot_manager) = &self.slot_manager {
                slot_manager.unregister_target(&binding.target);
            }
            self.affinity.remove_target(&binding.target);
        }
    }

    pub fn slot_manager(&self) -> Option<Arc<dyn SlotManager>> {
        self.slot_manager.clone()
    }

    pub fn target_for_worker(&self, worker: &dyn Worker) -> RoutableTarget {
        self.targets
            .get(worker.url())
            .map(|entry| entry.target.clone())
            .unwrap_or_else(|| RoutableTarget {
                instance_id: Arc::from(worker.base_url()),
                dp_rank: worker.dp_rank().unwrap_or(0) as u32,
            })
    }

    fn binding_for_worker(&self, worker: &dyn Worker) -> TargetBinding {
        self.targets
            .get(worker.url())
            .map(|entry| entry.clone())
            .unwrap_or_else(|| TargetBinding {
                target: RoutableTarget {
                    instance_id: Arc::from(worker.base_url()),
                    dp_rank: worker.dp_rank().unwrap_or(0) as u32,
                },
                generation: 0,
                cache_key: None,
                matches: self.matches.clone(),
                block_size: self.params.block_size,
            })
    }

    pub fn local_hashes(token_ids: &[u32], block_size: u32) -> Vec<Arc<str>> {
        Self::local_hashes_with_lora(token_ids, block_size, None)
    }

    fn tiers_of_interest() -> Vec<StorageTier> {
        vec![
            StorageTier::Device,
            StorageTier::HostPinned,
            StorageTier::External,
            StorageTier::Disk,
        ]
    }

    fn local_hashes_with_lora(
        token_ids: &[u32],
        block_size: u32,
        lora_name: Option<&str>,
    ) -> Vec<Arc<str>> {
        crate::kv_index::local_block_hash_strings(token_ids, block_size, lora_name)
    }

    pub fn select_worker_with_token_ids(
        &self,
        workers: &[Arc<dyn Worker>],
        token_ids: Option<&[u32]>,
    ) -> Option<usize> {
        self.select_worker_pure(workers, token_ids, None, None)
            .map(|(worker_index, _)| worker_index)
    }

    /// Pure selection without reservation, used by the generic
    /// `LoadBalancingPolicy` entry points that account load themselves.
    fn select_worker_pure(
        &self,
        workers: &[Arc<dyn Worker>],
        token_ids: Option<&[u32]>,
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
    ) -> Option<(usize, CostDecision)> {
        let plan = self.prepare_selection(
            workers,
            KvRequestTokens::Shared(token_ids),
            lora_name,
            cache_salt,
        )?;
        let candidates = self.candidate_views(workers, &plan);
        let (costs, components) = self.score_candidates(&plan, &candidates)?;
        let position = self.pick_position(plan.has_tokens, &costs, &components, None)?;
        let (worker_index, _, decision) =
            self.decision_for_position(&plan, &candidates, &costs, &components, position);
        Some((worker_index, decision))
    }

    /// Select, reserve, and bind atomically. KV queries run outside every
    /// lock; the short critical section rechecks the affinity table, samples
    /// fresh load, picks (or reuses the bound target), and reserves slot and
    /// worker load. Ordinary load changes never migrate a valid binding, and
    /// a per-request retry exclusion never rewrites one.
    pub fn select_worker_with_receipt(
        &self,
        workers: &[Arc<dyn Worker>],
        token_ids: Option<&[u32]>,
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
        affinity_key: Option<&AffinityKey>,
        excluded_workers: &[String],
    ) -> Option<KvSelectionReceipt> {
        self.select_worker_with_inputs(
            workers,
            KvRequestTokens::Shared(token_ids),
            lora_name,
            cache_salt,
            affinity_key,
            excluded_workers,
        )
    }

    pub fn select_worker_with_inputs(
        &self,
        workers: &[Arc<dyn Worker>],
        tokens: KvRequestTokens<'_>,
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
        affinity_key: Option<&AffinityKey>,
        excluded_workers: &[String],
    ) -> Option<KvSelectionReceipt> {
        let digest = affinity_key.map(|key| key.scoped_digest(lora_name, cache_salt));
        loop {
            let plan = self.prepare_selection(workers, tokens, lora_name, cache_salt)?;
            let excluded = |position: usize| {
                excluded_workers
                    .iter()
                    .any(|url| workers[plan.healthy[position]].url() == url.as_str())
            };

            let guard = self.selection_lock.lock();
            let stale = plan
                .healthy
                .iter()
                .zip(&plan.bindings)
                .any(|(&index, prepared)| {
                    let worker = &workers[index];
                    !worker.is_available()
                        || self
                            .targets
                            .get(worker.url())
                            .map_or(0, |binding| binding.generation)
                            != prepared.generation
                });
            if stale {
                drop(guard);
                continue;
            }
            let candidates = self.candidate_views(workers, &plan);
            let (costs, components) = self.score_candidates(&plan, &candidates)?;

            // Recheck the table inside the critical section.
            let acquired = digest
                .as_ref()
                .and_then(|digest| self.affinity.acquire(digest));
            let (position, mut record) = match acquired
                .as_ref()
                .and_then(|record| self.affinity_position(&plan, record))
            {
                Some(position) if !excluded(position) => (position, acquired),
                valid => {
                    if let Some(record) = acquired {
                        record.release_ref();
                    }
                    let eligible = Self::eligible_positions(workers, &plan, excluded_workers);
                    let position = self.pick_position(
                        plan.has_tokens,
                        &costs,
                        &components,
                        eligible.as_deref(),
                    )?;
                    let binding = &plan.bindings[position];
                    // Retry exclusions preserve a valid binding; unregistered
                    // targets cannot establish one without a generation.
                    let record = digest
                        .filter(|_| valid.is_none() && binding.generation != 0)
                        .and_then(|digest| {
                            self.affinity
                                .bind(digest, binding.target.clone(), binding.generation)
                        });
                    (position, record)
                }
            };
            let target = plan.bindings[position].target.clone();

            let (worker_index, new_prefill_tokens, decision) =
                self.decision_for_position(&plan, &candidates, &costs, &components, position);
            // Reserve the actual target's slot and worker load before unlocking.
            let slot = self.slot_manager.as_ref().map(|manager| {
                manager.begin_prefill(&target, new_prefill_tokens);
                KvRequestSlot::new(manager.clone(), target, new_prefill_tokens, record.take())
            });
            // No slot guard will run; release the reference inline.
            if slot.is_none() {
                if let Some(record) = record {
                    record.release_ref();
                }
            }
            if let Some(worker) = workers.get(worker_index) {
                worker.increment_load();
            }
            return Some(KvSelectionReceipt {
                worker_index,
                new_prefill_tokens,
                decision,
                slot,
            });
        }
    }

    /// Position of a record's target among the plan's healthy bindings while
    /// its registration generation is still current.
    fn affinity_position(&self, plan: &SelectionPlan, record: &AffinityRecord) -> Option<usize> {
        let position = plan
            .bindings
            .iter()
            .position(|binding| binding.target == record.target)?;
        (record.generation != 0 && plan.bindings[position].generation == record.generation)
            .then_some(position)
    }

    /// Healthy workers, their target bindings, and per-candidate KV matches.
    /// All provider queries happen here, outside every lock.
    fn prepare_selection(
        &self,
        workers: &[Arc<dyn Worker>],
        tokens: KvRequestTokens<'_>,
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
    ) -> Option<SelectionPlan> {
        let (healthy, bindings): (_, Vec<TargetBinding>) = {
            let _guard = self.selection_lock.lock();
            let mut healthy = get_healthy_worker_indices(workers);
            // Once lifecycle registration is in use, removed worker handles
            // must not reappear as never-registered load-only fallbacks.
            if self.next_generation.load(Ordering::Acquire) > 1 {
                healthy.retain(|&index| self.targets.contains_key(workers[index].url()));
            }
            if healthy.is_empty() {
                return None;
            }
            let bindings = healthy
                .iter()
                .map(|&index| self.binding_for_worker(workers[index].as_ref()))
                .collect();
            (healthy, bindings)
        };

        // KV events carry lora_name but not cache_salt. Hash LoRA requests the
        // same way as ingestion; for salted requests, skip placement credit
        // instead of claiming an incompatible unsalted cache entry.
        let placement_enabled = cache_salt.is_none_or(|salt| salt.is_empty());
        let prompts: Vec<_> = healthy
            .iter()
            .map(|&index| tokens.prompts_for(workers[index].as_ref()))
            .collect();
        // A newly registered or unrendered candidate is unknown, not a
        // zero-cost cache hit. Compare the whole pool using load in that case.
        let has_tokens = prompts
            .iter()
            .all(|items| !items.is_empty() && items.iter().all(|ids| !ids.is_empty()));
        if !has_tokens {
            crate::metrics::RouterMetrics::record_kv_placement_lookup("missing_tokens");
        }
        let mut query_cache: HashMap<(Option<CacheKey>, &[u32]), Vec<TierMatch>> = HashMap::new();
        let mut grouped = Vec::with_capacity(bindings.len());
        let mut input_tokens = Vec::with_capacity(bindings.len());
        let mut prefill_tokens = Vec::with_capacity(bindings.len());
        for (binding, prompts) in bindings.iter().zip(prompts) {
            let mut total_input = 0usize;
            let mut total_prefill = 0u64;
            let mut single_matches = Vec::new();
            if has_tokens {
                for ids in &prompts {
                    let hits = query_cache
                        .entry((binding.cache_key.clone(), ids))
                        .or_insert_with(|| {
                            if !placement_enabled {
                                crate::metrics::RouterMetrics::record_kv_placement_lookup(
                                    "cache_salt",
                                );
                                return Vec::new();
                            }
                            let local_hashes =
                                Self::local_hashes_with_lora(ids, binding.block_size, lora_name);
                            if local_hashes.is_empty() {
                                return Vec::new();
                            }
                            let hits = binding.matches.find_tiered_matches(&MatchQuery {
                                group_idx: 0,
                                local_hashes,
                                tiers_of_interest: Self::tiers_of_interest(),
                            });
                            crate::metrics::RouterMetrics::record_kv_placement_lookup(
                                if binding.cache_key.is_none() {
                                    "index_unavailable"
                                } else if hits.is_empty() {
                                    "miss"
                                } else {
                                    "hit"
                                },
                            );
                            hits
                        });
                    let matches: Vec<_> = hits
                        .iter()
                        .filter(|hit| hit.target == binding.target)
                        .cloned()
                        .collect();
                    let params = CostParams {
                        block_size: binding.block_size,
                        ..self.params
                    };
                    total_prefill = total_prefill.saturating_add(request_prefill_tokens(
                        &CandidateView {
                            target: &binding.target,
                            matches: &matches,
                            load: LoadSample::default(),
                            input_tokens: ids.len(),
                            prefill_tokens: None,
                        },
                        &params,
                    ));
                    total_input = total_input.saturating_add(ids.len());
                    if prompts.len() == 1 {
                        single_matches = matches;
                    }
                }
            }
            grouped.push(single_matches);
            input_tokens.push(total_input);
            prefill_tokens.push((has_tokens && prompts.len() > 1).then_some(total_prefill));
        }
        Some(SelectionPlan {
            healthy,
            bindings,
            grouped,
            input_tokens,
            prefill_tokens,
            has_tokens,
        })
    }

    fn candidate_views<'p>(
        &self,
        workers: &[Arc<dyn Worker>],
        plan: &'p SelectionPlan,
    ) -> Vec<CandidateView<'p>> {
        plan.bindings
            .iter()
            .zip(plan.grouped.iter())
            .zip(plan.healthy.iter())
            .enumerate()
            .map(|(position, ((binding, matches), &worker_index))| {
                let mut load = self.load.sample(&binding.target);
                // Existing request paths already bracket Worker::load. Use it
                // as the live batch-size floor while the richer slot manager
                // additionally carries KV-credited prefill tokens.
                load.active_requests = load.active_requests.max(workers[worker_index].load());
                CandidateView {
                    target: &binding.target,
                    matches,
                    load,
                    input_tokens: plan.input_tokens[position],
                    prefill_tokens: plan.prefill_tokens[position],
                }
            })
            .collect()
    }

    fn score_candidates(
        &self,
        plan: &SelectionPlan,
        candidates: &[CandidateView<'_>],
    ) -> Option<(Vec<f64>, Vec<CostComponents>)> {
        let base_params = self.params;
        let mut costs = vec![0.0; candidates.len()];
        let mut components = vec![CostComponents::default(); candidates.len()];
        let mut indices_by_identity: HashMap<Option<CacheKey>, Vec<usize>> = HashMap::new();
        for (index, binding) in plan.bindings.iter().enumerate() {
            indices_by_identity
                .entry(binding.cache_key.clone())
                .or_default()
                .push(index);
        }
        for indices in indices_by_identity.values() {
            let mut params = base_params;
            params.block_size = plan.bindings[*indices.first()?].block_size;
            let identity_candidates: Vec<CandidateView<'_>> =
                indices.iter().map(|&index| candidates[index]).collect();
            let mut identity_costs = vec![0.0; identity_candidates.len()];
            for scorer in &self.scorers {
                scorer
                    .score(&identity_candidates, &params, &mut identity_costs)
                    .ok()?;
            }
            for ((&candidate_index, cost), candidate) in indices
                .iter()
                .zip(identity_costs)
                .zip(identity_candidates.iter())
            {
                costs[candidate_index] = cost;
                components[candidate_index] = cost_components(candidate, &params);
            }
        }
        Some((costs, components))
    }

    /// Positions the picker may choose: every binding, minus workers that
    /// already failed an attempt of this request. `None` when no exclusion
    /// applies or every worker is excluded (retry them all).
    fn eligible_positions(
        workers: &[Arc<dyn Worker>],
        plan: &SelectionPlan,
        excluded_workers: &[String],
    ) -> Option<Vec<usize>> {
        if excluded_workers.is_empty() {
            return None;
        }
        let eligible: Vec<usize> = (0..plan.bindings.len())
            .filter(|&position| {
                !excluded_workers
                    .iter()
                    .any(|url| workers[plan.healthy[position]].url() == url.as_str())
            })
            .collect();
        (!eligible.is_empty()).then_some(eligible)
    }

    fn pick_position(
        &self,
        has_tokens: bool,
        costs: &[f64],
        components: &[CostComponents],
        eligible: Option<&[usize]>,
    ) -> Option<usize> {
        let Some(eligible) = eligible else {
            return self.pick_among(has_tokens, costs, components);
        };
        let eligible_costs: Vec<f64> = eligible.iter().map(|&index| costs[index]).collect();
        let eligible_components: Vec<CostComponents> =
            eligible.iter().map(|&index| components[index]).collect();
        let offset = self.pick_among(has_tokens, &eligible_costs, &eligible_components)?;
        eligible.get(offset).copied()
    }

    fn pick_among(
        &self,
        has_tokens: bool,
        costs: &[f64],
        components: &[CostComponents],
    ) -> Option<usize> {
        // Zero work remains possible for token-less requests.
        // Preserve KV preference, breaking equal-work ties by projected load.
        if costs.iter().all(|cost| *cost == 0.0)
            || (self.params.router_temperature <= 0.0 && costs.contains(&0.0))
        {
            if !has_tokens {
                // Requests without verified token IDs degrade to live load
                // instead of deterministically concentrating on worker zero.
                components
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| costs[*index] == 0.0)
                    .min_by_key(|(_, component)| component.batchsize)
                    .map(|(index, _)| index)
            } else {
                components
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| costs[*index] == 0.0)
                    .min_by_key(|(_, component)| (component.ptoken, component.batchsize))
                    .map(|(index, _)| index)
            }
        } else {
            self.picker.pick(costs)
        }
    }

    fn decision_for_position(
        &self,
        plan: &SelectionPlan,
        candidates: &[CandidateView<'_>],
        costs: &[f64],
        components: &[CostComponents],
        position: usize,
    ) -> (usize, u64, CostDecision) {
        let worker_index = plan.healthy[position];
        let mut selected_params = self.params;
        selected_params.block_size = plan.bindings[position].block_size;
        (
            worker_index,
            request_prefill_tokens(&candidates[position], &selected_params),
            CostDecision {
                target: plan.bindings[position].target.clone(),
                cost: costs[position],
                components: components[position],
            },
        )
    }

    pub fn decision_for(
        &self,
        workers: &[Arc<dyn Worker>],
        token_ids: Option<&[u32]>,
    ) -> Option<CostDecision> {
        self.select_worker_pure(workers, token_ids, None, None)
            .map(|(_, decision)| decision)
    }
}

impl Default for KvAwarePolicy {
    fn default() -> Self {
        Self::new(CostParams::default())
    }
}

impl LoadBalancingPolicy for KvAwarePolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        _request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        self.select_worker_with_token_ids(workers, None)
    }

    fn name(&self) -> &'static str {
        "kv_aware"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl KvAwareOps for KvAwarePolicy {
    fn select_worker_with_inputs(
        &self,
        workers: &[Arc<dyn Worker>],
        tokens: KvRequestTokens<'_>,
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
        affinity_key: Option<&AffinityKey>,
        excluded_workers: &[String],
    ) -> Option<KvSelectionReceipt> {
        KvAwarePolicy::select_worker_with_inputs(
            self,
            workers,
            tokens,
            lora_name,
            cache_salt,
            affinity_key,
            excluded_workers,
        )
    }

    fn register_worker(
        &self,
        worker: &dyn Worker,
        info: Option<&WorkerKvInfo>,
        matches: Option<Arc<dyn TieredMatchProvider>>,
    ) {
        let target = RoutableTarget {
            instance_id: Arc::from(
                info.map(|info| info.instance_id.as_str())
                    .filter(|id| !id.is_empty())
                    .unwrap_or_else(|| worker.base_url()),
            ),
            dp_rank: worker.dp_rank().unwrap_or(0) as u32,
        };
        match (info.and_then(|info| info.cache_key.clone()), matches) {
            (Some(cache_key), Some(matches)) => {
                self.register_target_with_cache_identity(worker.url(), target, cache_key, matches);
            }
            _ => self.register_target(worker.url(), target),
        }
    }

    fn unregister_worker(&self, worker_url: &str) {
        self.unregister_target(worker_url);
    }

    fn select_worker_with_receipt(
        &self,
        workers: &[Arc<dyn Worker>],
        token_ids: Option<&[u32]>,
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
        affinity_key: Option<&AffinityKey>,
        excluded_workers: &[String],
    ) -> Option<KvSelectionReceipt> {
        KvAwarePolicy::select_worker_with_receipt(
            self,
            workers,
            token_ids,
            lora_name,
            cache_salt,
            affinity_key,
            excluded_workers,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, DPAwareWorker, WorkerType};
    use crate::kv_index::Locality;
    use crate::policies::RoundRobinPolicy;

    fn target(name: &str, rank: u32) -> RoutableTarget {
        RoutableTarget {
            instance_id: Arc::from(name),
            dp_rank: rank,
        }
    }

    /// Undo a reserving selection's reservations (mirrors the router's request
    /// guard) so sequential test selections observe the production baseline.
    fn release_receipt(workers: &[Arc<dyn Worker>], receipt: &KvSelectionReceipt) {
        if let Some(slot) = &receipt.slot {
            slot.release();
        }
        if let Some(worker) = workers.get(receipt.worker_index) {
            worker.decrement_load();
        }
    }

    fn tier_match(
        target: &RoutableTarget,
        tier: StorageTier,
        segments: &[(u32, u32)],
    ) -> TierMatch {
        TierMatch {
            target: target.clone(),
            tier,
            segments: segments.to_vec(),
            locality: Locality::Local,
        }
    }

    fn policy_with_index(
        index: Arc<crate::kv_index::KvBlockIndexer>,
        params: CostParams,
    ) -> KvAwarePolicy {
        KvAwarePolicy::with_components(
            index,
            Arc::new(InMemorySlotManager::new()),
            vec![Arc::new(MultiplicativeScorer)],
            Arc::new(TemperaturePicker::new(params.router_temperature)),
            params,
        )
    }

    #[test]
    fn slot_manager_moves_colocated_and_separated_load() {
        let slots = InMemorySlotManager::new();
        let prefill = target("p", 1);
        let decode = target("d", 2);
        slots.begin_prefill(&prefill, 90);
        assert_eq!(
            slots.sample(&prefill),
            LoadSample {
                active_prefill_tokens: 90,
                active_requests: 1
            }
        );
        slots.begin_decode(&prefill, &prefill, 90);
        assert_eq!(slots.sample(&prefill).active_prefill_tokens, 0);
        assert_eq!(slots.sample(&prefill).active_requests, 1);
        slots.free(&prefill);
        assert_eq!(slots.sample(&prefill).active_requests, 0);

        slots.begin_prefill(&prefill, 30);
        slots.begin_decode(&prefill, &decode, 30);
        assert_eq!(slots.sample(&prefill), LoadSample::default());
        assert_eq!(slots.sample(&decode).active_requests, 1);
        slots.free(&decode);
        slots.free(&decode); // duplicate cleanup saturates
        assert_eq!(slots.sample(&decode).active_requests, 0);
    }

    #[test]
    fn unregister_and_request_cleanup_prune_only_idle_slots() {
        let slots = InMemorySlotManager::new();
        let idle = target("idle", 0);
        slots
            .slots
            .insert(idle.clone(), Arc::new(SlotCounters::default()));
        slots.unregister_target(&idle);
        assert!(!slots.slots.contains_key(&idle));

        let active = target("active", 1);
        slots.begin_prefill(&active, 32);
        slots.unregister_target(&active);
        assert_eq!(slots.sample(&active).active_requests, 1);
        assert_eq!(slots.sample(&active).active_prefill_tokens, 32);

        slots.begin_decode(&active, &active, 32);
        slots.free(&active);
        assert!(!slots.slots.contains_key(&active));
    }

    #[test]
    fn two_segment_credit_uses_device_and_one_continuation() {
        let t = target("w", 0);
        let matches = vec![
            tier_match(&t, StorageTier::Device, &[(0, 5)]),
            tier_match(&t, StorageTier::HostPinned, &[(5, 9)]),
            tier_match(&t, StorageTier::Disk, &[(5, 8)]),
        ];
        let candidate = CandidateView {
            target: &t,
            matches: &matches,
            load: LoadSample {
                active_prefill_tokens: 7,
                active_requests: 3,
            },
            prefill_tokens: None,
            input_tokens: 160,
        };
        let params = CostParams {
            block_size: 16,
            transfer_credits: TransferCredits {
                host_pinned: 0.5,
                ..TransferCredits::default()
            },
            ..CostParams::default()
        };
        // 5 device blocks + 4 * .5 host blocks = 112 credited tokens.
        assert_eq!(
            cost_components(&candidate, &params),
            CostComponents {
                ptoken: 55,
                batchsize: 3
            }
        );
    }

    #[test]
    fn disconnected_external_segment_is_not_credited() {
        let t = target("w", 0);
        let matches = vec![
            tier_match(&t, StorageTier::Device, &[(0, 3)]),
            tier_match(&t, StorageTier::HostPinned, &[(4, 10)]),
        ];
        let candidate = CandidateView {
            target: &t,
            matches: &matches,
            load: LoadSample::default(),
            prefill_tokens: None,
            input_tokens: 160,
        };
        assert_eq!(
            cost_components(&candidate, &CostParams::default()).ptoken,
            112
        );
    }

    #[test]
    fn none_token_ids_reduce_ptoken_to_backlog() {
        let t = target("w", 0);
        let candidate = CandidateView {
            target: &t,
            matches: &[],
            load: LoadSample {
                active_prefill_tokens: 42,
                active_requests: 2,
            },
            prefill_tokens: None,
            input_tokens: 0,
        };
        assert_eq!(
            cost_components(&candidate, &CostParams::default()).ptoken,
            42
        );
    }

    #[test]
    fn gpu_credit_reserves_the_engine_last_token_block() {
        let t = target("w", 0);
        let params = CostParams::default();
        for (tokens, cached, new_prefill) in [
            (0, 0, 0),
            (16, 0, 16),
            (17, 16, 1),
            (32, 16, 16),
            (48, 32, 16),
            (49, 48, 1),
        ] {
            let hits = [tier_match(&t, StorageTier::Device, &[(0, 10)])];
            let candidate = CandidateView {
                target: &t,
                matches: &hits,
                load: LoadSample::default(),
                prefill_tokens: None,
                input_tokens: tokens,
            };
            assert_eq!(
                credited_tokens(&candidate, &params),
                cached as f64,
                "tokens={tokens}"
            );
            assert_eq!(
                request_prefill_tokens(&candidate, &params),
                new_prefill,
                "tokens={tokens}"
            );
        }
    }

    #[test]
    fn external_credit_cannot_remove_last_token_work() {
        let t = target("w", 0);
        let params = CostParams {
            transfer_credits: TransferCredits {
                external: 1.0,
                ..TransferCredits::default()
            },
            ..CostParams::default()
        };
        let hits = [tier_match(&t, StorageTier::External, &[(0, 3)])];
        let candidate = CandidateView {
            target: &t,
            matches: &hits,
            load: LoadSample::default(),
            prefill_tokens: None,
            input_tokens: 48,
        };
        assert_eq!(request_prefill_tokens(&candidate, &params), 1);
    }

    #[test]
    fn mixed_zero_cost_candidates_still_choose_the_idle_worker() {
        let policy = KvAwarePolicy::default();
        let workers: Vec<Arc<dyn Worker>> = ["http://decode", "http://idle", "http://prefill"]
            .iter()
            .map(|url| {
                Arc::new(BasicWorker::new((*url).into(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect();
        let slots = policy.slot_manager().unwrap();
        let decode = policy.target_for_worker(workers[0].as_ref());
        slots.begin_prefill(&decode, 8);
        slots.begin_decode(&decode, &decode, 8);
        assert_eq!(
            policy.select_worker_with_token_ids(&workers[..2], None),
            Some(1)
        );
        let prefill = policy.target_for_worker(workers[2].as_ref());
        slots.begin_prefill(&prefill, 8);
        let receipt = policy
            .select_worker_with_receipt(&workers, None, None, None, None, &[])
            .unwrap();
        assert_eq!(receipt.worker_index, 1);
        assert_eq!(receipt.decision.cost, 0.0);
        assert_eq!(receipt.decision.components.batchsize, 0);
        assert!(receipt.slot.is_some());
        release_receipt(&workers, &receipt);
    }

    #[test]
    fn cached_and_empty_prompts_prefer_idle_worker_over_one_hundred_active_requests() {
        let index = Arc::new(crate::kv_index::KvBlockIndexer::new());
        let policy = policy_with_index(index.clone(), CostParams::default());
        let workers: Vec<Arc<dyn Worker>> = ["http://busy", "http://idle"]
            .iter()
            .map(|url| {
                Arc::new(BasicWorker::new((*url).into(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect();
        let ids: Vec<u32> = (0..32).collect();
        let blocks: Vec<_> = KvAwarePolicy::local_hashes(&ids, 16)
            .into_iter()
            .enumerate()
            .map(|(i, local)| (Arc::from(format!("seq-{i}")), local))
            .collect();
        for worker in &workers {
            let t = policy.target_for_worker(worker.as_ref());
            index.store(
                0,
                crate::kv_index::ResidencyOwner::Worker {
                    source: t.instance_id,
                    dp_rank: t.dp_rank,
                    incarnation: 0,
                },
                StorageTier::Device,
                None,
                &blocks,
            );
        }
        for _ in 0..100 {
            workers[0].increment_load();
        }
        for (tokens, expected_prefill) in [(&ids[..0], 0), (&ids[..], 16)] {
            let receipt = policy
                .select_worker_with_receipt(&workers, Some(tokens), None, None, None, &[])
                .unwrap();
            assert_eq!(receipt.worker_index, 1);
            assert_eq!(receipt.new_prefill_tokens, expected_prefill);
            assert_eq!(receipt.decision.components.batchsize, 0);
            release_receipt(&workers, &receipt);
        }
    }

    #[test]
    fn scorer_counts_incoming_request_in_batchsize() {
        let t = target("w", 0);
        let candidate = CandidateView {
            target: &t,
            matches: &[],
            load: LoadSample {
                active_prefill_tokens: 5,
                active_requests: 2,
            },
            prefill_tokens: None,
            input_tokens: 10,
        };
        let mut costs = [3.0];
        MultiplicativeScorer
            .score(&[candidate], &CostParams::default(), &mut costs)
            .unwrap();
        assert_eq!(costs, [48.0]);
    }

    #[test]
    fn idle_cold_worker_is_charged_for_candidate_prefill() {
        let hot = target("hot", 0);
        let cold = target("cold", 0);
        let matches = [tier_match(&hot, StorageTier::Device, &[(0, 2468)])];
        let candidates = [
            CandidateView {
                target: &hot,
                matches: &matches,
                load: LoadSample {
                    active_requests: 1,
                    ..LoadSample::default()
                },
                prefill_tokens: None,
                input_tokens: 40_000,
            },
            CandidateView {
                target: &cold,
                matches: &[],
                load: LoadSample::default(),
                prefill_tokens: None,
                input_tokens: 40_000,
            },
        ];
        let mut costs = [0.0; 2];
        MultiplicativeScorer
            .score(&candidates, &CostParams::default(), &mut costs)
            .unwrap();
        assert_eq!(costs, [1024.0, 40_000.0]);
        assert_eq!(TemperaturePicker::new(0.0).pick(&costs), Some(0));
    }

    #[test]
    fn active_batch_size_preserves_saturation() {
        let t = target("w", 0);
        let candidate = CandidateView {
            target: &t,
            matches: &[],
            load: LoadSample {
                active_requests: usize::MAX,
                ..LoadSample::default()
            },
            prefill_tokens: None,
            input_tokens: 1,
        };
        assert_eq!(
            cost_components(&candidate, &CostParams::default()).batchsize,
            usize::MAX
        );
        let mut costs = [0.0];
        MultiplicativeScorer
            .score(&[candidate], &CostParams::default(), &mut costs)
            .unwrap();
        assert!(costs[0] > 0.0);
    }

    #[test]
    fn cache_hit_offsets_load_until_cold_prefill_becomes_cheaper() {
        let index = Arc::new(crate::kv_index::KvBlockIndexer::new());
        let policy = policy_with_index(
            index.clone(),
            CostParams {
                block_size: 2,
                ..CostParams::default()
            },
        );
        let warm = target("http://warm", 0);
        policy.register_target("http://warm", warm.clone());
        policy.register_target("http://cold", target("http://cold", 0));
        let ids = [1_u32, 2, 3, 4, 5, 6];
        let hashes = KvAwarePolicy::local_hashes(&ids, 2);
        index.store(
            0,
            crate::kv_index::ResidencyOwner::Worker {
                source: warm.instance_id.clone(),
                dp_rank: 0,
                incarnation: 0,
            },
            StorageTier::Device,
            None,
            &[
                (Arc::from("seq-1"), hashes[0].clone()),
                (Arc::from("seq-2"), hashes[1].clone()),
            ],
        );
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new("http://warm".into(), WorkerType::Regular)),
            Arc::new(BasicWorker::new("http://cold".into(), WorkerType::Regular)),
        ];
        let receipt = policy
            .select_worker_with_receipt(&workers, Some(&ids), None, None, None, &[])
            .unwrap();
        assert_eq!(receipt.worker_index, 0);
        assert_eq!(receipt.new_prefill_tokens, 2);
        assert_eq!(receipt.decision.cost, 2.0);
        assert_eq!(receipt.decision.components.batchsize, 0);
        release_receipt(&workers, &receipt);

        workers[0].increment_load();
        let receipt = policy
            .select_worker_with_receipt(&workers, Some(&ids), None, None, None, &[])
            .unwrap();
        assert_eq!(receipt.worker_index, 0);
        assert_eq!(receipt.decision.cost, 4.0);
        assert_eq!(receipt.decision.components.batchsize, 1);
        release_receipt(&workers, &receipt);

        workers[0].increment_load();
        workers[0].increment_load();
        let receipt = policy
            .select_worker_with_receipt(&workers, Some(&ids), None, None, None, &[])
            .unwrap();
        assert_eq!(receipt.worker_index, 1);
        assert_eq!(receipt.decision.cost, 6.0);
        assert_eq!(receipt.decision.components.batchsize, 0);
        release_receipt(&workers, &receipt);
    }

    #[test]
    fn argmin_picker_is_deterministic() {
        assert_eq!(TemperaturePicker::new(0.0).pick(&[5.0, 2.0, 2.0]), Some(1));
    }

    #[test]
    fn missing_token_ids_fall_back_to_live_worker_load() {
        let policy = KvAwarePolicy::default();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new("http://busy".into(), WorkerType::Regular)),
            Arc::new(BasicWorker::new("http://idle".into(), WorkerType::Regular)),
        ];
        workers[0].increment_load();
        workers[0].increment_load();
        workers[1].increment_load();

        assert_eq!(policy.select_worker_with_token_ids(&workers, None), Some(1));
    }

    #[test]
    fn batches_match_each_prompt_from_the_root_and_sum_remaining_work() {
        let index = Arc::new(crate::kv_index::KvBlockIndexer::new());
        let policy = policy_with_index(
            index.clone(),
            CostParams {
                block_size: 2,
                ..Default::default()
            },
        );
        let warm = target("warm", 0);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new("http://warm".into(), WorkerType::Regular)),
            Arc::new(BasicWorker::new("http://cold".into(), WorkerType::Regular)),
        ];
        policy.register_target(workers[0].url(), warm.clone());
        policy.register_target(workers[1].url(), target("cold", 0));
        let prompts: Arc<[Vec<u32>]> =
            Arc::from(vec![vec![1, 2, 3, 4, 5, 6], vec![11, 12, 13, 14, 15, 16]]);
        for (p, ids) in prompts.iter().enumerate() {
            let blocks: Vec<_> = KvAwarePolicy::local_hashes(&ids[..4], 2)
                .into_iter()
                .enumerate()
                .map(|(i, hash)| (Arc::from(format!("{p}-{i}")), hash))
                .collect();
            index.store(
                0,
                crate::kv_index::ResidencyOwner::Worker {
                    source: warm.instance_id.clone(),
                    dp_rank: 0,
                    incarnation: 0,
                },
                StorageTier::Device,
                None,
                &blocks,
            );
        }
        let inputs = WorkerTokenInputs::from([
            (workers[0].base_url().into(), prompts.clone()),
            (workers[1].base_url().into(), prompts),
        ]);
        let receipt = policy
            .select_worker_with_inputs(
                &workers,
                KvRequestTokens::PerWorker(&inputs),
                None,
                None,
                None,
                &[],
            )
            .unwrap();
        assert_eq!(receipt.worker_index, 0);
        assert_eq!(receipt.new_prefill_tokens, 4);
        release_receipt(&workers, &receipt);
        let receipt = policy
            .select_worker_with_inputs(
                &workers,
                KvRequestTokens::PerWorker(&inputs),
                None,
                Some("salt"),
                None,
                &[],
            )
            .unwrap();
        assert_eq!(receipt.new_prefill_tokens, 12);
        release_receipt(&workers, &receipt);
    }

    #[test]
    fn lora_name_is_seeded_into_request_block_hashes() {
        let index = Arc::new(crate::kv_index::KvBlockIndexer::new());
        let policy = policy_with_index(
            index.clone(),
            CostParams {
                block_size: 2,
                ..CostParams::default()
            },
        );
        let base = target("http://base", 0);
        let lora = target("http://lora", 0);
        policy.register_target("http://base", base.clone());
        policy.register_target("http://lora", lora.clone());
        let ids = [1_u32, 2, 3, 4];
        let base_hashes = KvAwarePolicy::local_hashes(&ids, 2);
        let lora_hashes = KvAwarePolicy::local_hashes_with_lora(&ids, 2, Some("adapter"));
        for (target, hashes) in [(&base, base_hashes), (&lora, lora_hashes)] {
            index.store(
                0,
                crate::kv_index::ResidencyOwner::Worker {
                    source: target.instance_id.clone(),
                    dp_rank: 0,
                    incarnation: 0,
                },
                StorageTier::Device,
                None,
                &[
                    (Arc::from("seq-1"), hashes[0].clone()),
                    (Arc::from("seq-2"), hashes[1].clone()),
                ],
            );
        }
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new("http://base".into(), WorkerType::Regular)),
            Arc::new(BasicWorker::new("http://lora".into(), WorkerType::Regular)),
        ];
        for worker in &workers {
            worker.increment_load();
        }

        assert_eq!(
            policy
                .select_worker_with_receipt(&workers, Some(&ids), Some("adapter"), None, None, &[])
                .map(|receipt| receipt.worker_index),
            Some(1)
        );
    }

    #[test]
    fn cache_salt_disables_incompatible_placement_credit() {
        let index = Arc::new(crate::kv_index::KvBlockIndexer::new());
        let policy = policy_with_index(
            index.clone(),
            CostParams {
                block_size: 2,
                ..CostParams::default()
            },
        );
        let cached = target("http://cached", 0);
        let uncached = target("http://uncached", 0);
        policy.register_target("http://cached", cached.clone());
        policy.register_target("http://uncached", uncached);
        let ids = [1_u32, 2, 3, 4];
        let hashes = KvAwarePolicy::local_hashes(&ids, 2);
        index.store(
            0,
            crate::kv_index::ResidencyOwner::Worker {
                source: cached.instance_id.clone(),
                dp_rank: 0,
                incarnation: 0,
            },
            StorageTier::Device,
            None,
            &[
                (Arc::from("seq-1"), hashes[0].clone()),
                (Arc::from("seq-2"), hashes[1].clone()),
            ],
        );
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://cached".into(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://uncached".into(),
                WorkerType::Regular,
            )),
        ];
        workers[0].increment_load();
        workers[0].increment_load();
        workers[1].increment_load();

        assert_eq!(
            policy
                .select_worker_with_receipt(
                    &workers,
                    Some(&ids),
                    None,
                    Some("tenant-salt"),
                    None,
                    &[]
                )
                .map(|receipt| receipt.worker_index),
            Some(1)
        );
        assert_eq!(
            policy
                .select_worker_with_receipt(
                    std::slice::from_ref(&workers[0]),
                    Some(&ids),
                    None,
                    Some("tenant-salt"),
                    None,
                    &[],
                )
                .unwrap()
                .new_prefill_tokens,
            ids.len() as u64
        );
    }

    #[test]
    fn selection_queries_each_cache_identity_with_its_own_provider() {
        let first_index = Arc::new(crate::kv_index::KvBlockIndexer::new());
        let second_index = Arc::new(crate::kv_index::KvBlockIndexer::new());
        let policy = KvAwarePolicy::default();
        let first = target("first-instance", 0);
        let second = target("second-instance", 0);
        let first_key = CacheKey {
            model: Arc::from("model-a"),
            hash_mode: crate::kv_index::HashMode::Sha256,
            block_size: 2,
        };
        let second_key = CacheKey {
            model: Arc::from("model-b"),
            hash_mode: crate::kv_index::HashMode::Sha256,
            block_size: 4,
        };
        policy.register_target_with_cache_identity(
            "http://first",
            first.clone(),
            first_key,
            first_index.clone(),
        );
        policy.register_target_with_cache_identity(
            "http://second",
            second,
            second_key,
            second_index,
        );

        let ids = [1_u32, 2, 3, 4];
        let hashes = KvAwarePolicy::local_hashes(&ids, 2);
        first_index.store(
            0,
            crate::kv_index::ResidencyOwner::Worker {
                source: first.instance_id.clone(),
                dp_rank: 0,
                incarnation: 0,
            },
            StorageTier::Device,
            None,
            &[
                (Arc::from("seq-1"), hashes[0].clone()),
                (Arc::from("seq-2"), hashes[1].clone()),
            ],
        );
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new("http://first".into(), WorkerType::Regular)),
            Arc::new(BasicWorker::new(
                "http://second".into(),
                WorkerType::Regular,
            )),
        ];
        for worker in &workers {
            worker.increment_load();
        }

        let receipt = policy
            .select_worker_with_receipt(&workers, Some(&ids), None, None, None, &[])
            .unwrap();
        assert_eq!(receipt.worker_index, 0);
        assert_eq!(receipt.decision.target, first);
        assert_eq!(receipt.new_prefill_tokens, 2);
    }

    #[test]
    fn idle_pool_prefers_deep_hit() {
        let index = Arc::new(crate::kv_index::KvBlockIndexer::new());
        let policy = policy_with_index(
            index.clone(),
            CostParams {
                block_size: 2,
                ..CostParams::default()
            },
        );
        let first = target("http://w1", 0);
        let second = target("http://w2", 0);
        policy.register_target("http://w1", first.clone());
        policy.register_target("http://w2", second);
        let ids = [1_u32, 2, 3, 4];
        let locals = KvAwarePolicy::local_hashes(&ids, 2);
        index.store(
            0,
            crate::kv_index::ResidencyOwner::Worker {
                source: first.instance_id.clone(),
                dp_rank: 0,
                incarnation: 0,
            },
            StorageTier::Device,
            None,
            &[
                (Arc::from("seq-1"), locals[0].clone()),
                (Arc::from("seq-2"), locals[1].clone()),
            ],
        );
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new("http://w1".into(), WorkerType::Regular)),
            Arc::new(BasicWorker::new("http://w2".into(), WorkerType::Regular)),
        ];
        assert_eq!(
            policy.select_worker_with_token_ids(&workers, Some(&ids)),
            Some(0)
        );
    }
    #[test]
    fn stateless_policy_has_no_kv_capability() {
        assert!(kv_aware_ops(&RoundRobinPolicy::new()).is_none());
    }

    fn registered_worker_pair(policy: &KvAwarePolicy) -> Vec<Arc<dyn Worker>> {
        let workers: Vec<Arc<dyn Worker>> = ["http://a", "http://b"]
            .iter()
            .map(|url| {
                Arc::new(BasicWorker::new((*url).into(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect();
        policy.register_target("http://a", target("instance-a", 0));
        policy.register_target("http://b", target("instance-b", 0));
        workers
    }

    fn affinity_key(session: &str) -> AffinityKey {
        AffinityKey::from_request(
            None,
            Some(&serde_json::json!({"session_id": session})),
            "default",
        )
        .unwrap()
    }

    #[test]
    fn affinity_reuses_recorded_worker_across_requests() {
        let policy = KvAwarePolicy::default();
        let workers = registered_worker_pair(&policy);
        let key = affinity_key("session-a");

        let first = policy
            .select_worker_with_receipt(&workers, None, None, None, Some(&key), &[])
            .unwrap();
        release_receipt(&workers, &first);
        // Ordinary load changes must not migrate a valid binding.
        workers[first.worker_index].increment_load();
        workers[first.worker_index].increment_load();

        let second = policy
            .select_worker_with_receipt(&workers, None, None, None, Some(&key), &[])
            .unwrap();
        assert_eq!(second.worker_index, first.worker_index);
        release_receipt(&workers, &second);

        // A different key selects by fresh load instead.
        let other = policy
            .select_worker_with_receipt(
                &workers,
                None,
                None,
                None,
                Some(&affinity_key("session-b")),
                &[],
            )
            .unwrap();
        assert_ne!(other.worker_index, first.worker_index);
        release_receipt(&workers, &other);

        // Requests without a key keep the original selection behavior.
        let unkeyed = policy
            .select_worker_with_receipt(&workers, None, None, None, None, &[])
            .unwrap();
        assert_eq!(unkeyed.worker_index, other.worker_index);
        release_receipt(&workers, &unkeyed);
    }

    #[test]
    fn affinity_scope_separates_lora_and_salt_bindings() {
        let policy = KvAwarePolicy::default();
        let workers = registered_worker_pair(&policy);
        let key = affinity_key("session-a");

        let plain = policy
            .select_worker_with_receipt(&workers, None, None, None, Some(&key), &[])
            .unwrap();
        release_receipt(&workers, &plain);
        workers[plain.worker_index].increment_load();

        // Same session under a different LoRA/salt scope is a fresh binding
        // and picks by load, not by the plain-scope record.
        let salted = policy
            .select_worker_with_receipt(&workers, None, None, Some("tenant"), Some(&key), &[])
            .unwrap();
        assert_ne!(salted.worker_index, plain.worker_index);
        release_receipt(&workers, &salted);
        let lora = policy
            .select_worker_with_receipt(&workers, None, Some("adapter"), None, Some(&key), &[])
            .unwrap();
        assert_ne!(lora.worker_index, plain.worker_index);
    }

    #[test]
    fn affinity_rebinds_on_unregistration_and_reregistration() {
        let policy = KvAwarePolicy::default();
        let workers = registered_worker_pair(&policy);
        let first_target = target("instance-a", 0);
        let second_target = target("instance-b", 0);
        let key = affinity_key("session-a");

        // First placement binds to the idle worker a.
        let bound = policy
            .select_worker_with_receipt(&workers, None, None, None, Some(&key), &[])
            .unwrap();
        assert_eq!(bound.decision.target, first_target);
        release_receipt(&workers, &bound);

        // Removal invalidates the binding; under load the next request rebinds.
        workers[0].increment_load();
        policy.unregister_target("http://a");
        policy.register_target("http://a", first_target.clone());
        let rebound = policy
            .select_worker_with_receipt(&workers, None, None, None, Some(&key), &[])
            .unwrap();
        assert_eq!(rebound.decision.target, second_target);
        release_receipt(&workers, &rebound);

        // Re-registering the same target bumps its registration generation,
        // so the stale record rebinds through fresh selection.
        workers[1].increment_load();
        workers[1].increment_load();
        policy.register_target("http://b", second_target.clone());
        let fresh = policy
            .select_worker_with_receipt(&workers, None, None, None, Some(&key), &[])
            .unwrap();
        assert_eq!(fresh.decision.target, first_target);
    }

    #[test]
    fn affinity_release_retains_binding_for_next_request() {
        let policy = KvAwarePolicy::default();
        let workers = registered_worker_pair(&policy);
        let key = affinity_key("session-a");

        let first = policy
            .select_worker_with_receipt(&workers, None, None, None, Some(&key), &[])
            .unwrap();
        let second = policy
            .select_worker_with_receipt(&workers, None, None, None, Some(&key), &[])
            .unwrap();
        assert_eq!(second.worker_index, first.worker_index);
        release_receipt(&workers, &first);
        release_receipt(&workers, &second);

        let third = policy
            .select_worker_with_receipt(&workers, None, None, None, Some(&key), &[])
            .unwrap();
        assert_eq!(third.worker_index, first.worker_index);
        release_receipt(&workers, &third);
    }

    #[test]
    fn affinity_retry_exclusion_keeps_binding() {
        let policy = KvAwarePolicy::default();
        let workers = registered_worker_pair(&policy);
        let key = affinity_key("session-a");

        let first = policy
            .select_worker_with_receipt(&workers, None, None, None, Some(&key), &[])
            .unwrap();
        release_receipt(&workers, &first);

        let excluded: Vec<String> = vec![workers[first.worker_index].url().to_string()];
        let retry = policy
            .select_worker_with_receipt(&workers, None, None, None, Some(&key), &excluded)
            .unwrap();
        assert_ne!(retry.worker_index, first.worker_index);
        release_receipt(&workers, &retry);

        let third = policy
            .select_worker_with_receipt(&workers, None, None, None, Some(&key), &[])
            .unwrap();
        assert_eq!(third.worker_index, first.worker_index);
        release_receipt(&workers, &third);
    }

    #[test]
    fn affinity_key_ignores_program_scheduling_constraints() {
        let body = serde_json::json!({
            "agent_hint": {"session_id": "hint-session", "blocks_parent": true}
        });
        let key = AffinityKey::from_request(None, Some(&body), "model").unwrap();
        assert_eq!(
            (key.source, key.identity.as_str()),
            ("agent_hint", "hint-session")
        );
    }

    #[test]
    fn affinity_never_binds_unregistered_targets() {
        let table = Arc::new(KvAffinityTable::new());
        let policy = KvAwarePolicy::default().with_shared_affinity(table.clone());
        let workers: Vec<Arc<dyn Worker>> = ["http://a", "http://b"]
            .iter()
            .map(|url| {
                Arc::new(BasicWorker::new((*url).into(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect();
        let key = affinity_key("session-a");

        let first = policy
            .select_worker_with_receipt(&workers, None, None, None, Some(&key), &[])
            .unwrap();
        release_receipt(&workers, &first);

        assert!(table.get(&key.scoped_digest(None, None)).is_none());
    }

    #[test]
    fn affinity_releases_reference_without_slot_manager() {
        let policy = KvAwarePolicy {
            slot_manager: None,
            ..KvAwarePolicy::default()
        };
        let table = policy.affinity.clone();
        let workers = registered_worker_pair(&policy);
        let key = affinity_key("session-a");

        let first = policy
            .select_worker_with_receipt(&workers, None, None, None, Some(&key), &[])
            .unwrap();
        release_receipt(&workers, &first);

        let record = table
            .get(&key.scoped_digest(None, None))
            .expect("registered target binds");
        assert_eq!(record.state.lock().in_flight, 0);
    }

    fn header_map(name: &str, value: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            http::HeaderName::from_bytes(name.as_bytes()).unwrap(),
            value.parse().unwrap(),
        );
        headers
    }

    #[test]
    fn affinity_key_follows_program_identity_precedence() {
        let canonical = serde_json::json!({
            "vllm_xargs": {"agentic_context": {"program_id": "task-a:researcher"}},
            "session_id": "ignored"
        });
        let key = AffinityKey::from_request(None, Some(&canonical), "model").unwrap();
        assert_eq!(
            (key.source, key.identity.as_str()),
            ("agentic_context", "task-a:researcher")
        );

        let agent_hint = serde_json::json!({
            "agent_hint": {"session_id": "hint-session"}
        });
        let key = AffinityKey::from_request(
            Some(&header_map("x-session-id", "header-session")),
            Some(&agent_hint),
            "model",
        )
        .unwrap();
        assert_eq!(
            (key.source, key.identity.as_str()),
            ("agent_hint", "hint-session")
        );

        let mut claude = header_map("x-claude-code-session-id", "claude-session");
        claude.insert("x-claude-code-agent-id", "researcher".parse().unwrap());
        let key = AffinityKey::from_request(Some(&claude), None, "model").unwrap();
        assert_eq!(
            (key.source, key.identity.as_str()),
            ("framework_header", "claude-session:researcher")
        );

        let mut codex = header_map("session-id", "codex-session");
        codex.insert("thread-id", "worker".parse().unwrap());
        let key = AffinityKey::from_request(Some(&codex), None, "model").unwrap();
        assert_eq!(key.identity, "codex-session:worker");

        let body = serde_json::json!({"session_id": "body-session"});
        let key = AffinityKey::from_request(
            Some(&header_map("x-session-id", "generic-session")),
            Some(&body),
            "model",
        )
        .unwrap();
        assert_eq!(key.identity, "generic-session:lead");

        let key = AffinityKey::from_request(None, Some(&body), "model").unwrap();
        assert_eq!(
            (key.source, key.identity.as_str()),
            ("body_session", "body-session")
        );
        let params = serde_json::json!({"session_params": {"session_id": "params-session"}});
        let key = AffinityKey::from_request(None, Some(&params), "model").unwrap();
        assert_eq!(key.identity, "params-session");
    }

    #[test]
    fn affinity_key_rejects_unstable_and_malformed_identities() {
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", "request-1".parse().unwrap());
        headers.insert("x-trace-id", "trace-1".parse().unwrap());
        assert!(AffinityKey::from_request(Some(&headers), None, "model").is_none());

        let user_scoped = serde_json::json!({"user": "user-a", "user_id": "user-a"});
        assert!(AffinityKey::from_request(None, Some(&user_scoped), "model").is_none());

        let malformed_hint = serde_json::json!({"agent_hint": {"session_id": []}});
        assert!(AffinityKey::from_request(None, Some(&malformed_hint), "model").is_none());

        let malformed_context = serde_json::json!({
            "vllm_xargs": {"agentic_context": {"task_id": [], "agent_id": "worker"}}
        });
        assert!(AffinityKey::from_request(None, Some(&malformed_context), "model").is_none());
        for params in [serde_json::json!({"session_id": []}), serde_json::json!([])] {
            let body = serde_json::json!({"session_params": params, "session_id": "fallback"});
            assert!(AffinityKey::from_request(None, Some(&body), "model").is_none());
        }
        for params in [serde_json::json!({}), JsonValue::Null] {
            let body = serde_json::json!({"session_params": params, "session_id": "fallback"});
            assert_eq!(
                AffinityKey::from_request(None, Some(&body), "model")
                    .unwrap()
                    .identity,
                "fallback"
            );
        }
    }

    #[test]
    fn affinity_digest_scopes_model_lora_salt_and_identity_kind() {
        let body = serde_json::json!({"session_id": "session-a"});
        let key = AffinityKey::from_request(None, Some(&body), "model-a").unwrap();
        let header_key = AffinityKey::from_request(
            Some(&header_map("x-session-id", "session-a")),
            None,
            "model-a",
        )
        .unwrap();
        assert_ne!(
            key.scoped_digest(None, None),
            header_key.scoped_digest(None, None)
        );

        let other_model = AffinityKey::from_request(None, Some(&body), "model-b").unwrap();
        assert_ne!(
            key.scoped_digest(None, None),
            other_model.scoped_digest(None, None)
        );
        assert_ne!(
            key.scoped_digest(Some("adapter"), None),
            key.scoped_digest(None, None)
        );
        assert_ne!(
            key.scoped_digest(None, Some("salt")),
            key.scoped_digest(None, None)
        );
    }

    fn affinity_digest(index: u64) -> [u8; 32] {
        let mut digest = [0u8; 32];
        digest[..8].copy_from_slice(&index.to_le_bytes());
        digest
    }

    fn small_affinity_table() -> KvAffinityTable {
        KvAffinityTable {
            records: Mutex::new(LruCache::new(NonZeroUsize::new(2).unwrap())),
            ..KvAffinityTable::new()
        }
    }

    #[test]
    fn affinity_table_evicts_lru_idle_at_capacity() {
        let table = small_affinity_table();
        let target = target("worker", 0);
        for index in 0..2 {
            table
                .bind(affinity_digest(index), target.clone(), 1)
                .unwrap()
                .release_ref();
        }
        table.acquire(&affinity_digest(0)).unwrap().release_ref();
        assert!(table.bind(affinity_digest(2), target, 1).is_some());
        assert_eq!(table.len(), 2);
        assert!(table.get(&affinity_digest(0)).is_some());
        assert!(table.get(&affinity_digest(1)).is_none());
    }

    #[test]
    fn affinity_table_bypasses_when_all_records_are_in_flight() {
        let table = small_affinity_table();
        let target = target("worker", 0);
        let old = table.bind(affinity_digest(0), target.clone(), 1).unwrap();
        table.bind(affinity_digest(1), target.clone(), 1).unwrap();
        assert!(table.bind(affinity_digest(2), target.clone(), 1).is_none());

        // Replacing an active key preserves capacity and owns its own references.
        let replacement = table.bind(affinity_digest(0), target.clone(), 2).unwrap();
        old.release_ref();
        assert_eq!(replacement.state.lock().in_flight, 1);
        assert_eq!(table.get(&affinity_digest(0)).unwrap().generation, 2);
        assert_eq!(table.len(), 2);
        replacement.release_ref();
        assert!(table.bind(affinity_digest(2), target.clone(), 1).is_some());
        table.remove_target(&target);
        assert!(table.is_empty());
    }

    #[test]
    fn affinity_ttl_expires_after_last_in_flight_reference_is_released() {
        let ttl = Duration::from_secs(60);
        let table = KvAffinityTable::with_ttl(Some(ttl));
        let digest = affinity_digest(7);
        let first = table.bind(digest, target("worker", 0), 1).unwrap();
        first.state.lock().idle_deadline = Some(Instant::now() - ttl);
        let second = table
            .acquire(&digest)
            .expect("in-flight binding stays valid");
        first.release_ref();
        assert!(!second.is_expired(Instant::now() + ttl * 2));

        let released_at = Instant::now();
        second.release_ref();
        assert!(!second.is_expired(released_at + ttl));
        let deadline = second.state.lock().idle_deadline.unwrap();
        assert!(second.is_expired(deadline));
        second.state.lock().idle_deadline = Some(Instant::now() - ttl);
        assert!(table.acquire(&digest).is_none());
    }

    #[test]
    fn affinity_concurrent_release_and_acquire_preserve_binding() {
        let ttl = Duration::from_secs(60);
        let table = KvAffinityTable::with_ttl(Some(ttl));
        let digest = affinity_digest(8);
        let record = table.bind(digest, target("worker", 0), 1).unwrap();
        record.state.lock().idle_deadline = Some(Instant::now() - ttl);
        let ready = std::sync::Barrier::new(2);
        let acquired = std::thread::scope(|scope| {
            let acquire = scope.spawn(|| {
                ready.wait();
                table.acquire(&digest).expect("binding must not expire")
            });
            ready.wait();
            record.release_ref();
            acquire.join().unwrap()
        });
        assert!(Arc::ptr_eq(&record, &acquired));
        assert_eq!(record.state.lock().in_flight, 1);
        let released_at = Instant::now();
        acquired.release_ref();
        assert!(record.state.lock().idle_deadline.unwrap() >= released_at + ttl);
    }

    #[test]
    fn registration_uses_the_discovered_cache_block_size_and_provider() {
        #[derive(Default)]
        struct QueryBlocks(AtomicUsize, Mutex<Option<Box<dyn FnOnce() + Send>>>);
        impl TieredMatchProvider for QueryBlocks {
            fn find_tiered_matches(&self, query: &MatchQuery) -> Vec<TierMatch> {
                self.0.store(query.local_hashes.len(), Ordering::Relaxed);
                if let Some(mutate) = self.1.lock().take() {
                    mutate();
                }
                Vec::new()
            }
        }
        let policy = Arc::new(KvAwarePolicy::default());
        let ops = kv_aware_ops(policy.as_ref()).unwrap();
        let worker: Arc<dyn Worker> = Arc::new(DPAwareWorker::new(
            "grpc://worker:50051".into(),
            1,
            2,
            WorkerType::Regular,
        ));
        let matches = Arc::new(QueryBlocks::default());
        let info = WorkerKvInfo {
            instance_id: "engine-instance".into(),
            cache_key: Some(CacheKey {
                model: Arc::from("model"),
                hash_mode: crate::kv_index::HashMode::Sha256,
                block_size: 2,
            }),
            ..WorkerKvInfo::default()
        };
        ops.register_worker(worker.as_ref(), Some(&info), Some(matches.clone()));
        let workers = vec![worker.clone()];
        let select = || {
            ops.select_worker_with_receipt(&workers, Some(&[1, 2, 3, 4, 5]), None, None, None, &[])
        };
        release_receipt(&workers, &select().unwrap());
        assert_eq!(matches.0.load(Ordering::Relaxed), 3);

        // Mutate the live candidate after its snapshot, during a KV query.
        for mutation in 0..3 {
            ops.register_worker(worker.as_ref(), Some(&info), Some(matches.clone()));
            let weak = Arc::downgrade(&policy);
            let worker = worker.clone();
            *matches.1.lock() = Some(Box::new(move || match mutation {
                0 => worker.set_healthy(false),
                1 => weak.upgrade().unwrap().unregister_target(worker.url()),
                _ => weak
                    .upgrade()
                    .unwrap()
                    .register_target(worker.url(), target("restarted", 1)),
            }));
            let selected = select();
            if let Some(receipt) = selected {
                assert_eq!(mutation, 2);
                assert_eq!(receipt.decision.target.instance_id.as_ref(), "restarted");
                release_receipt(&workers, &receipt);
            } else {
                assert_ne!(mutation, 2);
            }
            workers[0].set_healthy(true);
        }
    }

    #[test]
    fn capability_preserves_rank_identity_and_request_lifecycle() {
        let policy = KvAwarePolicy::default();
        let ops = kv_aware_ops(&policy).unwrap();
        let worker: Arc<dyn Worker> = Arc::new(DPAwareWorker::new(
            "grpc://worker:50051".into(),
            2,
            4,
            WorkerType::Regular,
        ));
        // Router registration alone must support affinity, even without KV discovery.
        let key = affinity_key("load-only");
        ops.register_worker(worker.as_ref(), None, None);
        let initial = ops
            .select_worker_with_receipt(
                std::slice::from_ref(&worker),
                None,
                None,
                None,
                Some(&key),
                &[],
            )
            .unwrap();
        let original = policy.affinity.get(&key.scoped_digest(None, None)).unwrap();
        release_receipt(std::slice::from_ref(&worker), &initial);
        ops.unregister_worker(worker.url());
        assert!(policy.affinity.is_empty());
        ops.register_worker(
            worker.as_ref(),
            Some(&WorkerKvInfo {
                instance_id: "engine-instance".into(),
                ..WorkerKvInfo::default()
            }),
            None,
        );
        let target = policy.target_for_worker(worker.as_ref());
        assert_eq!(target.instance_id.as_ref(), "engine-instance");
        assert_eq!(target.dp_rank, 2);
        let manager = policy.slot_manager().unwrap();
        let receipt = ops
            .select_worker_with_receipt(
                std::slice::from_ref(&worker),
                Some(&[1, 2, 3]),
                None,
                None,
                Some(&key),
                &[],
            )
            .unwrap();
        assert_ne!(
            policy
                .affinity
                .get(&key.scoped_digest(None, None))
                .unwrap()
                .generation,
            original.generation
        );
        assert_eq!(receipt.decision.target, target);
        assert_eq!(worker.load(), 1);
        let tokens = receipt.new_prefill_tokens;
        let slot = receipt.slot.expect("selection reserves a request slot");
        assert_eq!(
            manager.sample(&target),
            LoadSample {
                active_prefill_tokens: tokens,
                active_requests: 1,
            }
        );
        let cloned_slot = slot.clone();
        slot.begin_decode();
        cloned_slot.begin_decode();
        assert_eq!(
            manager.sample(&target),
            LoadSample {
                active_prefill_tokens: 0,
                active_requests: 1,
            }
        );
        slot.release();
        cloned_slot.release();
        assert_eq!(manager.sample(&target), LoadSample::default());
        ops.unregister_worker(worker.url());
        assert_eq!(
            policy
                .target_for_worker(worker.as_ref())
                .instance_id
                .as_ref(),
            worker.base_url()
        );
    }
}
