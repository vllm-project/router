//! Exact KV-placement and live-load aware worker selection.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use lru::LruCache;
use parking_lot::Mutex;
use rand::Rng;
use sha2::{Digest as _, Sha256};

use super::{get_healthy_worker_indices, pick_tied, LoadBalancingPolicy, RequestHeaders};
use crate::core::Worker;
use crate::kv_index::prefix_history::{CompletionHistory, HistoryHit};
use crate::kv_index::{
    CacheKey, PromptKey, RequestHashCache, RoutableTarget, SharedPrefixLookup, StorageTier,
    TierMatch, WorkerKvInfo,
};
use crate::request_identity::StableIdentity;

pub use crate::kv_index::TieredMatchProvider;

mod history;
mod history_hook;
pub use crate::kv_index::prefix_history::HistoryObservation;
pub(crate) use history_hook::{HistoryAttempt, HistoryCapture, KvAwareHistoryHook};

/// Confirms that a rank still belongs to the owner captured at registration.
/// Temporary loss of placement trust does not end that registration.
pub trait RegistrationCurrent: Send + Sync {
    fn current(&self, rank: u32) -> bool;
}

/// Registrations captured before input preparation. A retry must not attach
/// their tokens to another Worker object or a replacement at the same address.
pub struct CompletionHistoryRequest {
    registrations: HashSet<u64>,
}

impl CompletionHistoryRequest {
    fn registration_id(&self, worker: &dyn Worker, binding: &TargetBinding) -> Option<u64> {
        (self.registrations.contains(&binding.registration_id) && binding.is_history_worker(worker))
            .then_some(binding.registration_id)
    }
}

// Process-wide identity, allocated for every registration including load-only
// targets. Never reuse an ID: even unreachable overflow fails closed.
static NEXT_REGISTRATION: AtomicU64 = AtomicU64::new(1);

/// Exact prompt sequences rendered by each HTTP backend. DP ranks share their
/// base URL's input; separate prompts in a batch never form one prefix chain.
pub type WorkerTokenInputs = HashMap<String, Arc<[Vec<u32>]>>;

#[derive(Clone, Copy)]
pub enum KvRequestTokens<'a> {
    Shared(Option<&'a [u32]>),
    PerWorker(&'a WorkerTokenInputs),
}

/// Request-local placement qualification; it restricts credit, not candidacy.
#[derive(Clone, Copy)]
pub(crate) struct QualifiedKvInput<'a> {
    pub(crate) tokens: KvRequestTokens<'a>,
    pub(crate) eligibility: Option<&'a [bool]>,
}

/// Cache and affinity scope of the actual request being selected.
#[derive(Default)]
pub struct SelectionIdentity<'a> {
    pub lora_name: Option<&'a str>,
    pub cache_salt: Option<&'a str>,
    pub affinity_key: Option<&'a AffinityKey>,
}

impl QualifiedKvInput<'_> {
    fn allows_credit(self, index: usize, worker_count: usize) -> bool {
        self.eligibility
            .is_none_or(|mask| mask.len() == worker_count && mask.get(index) == Some(&true))
    }
}

/// Prompts for one worker, borrowing the request's token storage.
pub(crate) enum WorkerPrompts<'a> {
    Single(&'a [u32]),
    Batch(&'a [Vec<u32>]),
}

impl<'a> WorkerPrompts<'a> {
    fn len(&self) -> usize {
        match self {
            Self::Single(_) => 1,
            Self::Batch(prompts) => prompts.len(),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn iter(&self) -> impl ExactSizeIterator<Item = &'a [u32]> + '_ {
        (0..self.len()).map(|i| match self {
            Self::Single(tokens) => *tokens,
            Self::Batch(prompts) => prompts[i].as_slice(),
        })
    }

    fn single(&self) -> Option<&'a [u32]> {
        match self {
            Self::Single(tokens) => Some(tokens),
            Self::Batch([tokens]) => Some(tokens),
            Self::Batch(_) => None,
        }
    }

    fn input_tokens(&self) -> Option<usize> {
        (!self.is_empty()).then(|| self.iter().map(<[u32]>::len).fold(0, usize::saturating_add))
    }
}

impl<'a> KvRequestTokens<'a> {
    pub(crate) fn prompts_for(self, worker: &dyn Worker) -> WorkerPrompts<'a> {
        match self {
            Self::Shared(Some(ids)) => WorkerPrompts::Single(ids),
            Self::Shared(None) => WorkerPrompts::Batch(&[]),
            Self::PerWorker(inputs) => {
                WorkerPrompts::Batch(inputs.get(worker.base_url()).map_or(&[], AsRef::as_ref))
            }
        }
    }
}

/// Immutable P work estimate from one placement snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillWorkEstimate {
    pub new_prefill_tokens: u64,
    pub placement_verified: bool,
}

/// Marginal Decode context from one placement snapshot: what the request adds
/// to a candidate after its own cached blocks are reused.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecodeWorkEstimate {
    /// Logical context tokens the candidate must newly allocate: the prompt
    /// once per request - sequences share that copy - less the device blocks it already
    /// holds.
    /// `None` when its cache identity is unverified, so the caller falls back
    /// to plain occupancy instead of comparing a credited figure against a
    /// full-price one.
    pub additional_tokens: Option<u64>,
}

/// Optional operations used by adapters with tokenized requests and KV discovery.
pub trait KvAwareOps: Send + Sync {
    /// P/D uses the same placement/credit contract as colocated routing,
    /// but accounts token backlog in the shared P/D admission manager.
    fn estimate_prefill_work<'tokens>(
        &self,
        inputs: &[(&dyn Worker, &'tokens [Vec<u32>], bool)],
        hashes: &mut PromptHashes<'tokens>,
    ) -> Vec<PrefillWorkEstimate>;
    /// Marginal Decode context per candidate, from the same placement snapshot
    /// as [`Self::estimate_prefill_work`]. `inputs` pairs each candidate with
    /// the prompt stream it will hold and whether its cache identity is
    /// verified; the returned estimate is `None` for an unverified candidate.
    ///
    /// `hashes` is shared with the Prefill estimate so the same prompt is
    /// hashed once per request rather than once per stage.
    fn estimate_decode_work<'tokens>(
        &self,
        inputs: &[(&dyn Worker, &'tokens [Vec<u32>], bool)],
        hashes: &mut PromptHashes<'tokens>,
    ) -> Vec<DecodeWorkEstimate>;
    #[doc(hidden)]
    fn estimate_decode_work_with_transfer<'tokens>(
        &self,
        inputs: &[(&dyn Worker, &'tokens [Vec<u32>], bool)],
        hashes: &mut PromptHashes<'tokens>,
        _nixl_pull: bool,
    ) -> Vec<DecodeWorkEstimate> {
        self.estimate_decode_work(inputs, hashes)
    }
    fn pick_prefill(&self, costs: &[f64], occupancy: &[usize]) -> Option<usize>;
    /// Decode-side occupancy picker for P/D: deterministic argmin over the
    /// projected stage occupancies, rotating across equal-load targets so
    /// consecutive new sessions do not converge on one decoder.
    fn pick_decode(&self, loads: &[usize], additional: &[u64]) -> Option<usize>;
    /// Minimum-occupancy decode selection with optional soft session affinity.
    /// P/D admission serializes selection and reservation; the returned binding
    /// reference remains owned by the attempt until cleanup.
    ///
    /// `new_context` is, per candidate, the context the request must newly
    /// allocate there: its prompt length less the blocks that candidate already
    /// holds, so a Decode engine that can reuse a cached prefix is charged only
    /// for what it has to add. It is computed before admission (the KV index is
    /// never queried under the selection lock). A table that misses any present
    /// candidate is treated as `None`, so a partial list cannot compare
    /// candidates in different units: a missing entry read as zero would make a
    /// recovered worker look free.
    ///
    /// `new_context` is keyed by worker URL, so it survives the caller
    /// projecting each candidate onto a load-only handle.
    fn select_decode(
        &self,
        workers: &[Arc<dyn Worker>],
        session: Option<&AffinityKey>,
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
        new_context: Option<&HashMap<String, u64>>,
    ) -> Option<StagePick>;
    /// Whether the optional session binding is switched on for this policy.
    fn session_affinity_enabled(&self) -> bool;
    fn register_worker(
        &self,
        worker: &dyn Worker,
        info: Option<&WorkerKvInfo>,
        matches: Option<Arc<dyn TieredMatchProvider>>,
    );
    fn unregister_worker(&self, worker_url: &str);
    /// Register the actual Worker object atomically with its generation.
    /// Implementations without history retain ordinary registration behavior.
    fn register_history_worker(
        &self,
        worker: &Arc<dyn Worker>,
        info: Option<&WorkerKvInfo>,
        matches: Option<Arc<dyn TieredMatchProvider>>,
        _current: Option<Arc<dyn RegistrationCurrent>>,
    ) {
        self.register_worker(worker.as_ref(), info, matches);
    }

    /// Capture the eligible registrations before any input preparation RPC.
    fn capture_completion_history(
        &self,
        _workers: &[Arc<dyn Worker>],
        _requested_model: &str,
    ) -> Option<CompletionHistoryRequest> {
        None
    }

    /// Commit once at the first verified generated output, without owning load.
    /// HTTP and gRPC both require output evidence, not response headers or stream opening.
    fn observe_completion_milestone(&self, _worker: &dyn Worker, _observation: HistoryObservation) {
    }

    /// Ordinary and history-assisted requests share selection and reservation.
    /// History applies only to requests without a LoRA or cache-salt identity.
    /// Implementers without history keep their existing qualification behavior.
    fn select_with_history(
        &self,
        workers: &[Arc<dyn Worker>],
        tokens: KvRequestTokens<'_>,
        eligibility: Option<&[bool]>,
        _history: Option<&CompletionHistoryRequest>,
        identity: SelectionIdentity<'_>,
        excluded_workers: &[String],
    ) -> Option<(KvSelectionReceipt, Option<HistoryObservation>)> {
        self.select_worker_with_qualified_inputs(
            workers,
            tokens,
            identity.lora_name,
            identity.cache_salt,
            identity.affinity_key,
            excluded_workers,
            eligibility,
        )
        .map(|receipt| (receipt, None))
    }
    /// Preparation hint only; selection revalidates the binding before reserving.
    fn has_affinity(
        &self,
        workers: &[Arc<dyn Worker>],
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
        affinity_key: &AffinityKey,
    ) -> bool;
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

    fn select_worker_with_inputs(
        &self,
        workers: &[Arc<dyn Worker>],
        tokens: KvRequestTokens<'_>,
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
        affinity_key: Option<&AffinityKey>,
        excluded_workers: &[String],
    ) -> Option<KvSelectionReceipt>;
    /// Optional placement qualification aligned with the original `workers`
    /// slice, before health filtering. The built-in policy treats `None` as
    /// unrestricted. Implementers without this extension return `None` without
    /// querying or reserving, rather than silently ignoring a supplied mask.
    #[allow(clippy::too_many_arguments)]
    fn select_worker_with_qualified_inputs(
        &self,
        workers: &[Arc<dyn Worker>],
        tokens: KvRequestTokens<'_>,
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
        affinity_key: Option<&AffinityKey>,
        excluded_workers: &[String],
        eligibility: Option<&[bool]>,
    ) -> Option<KvSelectionReceipt> {
        if eligibility.is_some() {
            return None;
        }
        self.select_worker_with_inputs(
            workers,
            tokens,
            lora_name,
            cache_salt,
            affinity_key,
            excluded_workers,
        )
    }
}

/// One stage's selection result: the picked candidate, how session affinity
/// shaped the choice (into `vllm_router_pd_affinity_decisions_total{stage,
/// outcome}`), and the held binding reference for the admission attempt.
#[derive(Debug, Clone)]
pub struct StagePick {
    pub index: usize,
    pub outcome: &'static str,
    pub binding: Option<Arc<AffinityRecord>>,
}

/// Resolve the optional capability at the policy boundary. Keep concrete policy
/// discovery here so adapters and LoadBalancingPolicy stay independent of it.
pub fn kv_aware_ops(policy: &dyn LoadBalancingPolicy) -> Option<&dyn KvAwareOps> {
    kv_aware_policy(policy).map(|policy| policy as &dyn KvAwareOps)
}

pub(crate) fn kv_aware_policy(policy: &dyn LoadBalancingPolicy) -> Option<&KvAwarePolicy> {
    policy.as_any().downcast_ref::<KvAwarePolicy>()
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
    pub fn from_identity(identity: StableIdentity, model_scope: &str) -> Self {
        Self {
            model_scope: model_scope.to_string(),
            source: identity.source.as_str(),
            identity: identity.program_id,
        }
    }

    pub fn scoped_digest(&self, lora_name: Option<&str>, cache_salt: Option<&str>) -> [u8; 32] {
        self.digest_fields(None, lora_name, cache_salt)
    }

    /// Stage-scoped digest for P/D binding: the same session identity never
    /// shares a binding across the prefill and decode stages.
    pub fn staged_digest(
        &self,
        stage: &str,
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
    ) -> [u8; 32] {
        self.digest_fields(Some(stage), lora_name, cache_salt)
    }

    fn digest_fields(
        &self,
        stage: Option<&str>,
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
    ) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for field in [
            AFFINITY_DIGEST_DOMAIN.as_bytes(),
            self.model_scope.as_bytes(),
            self.source.as_bytes(),
            self.identity.as_bytes(),
            stage.unwrap_or_default().as_bytes(),
            lora_name.unwrap_or_default().as_bytes(),
            cache_salt.unwrap_or_default().as_bytes(),
        ] {
            hasher.update((field.len() as u64).to_le_bytes());
            hasher.update(field);
        }
        hasher.finalize().into()
    }
}

/// One affinity binding. Guards hold direct `Arc` references so a record
/// replaced mid-flight still releases its own in-flight count.
#[derive(Debug)]
pub struct AffinityRecord {
    target: RoutableTarget,
    generation: u64,
    state: Mutex<AffinityState>,
    ttl: Option<Duration>,
}

#[derive(Debug)]
struct AffinityState {
    in_flight: usize,
    idle_deadline: Option<Instant>,
}

impl AffinityRecord {
    pub fn target(&self) -> &RoutableTarget {
        &self.target
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn in_flight(&self) -> usize {
        self.state.lock().in_flight
    }

    pub fn release_ref(&self) {
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
pub struct KvAffinityTable {
    records: Mutex<LruCache<[u8; 32], Arc<AffinityRecord>>>,
    ttl: Option<Duration>,
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

    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    #[cfg_attr(coverage_nightly, coverage(off))]
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

    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn len(&self) -> usize {
        self.records.lock().len()
    }

    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    #[cfg_attr(coverage_nightly, coverage(off))]
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
    let device_blocks = crate::kv_index::layout::full_attention_prefix(
        candidate
            .matches
            .iter()
            .filter(|hit| hit.tier == StorageTier::Device)
            .flat_map(|hit| hit.segments.iter().copied()),
    )
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

    let mut best_continuation = 0.0_f64;
    for hit in candidate
        .matches
        .iter()
        .filter(|hit| hit.tier != StorageTier::Device)
    {
        for &(start, end) in &hit.segments {
            if start <= device_blocks && device_blocks < end {
                let blocks = end - device_blocks;
                let credit = params.transfer_credits.for_tier(hit.tier);
                best_continuation = best_continuation.max(f64::from(blocks) * credit);
            }
        }
    }

    let credited_blocks = f64::from(device_blocks) + best_continuation;
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
    remaining_work(candidate.input_tokens, credited_tokens(candidate, params))
}

fn remaining_work(input_tokens: usize, credit: f64) -> u64 {
    (input_tokens as f64 - credit).max(0.0).ceil() as u64
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
            // Token work and request count model separate costs: prefilling
            // the tokens and competing with other requests in the projected batch.
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
            return pick_minimum(costs);
        }

        let min = costs.iter().copied().reduce(f64::min)?;
        let weights: Vec<f64> = costs
            .iter()
            .map(|cost| (-(cost - min) / self.temperature).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return pick_minimum(costs);
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

/// Picks uniformly among the candidates that share the lowest cost.
///
/// The ranking still decides which candidates are eligible; chance only
/// resolves scores that are exactly equal.
fn pick_minimum(costs: &[f64]) -> Option<usize> {
    let min = *costs.iter().min_by(|left, right| left.total_cmp(right))?;
    let tied: Vec<usize> = costs
        .iter()
        .enumerate()
        .filter(|(_, cost)| **cost == min)
        .map(|(index, _)| index)
        .collect();
    pick_tied(&tied)
}

#[derive(Debug, Default)]
struct EmptyMatchProvider;

impl TieredMatchProvider for EmptyMatchProvider {}

struct TargetBinding {
    target: RoutableTarget,
    // Policy-local lifecycle fence used by selection and affinity.
    generation: u64,
    // Process-wide registration identity used by advisory history.
    registration_id: u64,
    cache_key: Option<CacheKey>,
    matches: Arc<dyn TieredMatchProvider>,
    shared_prefix: Option<SharedPrefixLookup>,
    block_size: u32,
    history_worker: Option<std::sync::Weak<dyn Worker>>,
    history_current: Option<Arc<dyn RegistrationCurrent>>,
}

impl TargetBinding {
    fn is_history_worker(&self, worker: &dyn Worker) -> bool {
        // Keep this allocation's address from being recycled. Registration ID
        // additionally fences re-registration of the same Worker object.
        self.history_worker
            .as_ref()
            .is_some_and(|registered| std::ptr::addr_eq(registered.as_ptr(), worker))
            && self.history_registration_current()
    }

    fn history_registration_current(&self) -> bool {
        self.history_current
            .as_ref()
            .is_some_and(|owner| owner.current(self.target.dp_rank))
    }
}

/// Registered workers share a snapshot. An unregistered fallback lives in the request.
enum WorkerBinding {
    Registered(Arc<TargetBinding>),
    Unregistered(TargetBinding),
}

impl std::ops::Deref for WorkerBinding {
    type Target = TargetBinding;
    fn deref(&self) -> &TargetBinding {
        match self {
            Self::Registered(registered) => registered,
            Self::Unregistered(fallback) => fallback,
        }
    }
}

#[derive(PartialEq, Eq, Hash)]
enum MatchScope {
    Registered(u64),
    Unregistered(RoutableTarget),
    Shared(ProviderIdentity, Option<CacheKey>),
}

// Own the allocation so a later registration cannot reuse its memo address.
struct ProviderIdentity(Arc<dyn TieredMatchProvider>);

impl PartialEq for ProviderIdentity {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ProviderIdentity {}

impl std::hash::Hash for ProviderIdentity {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(Arc::as_ptr(&self.0).cast::<()>(), state);
    }
}

/// Reuse prompt hashes across P/D and scored matches within one policy stage.
#[derive(Default)]
pub struct PromptHashes<'a> {
    // Each stage scores a single adapter identity for the entire request.
    matches: HashMap<(MatchScope, PromptKey<'a>), Vec<TierMatch>, foldhash::fast::RandomState>,
    hashes: RequestHashCache<'a>,
}

impl PromptHashes<'_> {
    pub fn new() -> Self {
        Self::default()
    }

    fn set_stage_targets(
        &mut self,
        inputs: &[(&dyn Worker, &[Vec<u32>], bool)],
        count_replay_pages: bool,
    ) {
        // Events and target registrations can change between prefill and decode.
        self.matches.clear();
        self.hashes.first_trust_check = None;
        self.hashes.stop_at_gap = true;
        self.hashes.count_replay_pages = count_replay_pages;
        self.hashes.single_target = inputs
            .iter()
            .filter(|(_, prompts, credit)| *credit && !prompts.is_empty())
            .nth(1)
            .is_none();
    }
}

fn all_unique<T: Eq + std::hash::Hash>(mut ids: impl ExactSizeIterator<Item = T>) -> bool {
    if ids.len() < 2 {
        return true;
    }
    let mut seen =
        HashSet::with_capacity_and_hasher(ids.len(), foldhash::fast::RandomState::default());
    ids.all(|id| seen.insert(id))
}

/// One prepared selection snapshot: KV queries fill `grouped` outside every
/// lock; the policy's short critical section then samples load, scores, and
/// reserves against this plan.
struct SelectionPlan {
    healthy: Vec<usize>,
    bindings: Vec<WorkerBinding>,
    grouped: Vec<Vec<TierMatch>>,
    input_tokens: Vec<usize>,
    prefill_tokens: Vec<Option<u64>>,
    // An integer token charge can hide fractional transfer credit.
    // Populated only when an input history ticket is present.
    history_credit: Vec<Option<bool>>,
    has_tokens: bool,
}

type HistoryCandidate<'a> = (u64, &'a [u32], Option<HistoryHit>);

/// Registered `kv_aware` policy. Discovery binds each worker target to the
/// placement provider for its cache identity.
pub struct KvAwarePolicy {
    matches: Arc<dyn TieredMatchProvider>,
    load: Arc<dyn LoadProvider>,
    slot_manager: Option<Arc<dyn SlotManager>>,
    scorers: Vec<Arc<dyn Scorer>>,
    prefix_scoring: bool,
    picker: Arc<dyn Picker>,
    params: CostParams,
    targets: DashMap<String, Arc<TargetBinding>, foldhash::fast::RandomState>,
    /// Round-robin cursor across equal-load decoders, so a stable first-index
    /// tie-break cannot magnetize every unbound session onto one decoder.
    decode_tie_cursor: AtomicU64,
    /// Cross-request session bindings, shared across the router's policies.
    affinity: Arc<KvAffinityTable>,
    /// Serializes load sampling, selection, and reservation.
    selection_lock: Mutex<()>,
    /// Policy-local lifecycle counter, also advanced for load-only targets.
    next_generation: AtomicU64,
    completion_history: OnceLock<Arc<CompletionHistory>>,
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
    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(crate) fn observed_completion_prefix(&self, worker: &dyn Worker, ids: &[u32]) -> usize {
        let now = Instant::now();
        let Some(store) = self.completion_history.get() else {
            return 0;
        };
        let Some((registration_id, block_size)) = self
            .targets
            .get(worker.url())
            .filter(|b| b.is_history_worker(worker))
            .map(|b| (b.registration_id, b.block_size))
        else {
            return 0;
        };
        store
            .match_prefix(registration_id, ids, block_size, now)
            .map_or(0, |hit| hit.len_at(now))
    }

    fn capture_history_registrations(
        &self,
        workers: &[Arc<dyn Worker>],
        requested_model: &str,
    ) -> CompletionHistoryRequest {
        let registrations = workers
            .iter()
            .filter_map(|worker| {
                let b = self.targets.get(worker.url())?;
                (worker.worker_type() == crate::core::WorkerType::Regular
                    && worker.dp_rank().is_none()
                    && b.target.dp_rank == 0
                    && b.is_history_worker(worker.as_ref())
                    && requested_model != worker.base_url()
                    && requested_model != worker.url()
                    && b.cache_key.as_ref()?.model.as_ref() == requested_model)
                    .then_some(b.registration_id)
            })
            .collect();
        self.completion_history.get_or_init(Default::default);
        CompletionHistoryRequest { registrations }
    }

    fn commit_history_observation(&self, worker: &dyn Worker, observation: HistoryObservation) {
        // Keep this read guard through publication so replacement either
        // removes this insertion or makes its old registration fail the check.
        let Some(binding) = self.targets.get(worker.url()) else {
            return;
        };
        let Some(store) = self.completion_history.get() else {
            return;
        };
        if binding.registration_id == observation.registration_id
            && binding.is_history_worker(worker)
        {
            store.insert(observation, binding.block_size, Instant::now());
        }
    }

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
        policy.prefix_scoring = true;
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
            prefix_scoring: false,
            picker,
            params,
            targets: DashMap::default(),
            decode_tie_cursor: AtomicU64::new(0),
            affinity: Arc::new(KvAffinityTable::new()),
            selection_lock: Mutex::new(()),
            next_generation: AtomicU64::new(1),
            completion_history: OnceLock::new(),
        }
    }

    pub fn register_target(&self, worker_url: impl Into<String>, target: RoutableTarget) {
        self.register_binding(
            worker_url.into(),
            target,
            None,
            self.matches.clone(),
            None,
            None,
        );
    }

    pub fn register_target_with_cache_identity(
        &self,
        worker_url: impl Into<String>,
        target: RoutableTarget,
        cache_key: CacheKey,
        matches: Arc<dyn TieredMatchProvider>,
    ) {
        self.register_binding(
            worker_url.into(),
            target,
            Some(cache_key),
            matches,
            None,
            None,
        );
    }

    fn register_binding(
        &self,
        worker_url: String,
        target: RoutableTarget,
        cache_key: Option<CacheKey>,
        matches: Arc<dyn TieredMatchProvider>,
        history_worker: Option<std::sync::Weak<dyn Worker>>,
        history_current: Option<Arc<dyn RegistrationCurrent>>,
    ) {
        let block_size = cache_key
            .as_ref()
            .map_or(self.params.block_size, |key| key.block_size);
        let shared_prefix = cache_key
            .as_ref()
            .and_then(|_| matches.shared_prefix_lookup(target.dp_rank));
        let _guard = self.selection_lock.lock();
        let old = self.targets.insert(
            worker_url,
            Arc::new(TargetBinding {
                target,
                generation: self.next_generation.fetch_add(1, Ordering::AcqRel),
                registration_id: NEXT_REGISTRATION
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
                    .expect("registration identity exhausted"),
                cache_key,
                matches,
                shared_prefix,
                block_size,
                history_worker,
                history_current,
            }),
        );
        if let (Some(old), Some(history)) = (old, self.completion_history.get()) {
            history.remove_registration(old.registration_id);
        }
    }

    fn register_worker_binding(
        &self,
        worker: &dyn Worker,
        info: Option<&WorkerKvInfo>,
        matches: Option<Arc<dyn TieredMatchProvider>>,
        history_worker: Option<std::sync::Weak<dyn Worker>>,
        history_current: Option<Arc<dyn RegistrationCurrent>>,
    ) {
        let target = RoutableTarget {
            instance_id: Arc::from(
                info.map(|info| info.instance_id.as_str())
                    .filter(|id| !id.is_empty())
                    .unwrap_or_else(|| worker.base_url()),
            ),
            dp_rank: worker.dp_rank().unwrap_or(0) as u32,
        };
        let (cache_key, matches) = match (info.and_then(|info| info.cache_key.clone()), matches) {
            (Some(cache_key), Some(matches)) => (Some(cache_key), matches),
            _ => (None, self.matches.clone()),
        };
        let history_worker = cache_key.as_ref().and(history_worker);
        let history_current = cache_key.as_ref().and(history_current);
        self.register_binding(
            worker.url().to_owned(),
            target,
            cache_key,
            matches,
            history_worker,
            history_current,
        );
    }

    pub fn unregister_target(&self, worker_url: &str) {
        let _guard = self.selection_lock.lock();
        if let Some((_, binding)) = self.targets.remove(worker_url) {
            if let Some(history) = self.completion_history.get() {
                history.remove_registration(binding.registration_id);
            }
            if let Some(slot_manager) = &self.slot_manager {
                slot_manager.unregister_target(&binding.target);
            }
            // Sessions bound to the removed target fail validation on their
            // next request and reselect by their stage's own score.
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

    /// Position of a record's target among the candidates while its
    /// registration generation is still current and the target is available.
    fn binding_position(
        &self,
        record: &AffinityRecord,
        workers: &[Arc<dyn Worker>],
    ) -> Option<usize> {
        workers.iter().position(|worker| {
            worker.is_available()
                && self.targets.get(worker.url()).is_some_and(|binding| {
                    binding.target == *record.target()
                        && binding.generation != 0
                        && binding.generation == record.generation()
                })
        })
    }

    fn binding_for_worker(&self, worker: &dyn Worker) -> WorkerBinding {
        self.targets
            .get(worker.url())
            .map(|entry| WorkerBinding::Registered(entry.value().clone()))
            .unwrap_or_else(|| {
                WorkerBinding::Unregistered(TargetBinding {
                    target: RoutableTarget {
                        instance_id: Arc::from(worker.base_url()),
                        dp_rank: worker.dp_rank().unwrap_or(0) as u32,
                    },
                    generation: 0,
                    registration_id: 0,
                    cache_key: None,
                    matches: self.matches.clone(),
                    shared_prefix: None,
                    block_size: self.params.block_size,
                    history_worker: None,
                    history_current: None,
                })
            })
    }

    /// A registered, target-scoped provider needs no memo when this stage has
    /// one prompt and can visit the registration only once.
    fn cached_matches<'a, 'tokens>(
        cache: &'a mut PromptHashes<'tokens>,
        lookups: &mut crate::metrics::PlacementLookups,
        binding: &TargetBinding,
        tokens: &'tokens [u32],
        lora_name: Option<&str>,
        fallback: Option<&'static str>,
        single_use: impl FnOnce() -> bool,
    ) -> Cow<'a, [TierMatch]> {
        let tokens = cache.hashes.canonical_prompt(tokens);
        let scoped = binding.matches.target_scoped();
        let mut lookup = |hashes: &mut RequestHashCache<'tokens>| {
            if let Some(reason) = fallback {
                lookups.record(reason);
                return Vec::new();
            }
            if tokens.is_empty() {
                return Vec::new();
            }
            let mut hits = binding.matches.matches_with_cache(
                tokens,
                binding.block_size,
                lora_name,
                binding.target.dp_rank,
                hashes,
            );
            lookups.record(if binding.cache_key.is_none() {
                "index_unavailable"
            } else if hits.is_empty() {
                "miss"
            } else {
                "hit"
            });
            if scoped {
                hits.retain(|hit| hit.target == binding.target);
            }
            hits
        };
        if scoped && binding.generation != 0 && single_use() {
            return Cow::Owned(lookup(&mut cache.hashes));
        }
        let scope = if !scoped {
            MatchScope::Shared(
                ProviderIdentity(binding.matches.clone()),
                binding.cache_key.clone(),
            )
        } else if binding.generation != 0 {
            MatchScope::Registered(binding.generation)
        } else {
            MatchScope::Unregistered(binding.target.clone())
        };
        let hits = cache
            .matches
            .entry((scope, PromptKey(tokens)))
            .or_insert_with(|| lookup(&mut cache.hashes));
        if scoped || hits.is_empty() {
            Cow::Borrowed(hits)
        } else {
            Cow::Owned(
                hits.iter()
                    .filter(|hit| hit.target == binding.target)
                    .cloned()
                    .collect(),
            )
        }
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
            QualifiedKvInput {
                tokens: KvRequestTokens::Shared(token_ids),
                eligibility: None,
            },
            lora_name,
            cache_salt,
            false,
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
        self.select_worker_with_qualified_inputs(
            workers,
            QualifiedKvInput {
                tokens,
                eligibility: None,
            },
            lora_name,
            cache_salt,
            affinity_key,
            excluded_workers,
        )
    }

    pub(crate) fn select_worker_with_qualified_inputs(
        &self,
        workers: &[Arc<dyn Worker>],
        input: QualifiedKvInput<'_>,
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
        affinity_key: Option<&AffinityKey>,
        excluded_workers: &[String],
    ) -> Option<KvSelectionReceipt> {
        self.select_with_context(
            workers,
            input,
            SelectionIdentity {
                lora_name,
                cache_salt,
                affinity_key,
            },
            excluded_workers,
            None,
        )
        .map(|(receipt, _)| receipt)
    }

    fn select_with_context(
        &self,
        workers: &[Arc<dyn Worker>],
        input: QualifiedKvInput<'_>,
        identity: SelectionIdentity<'_>,
        excluded_workers: &[String],
        history: Option<&CompletionHistoryRequest>,
    ) -> Option<(KvSelectionReceipt, Option<HistoryObservation>)> {
        let SelectionIdentity {
            lora_name,
            cache_salt,
            affinity_key,
        } = identity;
        let history = history.filter(|_| {
            self.params.router_temperature == 0.0 && lora_name.is_none() && cache_salt.is_none()
        });
        let digest = affinity_key.map(|key| key.scoped_digest(lora_name, cache_salt));
        // After three stale placement snapshots, score live load and full input
        // work without querying the index or granting KV credit.
        for attempt in 0..=3 {
            let history = history.filter(|_| attempt != 3);
            let fallback_guard = (attempt == 3).then(|| self.selection_lock.lock());
            let plan = if fallback_guard.is_some() {
                let (healthy, bindings) = self.selection_targets(workers)?;
                let count = healthy.len();
                let input_tokens: Vec<_> = healthy
                    .iter()
                    .map(|&index| {
                        input
                            .tokens
                            .prompts_for(workers[index].as_ref())
                            .input_tokens()
                    })
                    .collect();
                let max_known_input = input_tokens.iter().flatten().copied().max().unwrap_or(0);
                SelectionPlan {
                    healthy,
                    bindings,
                    grouped: vec![Vec::new(); count],
                    input_tokens: input_tokens
                        .into_iter()
                        .map(|tokens| tokens.unwrap_or(max_known_input))
                        .collect(),
                    prefill_tokens: vec![None; count],
                    history_credit: Vec::new(),
                    has_tokens: max_known_input > 0,
                }
            } else {
                self.prepare_selection(workers, input, lora_name, cache_salt, history.is_some())?
            };
            // Token comparisons happen outside the reservation lock.
            let history_inputs = history
                .map(|request| self.history_candidates(workers, input, &plan, request))
                .unwrap_or_default();
            let excluded = |position: usize| {
                excluded_workers
                    .iter()
                    .any(|url| workers[plan.healthy[position]].url() == url.as_str())
            };

            let guard = fallback_guard.unwrap_or_else(|| self.selection_lock.lock());
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
            let mut used_history = false;
            let mut history_reason = "missing_evidence";

            // Recheck the table inside the critical section. Affinity selects
            // a worker; cache credit still comes from that worker's evidence.
            let acquired = digest
                .as_ref()
                .and_then(|digest| self.affinity.acquire(digest));
            let used_binding = acquired
                .as_ref()
                .and_then(|record| self.affinity_position(&plan, record))
                .is_some_and(|position| !excluded(position));
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
                    let mut position = self.pick_position(
                        plan.has_tokens,
                        &costs,
                        &components,
                        eligible.as_deref(),
                    )?;
                    if history_inputs.iter().any(Option::is_some) {
                        let feasible: Vec<usize> =
                            eligible.unwrap_or_else(|| (0..plan.healthy.len()).collect());
                        let base = position;
                        let minimum = feasible
                            .iter()
                            .map(|&index| costs[index])
                            .reduce(f64::min)?;
                        let now = Instant::now();
                        // The source can retire without replacing the policy
                        // snapshot. Its empty result is then unknown, not zero.
                        (position, history_reason) = history::prefer(
                            base,
                            feasible.iter().map(|&i| {
                                plan.history_credit
                                    .get(i)
                                    .copied()
                                    .flatten()
                                    .filter(|_| plan.bindings[i].history_registration_current())
                            }),
                            feasible
                                .iter()
                                .copied()
                                .filter(|&index| {
                                    Self::history_tied(
                                        base,
                                        index,
                                        minimum,
                                        &costs,
                                        &components,
                                        &candidates,
                                    )
                                })
                                .map(|index| {
                                    let length = history_inputs[index]
                                        .as_ref()
                                        .and_then(|(_, _, hit)| hit.as_ref())
                                        .map_or(0, |hit| hit.len_at(now));
                                    (index, length)
                                }),
                        );
                        used_history = position != base;
                    }
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
            let outcome = if used_binding {
                "sticky_hit"
            } else if attempt == 3 {
                "registration_churn"
            } else if used_history {
                "history_hint"
            } else if !plan.has_tokens {
                "missing_tokens"
            } else if let Some(reason) = Self::credit_fallback(cache_salt, true) {
                reason
            } else if plan.bindings[position].cache_key.is_none() {
                "unverified_identity"
            } else {
                "prefix_cost"
            };
            crate::metrics::RouterMetrics::record_prefix_routing_decision(
                "kv_aware", "regular", outcome,
            );
            if history.is_some() {
                crate::metrics::RouterMetrics::record_history_decision(
                    "regular_history",
                    if used_binding {
                        "session_binding"
                    } else {
                        history_reason
                    },
                );
            }
            tracing::debug!(
                outcome,
                worker = workers[worker_index].url(),
                input_tokens = plan.input_tokens[position],
                new_prefill_tokens,
                credited_prefix_tokens =
                    (plan.input_tokens[position] as u64).saturating_sub(new_prefill_tokens),
                used_binding,
                "KV prefix decision"
            );
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
            let observation_input = history_inputs
                .get(position)
                .and_then(Option::as_ref)
                .filter(|_| plan.bindings[position].history_registration_current())
                .map(|(registration_id, ids, _)| (*registration_id, *ids));
            drop(guard);
            let observation = observation_input
                .map(|(registration_id, ids)| HistoryObservation::new(registration_id, ids));
            return Some((
                KvSelectionReceipt {
                    worker_index,
                    new_prefill_tokens,
                    decision,
                    slot,
                },
                observation,
            ));
        }
        None
    }

    fn history_candidates<'a>(
        &self,
        workers: &[Arc<dyn Worker>],
        input: QualifiedKvInput<'a>,
        plan: &SelectionPlan,
        request: &CompletionHistoryRequest,
    ) -> Vec<Option<HistoryCandidate<'a>>> {
        // Keep positions aligned with the plan. Only a single qualified,
        // backend-rendered prompt can establish history.
        plan.healthy
            .iter()
            .zip(&plan.bindings)
            .map(|(&index, binding)| {
                if !input.allows_credit(index, workers.len()) {
                    return None;
                }
                let worker = workers[index].as_ref();
                let ids = input.tokens.prompts_for(worker).single()?;
                if ids.is_empty() {
                    return None;
                }
                let registration_id = request.registration_id(worker, binding)?;
                let hit = self.completion_history.get().and_then(|store| {
                    store.match_prefix(registration_id, ids, binding.block_size, Instant::now())
                });
                Some((registration_id, ids, hit))
            })
            .collect()
    }

    fn history_tied(
        base: usize,
        index: usize,
        minimum: f64,
        costs: &[f64],
        components: &[CostComponents],
        candidates: &[CandidateView<'_>],
    ) -> bool {
        costs[base] == minimum
            && costs[index] == costs[base]
            && components[index] == components[base]
            && candidates[index].load.active_prefill_tokens
                <= candidates[base].load.active_prefill_tokens
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

    // Caller holds selection_lock so registration cannot invalidate the snapshot.
    fn selection_targets(
        &self,
        workers: &[Arc<dyn Worker>],
    ) -> Option<(Vec<usize>, Vec<WorkerBinding>)> {
        let mut healthy = get_healthy_worker_indices(workers);
        let mut bindings = Vec::with_capacity(healthy.len());
        // Once lifecycle registration is in use, removed worker handles
        // must not reappear as never-registered load-only fallbacks.
        if self.next_generation.load(Ordering::Acquire) > 1 {
            healthy.retain(|&index| {
                let Some(binding) = self.targets.get(workers[index].url()) else {
                    return false;
                };
                bindings.push(WorkerBinding::Registered(binding.value().clone()));
                true
            });
        } else {
            bindings.extend(
                healthy
                    .iter()
                    .map(|&index| self.binding_for_worker(workers[index].as_ref())),
            );
        }
        (!healthy.is_empty()).then_some((healthy, bindings))
    }

    /// Event identities do not include cache salts; unverified models receive no credit.
    pub(crate) fn credit_fallback(
        cache_salt: Option<&str>,
        model_verified: bool,
    ) -> Option<&'static str> {
        if cache_salt.is_some_and(|salt| !salt.is_empty()) {
            Some("cache_salt")
        } else if !model_verified {
            Some("unverified_model")
        } else {
            None
        }
    }

    /// Every candidate must be eligible for a shared Decode credit estimate.
    pub(crate) fn all_candidates_priceable(
        request_cacheable: bool,
        cache_salt: Option<&str>,
        model_verified: impl IntoIterator<Item = bool>,
    ) -> bool {
        let mut models = model_verified.into_iter().peekable();
        request_cacheable
            && models.peek().is_some()
            && models.all(|verified| Self::credit_fallback(cache_salt, verified).is_none())
    }

    /// Healthy workers, their target bindings, and per-candidate KV matches.
    /// All provider queries happen here, outside every lock.
    fn prepare_selection(
        &self,
        workers: &[Arc<dyn Worker>],
        input: QualifiedKvInput<'_>,
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
        collect_history_credit: bool,
    ) -> Option<SelectionPlan> {
        let (healthy, bindings) = {
            let _guard = self.selection_lock.lock();
            self.selection_targets(workers)?
        };

        // KV events carry lora_name but not cache_salt. Hash LoRA requests the
        // same way as ingestion; for salted requests, skip placement credit
        // instead of claiming an incompatible unsalted cache entry.
        let credit_fallback = Self::credit_fallback(cache_salt, true);
        let prompts: Vec<_> = healthy
            .iter()
            .map(|&index| input.tokens.prompts_for(workers[index].as_ref()))
            .collect();
        // Keep verified matches when another worker cannot render. Unknown
        // candidates receive no cache credit and the largest known input size.
        let known_input_tokens = prompts
            .iter()
            .filter_map(WorkerPrompts::input_tokens)
            .max()
            .unwrap_or(0);
        let has_tokens = known_input_tokens > 0;
        if !has_tokens {
            crate::metrics::RouterMetrics::record_kv_placement_lookup("missing_tokens");
        }
        let mut lookups = crate::metrics::PlacementLookups::default();
        let mut query_cache = PromptHashes::default();
        query_cache.hashes.stop_at_gap = self.prefix_scoring;
        query_cache.hashes.single_target = healthy
            .iter()
            .zip(&prompts)
            .filter(|(index, ids)| !ids.is_empty() && input.allows_credit(**index, workers.len()))
            .nth(1)
            .is_none();
        // A negative page observation covers every eligible rank with the same
        // domain and hash contract at this cutoff. Hits still require rank trust.
        let shared_miss = if self.prefix_scoring
            && has_tokens
            && !query_cache.hashes.single_target
            && lora_name.is_none()
            && credit_fallback.is_none()
        {
            prompts
                .first()
                .and_then(WorkerPrompts::single)
                .filter(|&tokens| {
                    let Some(shared) = bindings
                        .first()
                        .and_then(|binding| binding.shared_prefix.as_ref())
                    else {
                        return false;
                    };
                    shared.confirms_miss(tokens, &mut query_cache.hashes)
                        && bindings.iter().zip(&prompts).all(|(binding, prompts)| {
                            binding
                                .shared_prefix
                                .as_ref()
                                .is_some_and(|other| shared.same_contract(other))
                                && prompts
                                    .single()
                                    .is_some_and(|ids| std::ptr::eq(tokens, ids) || tokens == ids)
                        })
                })
        } else {
            None
        };
        if let Some(tokens) = shared_miss {
            let size = bindings.len();
            let count = input.eligibility.map_or(size, |_| {
                healthy
                    .iter()
                    .filter(|&&i| input.allows_credit(i, workers.len()))
                    .count()
            });
            lookups.record_count("miss", count as u64);
            // Every binding here has a verified contract and the same
            // nonempty single prompt, but ineligible positions are unknown.
            let history_credit = if collect_history_credit {
                healthy
                    .iter()
                    .map(|&index| input.allows_credit(index, workers.len()).then_some(false))
                    .collect()
            } else {
                Vec::new()
            };
            return Some(SelectionPlan {
                healthy,
                bindings,
                grouped: vec![Vec::new(); size],
                input_tokens: vec![tokens.len(); size],
                prefill_tokens: vec![None; size],
                history_credit,
                has_tokens,
            });
        }
        let mut unique = query_cache.hashes.single_target.then_some(true);
        let mut grouped = Vec::with_capacity(bindings.len());
        let mut input_tokens = Vec::with_capacity(bindings.len());
        let mut prefill_tokens = Vec::with_capacity(bindings.len());
        let mut history_credit = Vec::new();
        for ((binding, prompts), &worker_index) in bindings.iter().zip(prompts).zip(&healthy) {
            let allows_credit = input.allows_credit(worker_index, workers.len());
            let mut evidence_known = collect_history_credit
                && has_tokens
                && binding.cache_key.is_some()
                && !prompts.is_empty()
                && allows_credit
                && credit_fallback.is_none();
            let mut total_input = 0usize;
            let mut total_prefill = 0u64;
            let mut credited = false;
            let mut single_matches = Vec::new();
            let single = prompts.len() == 1;
            if has_tokens {
                for ids in prompts.iter() {
                    evidence_known &= !ids.is_empty();
                    total_input = total_input.saturating_add(ids.len());
                    if !allows_credit {
                        total_prefill = total_prefill.saturating_add(ids.len() as u64);
                        continue;
                    }
                    let matches = Self::cached_matches(
                        &mut query_cache,
                        &mut lookups,
                        binding,
                        ids,
                        lora_name,
                        credit_fallback,
                        || {
                            single
                                && *unique.get_or_insert_with(|| {
                                    all_unique(bindings.iter().map(|b| b.generation))
                                })
                        },
                    );
                    let params = CostParams {
                        block_size: binding.block_size,
                        ..self.params
                    };
                    let candidate = CandidateView {
                        target: &binding.target,
                        matches: &matches,
                        load: LoadSample::default(),
                        input_tokens: ids.len(),
                        prefill_tokens: None,
                    };
                    let credit = evidence_known.then(|| credited_tokens(&candidate, &params));
                    credited |= credit.is_some_and(|credit| credit > 0.0);
                    if single {
                        single_matches = matches.into_owned();
                        continue;
                    }
                    total_prefill = total_prefill.saturating_add(credit.map_or_else(
                        || request_prefill_tokens(&candidate, &params),
                        |credit| remaining_work(ids.len(), credit),
                    ));
                }
            }
            if prompts.is_empty() {
                total_input = known_input_tokens;
            }
            grouped.push(single_matches);
            if collect_history_credit {
                history_credit.push(evidence_known.then_some(credited));
            }
            input_tokens.push(total_input);
            prefill_tokens.push((has_tokens && prompts.len() > 1).then_some(total_prefill));
        }
        Some(SelectionPlan {
            healthy,
            bindings,
            grouped,
            input_tokens,
            prefill_tokens,
            history_credit,
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
            let rank = |component: &CostComponents| {
                (
                    if has_tokens { component.ptoken } else { 0 },
                    component.batchsize,
                )
            };
            let best = components
                .iter()
                .enumerate()
                .filter(|(index, _)| costs[*index] == 0.0)
                .min_by_key(|(_, component)| rank(component))
                .map(|(_, component)| rank(component))?;
            let tied: Vec<usize> = components
                .iter()
                .enumerate()
                .filter(|(index, component)| costs[*index] == 0.0 && rank(component) == best)
                .map(|(index, _)| index)
                .collect();
            pick_tied(&tied)
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

    fn tracks_request_load(&self) -> bool {
        true
    }

    /// Occupancy, KV events and affinity are all read live.
    fn can_redecide_at_dispatch(&self, _decode_cache_ready: bool) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "kv_aware"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl KvAwareOps for KvAwarePolicy {
    fn register_history_worker(
        &self,
        worker: &Arc<dyn Worker>,
        info: Option<&WorkerKvInfo>,
        matches: Option<Arc<dyn TieredMatchProvider>>,
        current: Option<Arc<dyn RegistrationCurrent>>,
    ) {
        self.register_worker_binding(
            worker.as_ref(),
            info,
            matches,
            info.is_some_and(|i| i.ranks.as_slice() == [0])
                .then(|| Arc::downgrade(worker)),
            current,
        );
    }

    fn capture_completion_history(
        &self,
        workers: &[Arc<dyn Worker>],
        requested_model: &str,
    ) -> Option<CompletionHistoryRequest> {
        Some(self.capture_history_registrations(workers, requested_model))
    }

    fn observe_completion_milestone(&self, worker: &dyn Worker, observation: HistoryObservation) {
        self.commit_history_observation(worker, observation);
    }

    fn select_with_history(
        &self,
        workers: &[Arc<dyn Worker>],
        tokens: KvRequestTokens<'_>,
        eligibility: Option<&[bool]>,
        history: Option<&CompletionHistoryRequest>,
        identity: SelectionIdentity<'_>,
        excluded_workers: &[String],
    ) -> Option<(KvSelectionReceipt, Option<HistoryObservation>)> {
        self.select_with_context(
            workers,
            QualifiedKvInput {
                tokens,
                eligibility,
            },
            identity,
            excluded_workers,
            history,
        )
    }

    fn has_affinity(
        &self,
        workers: &[Arc<dyn Worker>],
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
        affinity_key: &AffinityKey,
    ) -> bool {
        let _guard = self.selection_lock.lock();
        let records = self.affinity.records.lock();
        let record = records.peek(&affinity_key.scoped_digest(lora_name, cache_salt));
        let Some(record) = record else {
            return false;
        };
        !record.is_expired(Instant::now())
            && workers.iter().any(|worker| {
                worker.is_available()
                    && self.targets.get(worker.url()).is_some_and(|binding| {
                        binding.generation == record.generation && binding.target == record.target
                    })
            })
    }

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

    fn estimate_prefill_work<'tokens>(
        &self,
        inputs: &[(&dyn Worker, &'tokens [Vec<u32>], bool)],
        hashes: &mut PromptHashes<'tokens>,
    ) -> Vec<PrefillWorkEstimate> {
        hashes.set_stage_targets(inputs, false);
        let mut lookups = crate::metrics::PlacementLookups::default();
        let mut unique = hashes.hashes.single_target.then_some(true);
        inputs
            .iter()
            .map(|&(worker, prompts, credit)| {
                let binding = self.binding_for_worker(worker);
                let verified = credit && binding.cache_key.is_some();
                let params = CostParams {
                    block_size: binding.block_size,
                    ..self.params
                };
                let mut new_prefill_tokens = 0u64;
                for tokens in prompts {
                    let matches = if verified {
                        Self::cached_matches(
                            hashes,
                            &mut lookups,
                            &binding,
                            tokens,
                            None,
                            None,
                            || {
                                prompts.len() == 1
                                    && *unique.get_or_insert_with(|| {
                                        all_unique(inputs.iter().map(|(w, _, _)| w.url()))
                                    })
                            },
                        )
                    } else {
                        Cow::Borrowed(&[][..])
                    };
                    if matches.is_empty() {
                        new_prefill_tokens = new_prefill_tokens.saturating_add(tokens.len() as u64);
                        continue;
                    }
                    new_prefill_tokens = new_prefill_tokens.saturating_add(request_prefill_tokens(
                        &CandidateView {
                            target: &binding.target,
                            matches: &matches,
                            load: LoadSample::default(),
                            input_tokens: tokens.len(),
                            prefill_tokens: None,
                        },
                        &params,
                    ));
                }
                PrefillWorkEstimate {
                    new_prefill_tokens,
                    placement_verified: verified,
                }
            })
            .collect()
    }

    /// Marginal Decode context: the prompt, once per request (sequences share one copy of
    /// its blocks), less
    /// the device blocks the scheduler can retain on this candidate.
    fn estimate_decode_work<'tokens>(
        &self,
        inputs: &[(&dyn Worker, &'tokens [Vec<u32>], bool)],
        hashes: &mut PromptHashes<'tokens>,
    ) -> Vec<DecodeWorkEstimate> {
        self.estimate_decode_work_with_transfer(inputs, hashes, false)
    }

    fn estimate_decode_work_with_transfer<'tokens>(
        &self,
        inputs: &[(&dyn Worker, &'tokens [Vec<u32>], bool)],
        hashes: &mut PromptHashes<'tokens>,
        nixl_pull: bool,
    ) -> Vec<DecodeWorkEstimate> {
        hashes.set_stage_targets(inputs, nixl_pull);
        let mut lookups = crate::metrics::PlacementLookups::default();
        let mut unique = hashes.hashes.single_target.then_some(true);
        inputs
            .iter()
            .map(|&(worker, prompts, credit)| {
                let binding = self.binding_for_worker(worker);
                if !credit || binding.cache_key.is_none() {
                    return DecodeWorkEstimate::default();
                }
                // Decode occupancy shrinks only by blocks already on the device. A prefix
                // that lives in host memory or on disk still needs a GPU block when it
                // arrives, so the tiered credits must not discount it: that would understate
                // the new occupancy and can pick a busier worker. Prefill keeps the tiered
                // credits, because there they avoid recompute.
                let params = CostParams {
                    block_size: binding.block_size,
                    transfer_credits: TransferCredits {
                        host_pinned: 0.0,
                        external: 0.0,
                        disk: 0.0,
                        ..self.params.transfer_credits
                    },
                    ..self.params
                };
                let mut additional = 0u64;
                for tokens in prompts {
                    let matches = Self::cached_matches(
                        hashes,
                        &mut lookups,
                        &binding,
                        tokens,
                        None,
                        None,
                        || {
                            prompts.len() == 1
                                && *unique.get_or_insert_with(|| {
                                    all_unique(inputs.iter().map(|(w, _, _)| w.url()))
                                })
                        },
                    );
                    let context = u64::try_from(tokens.len()).unwrap_or(u64::MAX);
                    let reused = credited_tokens(
                        &CandidateView {
                            target: &binding.target,
                            matches: &matches,
                            load: LoadSample::default(),
                            input_tokens: tokens.len(),
                            prefill_tokens: None,
                        },
                        &params,
                    ) as u64;
                    let reused = if nixl_pull {
                        binding.matches.nixl_decode_credit(
                            binding.target.dp_rank,
                            tokens.len(),
                            reused,
                        )
                    } else {
                        reused
                    };
                    // The engine shares one copy of the prompt's blocks across the
                    // request's sequences, so the marginal cost is the prompt's
                    // un-cached remainder once - the same amount the ledger charges when
                    // this request becomes an existing one. Multiplying by `n` here made
                    // a request cost more before admission than after, so the two sides of
                    // the comparison disagreed.
                    additional = additional.saturating_add(context.saturating_sub(reused));
                }
                DecodeWorkEstimate {
                    additional_tokens: Some(additional),
                }
            })
            .collect()
    }

    fn pick_prefill(&self, costs: &[f64], occupancy: &[usize]) -> Option<usize> {
        if costs.len() != occupancy.len() || costs.iter().any(|cost| !cost.is_finite()) {
            return None;
        }
        if self.params.router_temperature <= 0.0 {
            let (best_index, best_cost) = costs
                .iter()
                .enumerate()
                .min_by(|(li, left), (ri, right)| {
                    left.total_cmp(right)
                        .then_with(|| occupancy[*li].cmp(&occupancy[*ri]))
                })
                .map(|(index, cost)| (index, *cost))?;
            // Occupancy is the last ranking signal. Candidates that also match
            // it are equally good, so spread them instead of taking the first.
            let tied: Vec<usize> = (0..costs.len())
                .filter(|&index| {
                    costs[index] == best_cost && occupancy[index] == occupancy[best_index]
                })
                .collect();
            pick_tied(&tied)
        } else {
            self.picker.pick(costs)
        }
    }

    fn pick_decode(&self, loads: &[usize], additional: &[u64]) -> Option<usize> {
        if loads.len() != additional.len() {
            return None;
        }
        let min_load = *loads.iter().min()?;
        let tied: Vec<usize> = loads
            .iter()
            .enumerate()
            .filter(|&(_, load)| *load == min_load)
            .map(|(index, _)| index)
            .collect();
        // A continuation reaches the same projected load on the engine that
        // holds its prefix with a smaller `additional`; rotating that tie sent
        // the request to an engine without the prefix and lost the reuse.
        let min_additional = tied.iter().map(|&index| additional[index]).min()?;
        let tied: Vec<usize> = tied
            .into_iter()
            .filter(|&index| additional[index] == min_additional)
            .collect();
        let cursor = self.decode_tie_cursor.fetch_add(1, Ordering::Relaxed) as usize;
        Some(tied[cursor % tied.len()])
    }

    fn session_affinity_enabled(&self) -> bool {
        self.affinity.ttl.is_some()
    }

    fn select_decode(
        &self,
        workers: &[Arc<dyn Worker>],
        session: Option<&AffinityKey>,
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
        new_context: Option<&HashMap<String, u64>>,
    ) -> Option<StagePick> {
        if workers.is_empty() {
            return None;
        }
        // A partial table would compare candidates in different units: a missing
        // entry read as zero makes the recovered worker look free. Enforce the
        // documented contract here instead of trusting every caller to filter.
        let new_context = new_context.filter(|context| {
            workers
                .iter()
                .all(|worker| context.contains_key(worker.url()))
        });
        // Occupancy is the caller's sample; the marginal context is ours. A
        // candidate that reuses its prefix adds only the remainder, so the two
        // are added in tokens rather than compared as separate rankings.
        let mut loads = Vec::with_capacity(workers.len());
        let mut additional_tokens = Vec::with_capacity(workers.len());
        for worker in workers {
            let occupancy = worker.load();
            let additional = new_context
                .and_then(|context| context.get(worker.url()).copied())
                .unwrap_or(0);
            additional_tokens.push(additional);
            loads.push(occupancy.saturating_add(usize::try_from(additional).unwrap_or(usize::MAX)));
        }
        let Some(session) = session.filter(|_| self.session_affinity_enabled()) else {
            let index = self.pick_decode(&loads, &additional_tokens)?;
            return Some(StagePick {
                index,
                outcome: "no_session",
                binding: None,
            });
        };
        // Bindings validate against live registrations: an unregistered
        // target has no generation, so a stale binding could survive a
        // restart. Register missing candidates up front (P workers also
        // register through the KV refresh loop; D workers only ever get
        // their load-only registration here).
        for worker in workers {
            if !self.targets.contains_key(worker.url()) {
                self.register_target(worker.url(), self.target_for_worker(worker.as_ref()));
            }
        }
        let digest = session.staged_digest("decode", lora_name, cache_salt);
        // Use the same projected request counts as P/D admission. Including
        // the incoming request allows one idle session to retain locality,
        // while simultaneous requests cannot pin an overloaded endpoint.
        // u128 keeps the sum and incoming +1 safe at usize::MAX loads.
        let fair_share = (loads.iter().map(|&load| load as u128).sum::<u128>() + 1)
            .div_ceil(workers.len() as u128);
        let mut rebind = false;
        if let Some(record) = self.affinity.acquire(&digest) {
            if let Some(index) = self.binding_position(&record, workers) {
                if (loads[index] as u128) < fair_share {
                    return Some(StagePick {
                        index,
                        outcome: "sticky_hit",
                        binding: Some(record),
                    });
                }
                rebind = true;
            }
            // Release this lookup's ref; existing attempts still own their
            // original record even when bind replaces the table entry.
            record.release_ref();
        }
        // A minimum-load worker always fits the rounded-up share.
        let index = self.pick_decode(&loads, &additional_tokens)?;
        let binding = self.targets.get(workers[index].url()).and_then(|entry| {
            self.affinity
                .bind(digest, entry.target.clone(), entry.generation)
        });
        Some(StagePick {
            index,
            outcome: if binding.is_some() {
                if rebind {
                    "rebind"
                } else {
                    "bind"
                }
            } else {
                "capacity_bypass"
            },
            binding,
        })
    }

    fn register_worker(
        &self,
        worker: &dyn Worker,
        info: Option<&WorkerKvInfo>,
        matches: Option<Arc<dyn TieredMatchProvider>>,
    ) {
        self.register_worker_binding(worker, info, matches, None, None);
    }

    fn unregister_worker(&self, worker_url: &str) {
        self.unregister_target(worker_url);
    }

    fn select_worker_with_qualified_inputs(
        &self,
        workers: &[Arc<dyn Worker>],
        tokens: KvRequestTokens<'_>,
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
        affinity_key: Option<&AffinityKey>,
        excluded_workers: &[String],
        eligibility: Option<&[bool]>,
    ) -> Option<KvSelectionReceipt> {
        KvAwarePolicy::select_worker_with_qualified_inputs(
            self,
            workers,
            QualifiedKvInput {
                tokens,
                eligibility,
            },
            lora_name,
            cache_salt,
            affinity_key,
            excluded_workers,
        )
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    use crate::request_identity::{resolve_affinity_identity, IdentityFields};
    use http::HeaderMap;
    use serde_json::Value as JsonValue;

    #[test]
    fn duplicate_workers_and_prompts_share_one_metric_per_registration() {
        let a = BasicWorker::new("http://repeat-a".to_string(), WorkerType::Regular);
        let duplicate = BasicWorker::new(a.url().to_owned(), WorkerType::Regular);
        let b = BasicWorker::new("http://repeat-b".to_string(), WorkerType::Regular);
        let ta = target("one", 0);
        let tb = target("two", 0);
        let provider = Arc::new(PlacementTestProvider {
            hits: vec![
                tier_match(&tb, StorageTier::Device, &[(0, 2)]),
                tier_match(&ta, StorageTier::Device, &[(0, 1)]),
            ],
            hybrid: true,
            queries: AtomicUsize::new(0),
        });
        let policy = KvAwarePolicy::default();
        let key = CacheKey {
            model: Arc::from("one"),
            block_size: 16,
        };
        policy.register_target_with_cache_identity(a.url(), ta, key.clone(), provider.clone());
        policy.register_target_with_cache_identity(b.url(), tb, key, provider.clone());
        let prompt = vec![vec![4u32; 49]];
        let prompts = vec![prompt[0].clone(), prompt[0].clone()];
        let prefill_recorder =
            metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let decode_recorder =
            metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let inputs = [
            (&a as &dyn Worker, prompt.as_slice(), true),
            (&b as &dyn Worker, prompts.as_slice(), true),
            (&duplicate as &dyn Worker, prompt.as_slice(), true),
        ];
        let mut h = PromptHashes::new();
        let pre = metrics::with_local_recorder(&prefill_recorder, || {
            policy.estimate_prefill_work(&inputs, &mut h)
        });
        assert_eq!(
            pre.iter().map(|s| s.new_prefill_tokens).collect::<Vec<_>>(),
            vec![33, 34, 33]
        );
        let dec = metrics::with_local_recorder(&decode_recorder, || {
            policy.estimate_decode_work(&inputs, &mut h)
        });
        assert_eq!(
            dec.iter().map(|s| s.additional_tokens).collect::<Vec<_>>(),
            vec![Some(33), Some(34), Some(33)]
        );
        assert_eq!(provider.queries.load(Ordering::Relaxed), 4);
        for recorder in [prefill_recorder, decode_recorder] {
            assert!(recorder
                .handle()
                .render()
                .contains("vllm_router_kv_placement_lookups_total{result=\"hit\"} 2"));
        }
    }
    fn affinity_from_request(
        headers: Option<&HeaderMap>,
        body: Option<&JsonValue>,
        model: &str,
    ) -> Option<AffinityKey> {
        resolve_affinity_identity(
            headers,
            body.map(IdentityFields::from_body).unwrap_or_default(),
        )
        .ok()
        .flatten()
        .map(|identity| AffinityKey::from_identity(identity, model))
    }

    use crate::core::{BasicWorker, DPAwareWorker, WorkerType};
    use crate::kv_index::source::test_support::{
        current_registration, event_batch, full_binding, full_binding_with_adapter, full_store,
        warm_full,
    };
    use crate::kv_index::{KVEvent, SourceBinding};
    use crate::policies::RoundRobinPolicy;

    fn target(name: &str, rank: u32) -> RoutableTarget {
        RoutableTarget {
            instance_id: Arc::from(name),
            dp_rank: rank,
        }
    }

    #[test]
    fn shared_first_page_miss_skips_rank_matching_only_for_one_contract() {
        struct Counted(SourceBinding, AtomicUsize);

        impl TieredMatchProvider for Counted {
            fn target_scoped(&self) -> bool {
                true
            }

            fn shared_prefix_lookup(&self, rank: u32) -> Option<SharedPrefixLookup> {
                self.0.shared_prefix_lookup(rank)
            }

            fn matches_with_cache<'a>(
                &self,
                tokens: &'a [u32],
                block_size: u32,
                lora: Option<&str>,
                rank: u32,
                cache: &mut RequestHashCache<'a>,
            ) -> Vec<TierMatch> {
                self.1.fetch_add(1, Ordering::Relaxed);
                self.0
                    .matches_with_cache(tokens, block_size, lora, rank, cache)
            }
        }

        let policy = KvAwarePolicy::default();
        let domain = Arc::new(crate::kv_index::KvBlockIndexer::new());
        let workers: Vec<Arc<dyn Worker>> = ["http://a:8000", "http://b:8000"]
            .into_iter()
            .map(|url| {
                Arc::new(BasicWorker::new(url.into(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect();
        let register = |worker: &dyn Worker, domain: &Arc<crate::kv_index::KvBlockIndexer>| {
            let target = target(worker.url(), 0);
            let source = full_binding(target.clone(), 16).with_shared_index(domain.clone());
            source.begin_empty(0);
            let provider = Arc::new(Counted(source, AtomicUsize::new(0)));
            policy.register_target_with_cache_identity(
                worker.url(),
                target,
                CacheKey {
                    model: "model".into(),
                    block_size: 16,
                },
                provider.clone(),
            );
            provider
        };
        let first = register(workers[0].as_ref(), &domain);
        let second = register(workers[1].as_ref(), &domain);
        let tokens = [42; 33];
        let inputs: WorkerTokenInputs = workers
            .iter()
            .map(|worker| (worker.base_url().into(), Arc::from([tokens.to_vec()])))
            .collect();
        let plan = |inputs: &WorkerTokenInputs, collect_history_credit| {
            policy
                .prepare_selection(
                    &workers,
                    QualifiedKvInput {
                        tokens: KvRequestTokens::PerWorker(inputs),
                        eligibility: None,
                    },
                    None,
                    None,
                    collect_history_credit,
                )
                .unwrap()
        };
        let ordinary = plan(&inputs, false);
        assert!(ordinary.grouped.iter().all(Vec::is_empty));
        assert!(ordinary.history_credit.is_empty());
        let with_history = plan(&inputs, true);
        assert!(with_history.grouped.iter().all(Vec::is_empty));
        assert_eq!(with_history.history_credit, [Some(false), Some(false)]);
        assert_eq!(first.1.load(Ordering::Relaxed), 0);
        assert_eq!(second.1.load(Ordering::Relaxed), 0);

        warm_full(&second.0, 0, &tokens[..32]);
        let warm = plan(&inputs, true);
        assert!(!warm.grouped[1].is_empty());
        assert_eq!(warm.history_credit, [Some(false), Some(true)]);

        let mut different = inputs.clone();
        different.insert(
            workers[0].base_url().into(),
            Arc::from([vec![7; tokens.len()]]),
        );
        assert!(!plan(&different, false).grouped[1].is_empty());

        // A separate index can have the same policy key and still hold
        // different pages. A miss on the first index must not mask this hit.
        let other_domain = Arc::new(crate::kv_index::KvBlockIndexer::new());
        let unlisted =
            full_binding(target("unlisted", 0), 16).with_shared_index(other_domain.clone());
        unlisted.begin_empty(0);
        register(workers[0].as_ref(), &other_domain);
        let plan = plan(&inputs, false);
        assert!(plan.grouped[0].is_empty());
        assert!(!plan.grouped[1].is_empty());
    }

    #[test]
    fn decode_observes_stores_and_clears_after_prefill() {
        let prefill = KvAwarePolicy::default();
        let decode = KvAwarePolicy::default();
        let worker = BasicWorker::new("http://worker".to_owned(), WorkerType::Regular);
        let target = target(worker.url(), 0);
        let source = Arc::new(full_binding(target.clone(), 16));
        source.begin_empty(0);
        for policy in [&prefill, &decode] {
            policy.register_target_with_cache_identity(
                worker.url(),
                target.clone(),
                CacheKey {
                    model: "model".into(),
                    block_size: 16,
                },
                source.clone(),
            );
        }
        let prompts = [vec![42; 65]];
        let inputs = [(&worker as &dyn Worker, prompts.as_slice(), true)];
        for (seq, before, after) in [(0, 65, 1), (1, 1, 65)] {
            let mut cache = PromptHashes::new();
            assert_eq!(
                prefill.estimate_prefill_work(&inputs, &mut cache)[0].new_prefill_tokens,
                before
            );
            if seq == 0 {
                warm_full(&source, seq, &prompts[0][..64]);
            } else {
                let clear = KVEvent::AllBlocksCleared(crate::kv_index::AllBlocksCleared {});
                source.apply_batch(seq, &event_batch(vec![clear])).unwrap();
            }
            assert_eq!(
                decode.estimate_decode_work(&inputs, &mut cache)[0].additional_tokens,
                Some(after)
            );
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
        }
    }

    fn register_full(
        policy: &KvAwarePolicy,
        url: &str,
        target: RoutableTarget,
        model: &str,
        block_size: u32,
    ) -> Arc<SourceBinding> {
        let binding = Arc::new(full_binding(target.clone(), block_size));
        binding.begin_empty(0);
        policy.register_target_with_cache_identity(
            url,
            target,
            CacheKey {
                model: Arc::from(model),
                block_size,
            },
            binding.clone(),
        );
        binding
    }

    #[test]
    fn hybrid_token_matches_are_scored_per_target_for_regular_and_prefill() {
        use crate::kv_index::source::binding::tests::qwen_index;

        let workers: Vec<Arc<dyn Worker>> = (0..2)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://worker-{i}"),
                    WorkerType::Regular,
                )) as Arc<dyn Worker>
            })
            .collect();
        let policy = KvAwarePolicy::default();
        let mut indexes = Vec::new();
        for worker in &workers {
            let target = target(worker.url(), 0);
            let index = Arc::new(qwen_index(target.clone(), 8));
            index.begin_empty(0);
            policy.register_target_with_cache_identity(
                worker.url(),
                target,
                CacheKey {
                    model: Arc::from("qwen"),
                    block_size: 528,
                },
                index.clone(),
            );
            indexes.push(index);
        }
        let payload = crate::kv_index::source::binding::tests::qwen35_payload();
        indexes[1].apply_payload(0, &payload).unwrap();
        let prompt: Vec<u32> = (0..1024).map(|i| 100 + i % 100).collect();
        assert_eq!(
            policy.select_worker_with_token_ids(&workers, Some(&prompt)),
            Some(1)
        );

        let prompts = [prompt.clone()];
        let estimates = policy.estimate_prefill_work(
            &[
                (workers[0].as_ref(), &prompts, true),
                (workers[1].as_ref(), &prompts, true),
            ],
            &mut PromptHashes::new(),
        );
        assert_eq!(estimates[0].new_prefill_tokens, 1024);
        assert_eq!(estimates[1].new_prefill_tokens, 496);
        assert!(indexes[1]
            .matches(&prompt, 528, Some("adapter"), 0)
            .is_empty());
        let _stream = crate::kv_index::EventStream::new(indexes[1].clone());
        assert_eq!(
            policy.estimate_prefill_work(
                &[(workers[1].as_ref(), &prompts, true)],
                &mut PromptHashes::new(),
            )[0]
            .new_prefill_tokens,
            1024
        );
    }

    struct PlacementTestProvider {
        hits: Vec<TierMatch>,
        hybrid: bool,
        queries: AtomicUsize,
    }
    impl TieredMatchProvider for PlacementTestProvider {
        fn matches(&self, _: &[u32], _: u32, _: Option<&str>, _: u32) -> Vec<TierMatch> {
            self.queries.fetch_add(1, Ordering::Relaxed);
            self.hits.clone()
        }
        fn target_scoped(&self) -> bool {
            self.hybrid
        }
    }

    #[test]
    fn request_qualification_applies_to_dense_and_hybrid_placement() {
        for hybrid in [false, true] {
            let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
            let metrics_handle = recorder.handle();
            let _metrics = metrics::set_default_local_recorder(&recorder);
            let policy = KvAwarePolicy::default();
            let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(BasicWorker::new(
                "http://qualified".into(),
                WorkerType::Regular,
            ))];
            let provider = Arc::new(PlacementTestProvider {
                hits: vec![tier_match(
                    &target(workers[0].url(), 0),
                    StorageTier::Device,
                    &[(0, 4)],
                )],
                hybrid,
                queries: AtomicUsize::new(0),
            });
            policy.register_target_with_cache_identity(
                workers[0].url(),
                target(workers[0].url(), 0),
                CacheKey {
                    model: Arc::from("model"),
                    block_size: 16,
                },
                provider.clone(),
            );
            let ids: Vec<u32> = (0..65).collect();
            for (allowed, prefill) in [(true, 1), (false, 65)] {
                let receipt = policy
                    .select_worker_with_qualified_inputs(
                        &workers,
                        QualifiedKvInput {
                            tokens: KvRequestTokens::Shared(Some(&ids)),
                            eligibility: Some(&[allowed]),
                        },
                        None,
                        None,
                        None,
                        &[],
                    )
                    .unwrap();
                assert_eq!(receipt.new_prefill_tokens, prefill, "hybrid={hybrid}");
                release_receipt(&workers, &receipt);
                let pd = policy.estimate_prefill_work(
                    &[(workers[0].as_ref(), std::slice::from_ref(&ids), allowed)],
                    &mut PromptHashes::new(),
                );
                assert_eq!(pd[0].new_prefill_tokens, prefill, "hybrid={hybrid}");
            }
            assert_eq!(
                provider.queries.load(Ordering::Relaxed),
                2,
                "hybrid={hybrid}"
            );
            assert!(
                metrics_handle
                    .render()
                    .contains(r#"vllm_router_kv_placement_lookups_total{result="hit"} 2"#),
                "both regular and prefill lookups must be recorded; hybrid={hybrid}"
            );
        }
    }

    #[test]
    fn placement_metrics_count_actual_misses_and_unavailable_lookups() {
        for (hybrid, keyed, label, per_lookup) in [
            (true, true, "miss", 2),
            (false, true, "miss", 1),
            (true, false, "index_unavailable", 2),
            (false, false, "index_unavailable", 1),
        ] {
            let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
            let handle = recorder.handle();
            let _metrics = metrics::set_default_local_recorder(&recorder);
            let provider = Arc::new(PlacementTestProvider {
                hits: Vec::new(),
                hybrid,
                queries: AtomicUsize::new(0),
            });
            let policy = KvAwarePolicy {
                matches: provider.clone(),
                ..KvAwarePolicy::default()
            };
            let workers: Vec<Arc<dyn Worker>> = (0..2)
                .map(|n| {
                    Arc::new(BasicWorker::new(
                        format!("http://worker-{n}"),
                        WorkerType::Regular,
                    )) as Arc<dyn Worker>
                })
                .collect();
            for worker in &workers {
                if keyed {
                    policy.register_target_with_cache_identity(
                        worker.url(),
                        target(worker.url(), 0),
                        CacheKey {
                            model: Arc::from("model"),
                            block_size: 16,
                        },
                        provider.clone(),
                    );
                } else {
                    policy.register_target(worker.url(), target(worker.url(), 0));
                }
            }
            let ids: Vec<u32> = (0..65).collect();
            let receipt = policy
                .select_worker_with_qualified_inputs(
                    &workers,
                    QualifiedKvInput {
                        tokens: KvRequestTokens::Shared(Some(&ids)),
                        eligibility: None,
                    },
                    None,
                    None,
                    None,
                    &[],
                )
                .unwrap();
            release_receipt(&workers, &receipt);
            assert_eq!(provider.queries.load(Ordering::Relaxed), per_lookup);
            assert!(handle.render().contains(&format!(
                "vllm_router_kv_placement_lookups_total{{result=\"{label}\"}} {per_lookup}"
            )));
            let prompts = vec![ids];
            policy.estimate_prefill_work(
                &[
                    (workers[0].as_ref(), &prompts, true),
                    (workers[1].as_ref(), &prompts, true),
                ],
                &mut PromptHashes::new(),
            );
            let expected = if keyed { 2 * per_lookup } else { per_lookup };
            assert_eq!(provider.queries.load(Ordering::Relaxed), expected);
            assert!(handle.render().contains(&format!(
                "vllm_router_kv_placement_lookups_total{{result=\"{label}\"}} {expected}"
            )));
        }
    }

    #[test]
    fn placement_eligibility_keeps_unknown_targets_selectable_without_shared_identity_credit() {
        struct Matches {
            queries: AtomicUsize,
            block_size: AtomicUsize,
            hits: Vec<TierMatch>,
        }
        impl TieredMatchProvider for Matches {
            fn matches(
                &self,
                _: &[u32],
                block_size: u32,
                _: Option<&str>,
                _: u32,
            ) -> Vec<TierMatch> {
                self.queries.fetch_add(1, Ordering::Relaxed);
                self.block_size
                    .store(block_size as usize, Ordering::Relaxed);
                self.hits.clone()
            }
        }
        for (case, count, block_size) in [
            ("shared_identity_filters_qualified_targets", 3, 2),
            ("single_worker_larger_block_retry_boundary", 1, 4),
        ] {
            let policy = KvAwarePolicy::new(CostParams {
                router_temperature: 0.0,
                ..CostParams::default()
            });
            let workers: Vec<Arc<dyn Worker>> = (0..count)
                .map(|index| {
                    Arc::new(BasicWorker::new(
                        format!("http://worker-{index}"),
                        WorkerType::Regular,
                    )) as Arc<dyn Worker>
                })
                .collect();
            let provider = Arc::new(Matches {
                queries: AtomicUsize::new(0),
                block_size: AtomicUsize::new(0),
                hits: workers
                    .iter()
                    .map(|worker| {
                        tier_match(&target(worker.url(), 0), StorageTier::Device, &[(0, 2)])
                    })
                    .collect(),
            });
            let key = CacheKey {
                model: Arc::from("engine-cache-identity"),
                block_size,
            };
            for worker in &workers {
                policy.register_target_with_cache_identity(
                    worker.url(),
                    target(worker.url(), 0),
                    key.clone(),
                    provider.clone(),
                );
            }
            let ids: Vec<u32> = (0..2 * block_size + 1).collect();
            let mut mask = vec![true; count];
            if count > 1 {
                mask[count - 1] = false;
            }
            let select = |mask: &[bool], excluded: &[String]| {
                kv_aware_ops(&policy)
                    .unwrap()
                    .select_worker_with_qualified_inputs(
                        &workers,
                        KvRequestTokens::Shared(Some(&ids)),
                        None,
                        None,
                        None,
                        excluded,
                        Some(mask),
                    )
                    .unwrap_or_else(|| panic!("{case}: no worker selected for {mask:?}"))
            };
            let receipt = select(&mask, &[]);
            assert!(mask[receipt.worker_index], "{case}: ineligible worker");
            assert_eq!(receipt.new_prefill_tokens, 1, "{case}");
            assert_eq!(provider.queries.load(Ordering::Relaxed), 1, "{case}");
            assert_eq!(
                provider.block_size.load(Ordering::Relaxed),
                block_size as usize,
                "{case}"
            );
            release_receipt(&workers, &receipt);

            if count == 1 {
                let retry = select(&mask, &[workers[0].url().to_owned()]);
                assert_eq!(retry.worker_index, 0, "{case}: exhausted exclusions");
                assert_eq!(retry.new_prefill_tokens, 1, "{case}: retained credit");
                assert_eq!(workers[0].load(), 1, "{case}: retry reservation");
                release_receipt(&workers, &retry);
                assert_eq!(workers[0].load(), 0, "{case}: retry release");
            }

            if count > 1 {
                let mask = vec![false; count];
                let key = affinity_key("qualified-input");
                let select = |excluded: &[String]| {
                    policy
                        .select_worker_with_qualified_inputs(
                            &workers,
                            QualifiedKvInput {
                                tokens: KvRequestTokens::Shared(Some(&ids)),
                                eligibility: Some(&mask),
                            },
                            None,
                            None,
                            Some(&key),
                            excluded,
                        )
                        .unwrap()
                };
                let bound = select(&[]);
                let bound_index = bound.worker_index;
                assert_eq!(
                    bound.new_prefill_tokens,
                    ids.len() as u64,
                    "{case}: affinity"
                );
                assert_eq!(workers[bound_index].load(), 1, "{case}: affinity");
                release_receipt(&workers, &bound);
                let excluded = vec![workers[bound_index].url().to_owned()];
                let retry = select(&excluded);
                assert_ne!(retry.worker_index, bound_index, "{case}: retry exclusion");
                assert_eq!(retry.new_prefill_tokens, ids.len() as u64, "{case}: retry");
                release_receipt(&workers, &retry);
                let retained = select(&[]);
                assert_eq!(
                    retained.worker_index, bound_index,
                    "{case}: retained affinity"
                );
                release_receipt(&workers, &retained);
                assert!(
                    workers.iter().all(|worker| worker.load() == 0),
                    "{case}: release"
                );
            }

            // A provider query for one qualified target returns hits for
            // every target in this identity. None may leak to an unproved
            // candidate, even though it remains in the common cost model.
            for worker in &workers[..count.saturating_sub(1)] {
                for _ in 0..ids.len() {
                    worker.increment_load();
                }
            }
            if count > 1 {
                let receipt = select(&mask, &[]);
                assert_eq!(receipt.worker_index, count - 1, "{case}: unproved target");
                assert_eq!(
                    receipt.new_prefill_tokens,
                    ids.len() as u64,
                    "{case}: unproved target"
                );
                assert_eq!(
                    receipt.decision.cost,
                    ids.len() as f64,
                    "{case}: unproved target"
                );
                release_receipt(&workers, &receipt);
            }

            for (mask_case, mask) in [
                ("all_ineligible", vec![false; count]),
                ("wrong_length", vec![true; count + 1]),
                ("empty", vec![]),
            ] {
                provider.queries.store(0, Ordering::Relaxed);
                let receipt = select(&mask, &[]);
                assert_eq!(
                    provider.queries.load(Ordering::Relaxed),
                    0,
                    "{case}: {mask_case}"
                );
                assert_eq!(
                    receipt.new_prefill_tokens,
                    ids.len() as u64,
                    "{case}: {mask_case}"
                );
                release_receipt(&workers, &receipt);
            }

            if count > 1 {
                workers[0].set_healthy(false);
                let mut mask = vec![false; count];
                mask[count - 1] = true;
                let receipt = select(&mask, &[]);
                assert_eq!(
                    receipt.worker_index,
                    count - 1,
                    "{case}: healthy candidate positions"
                );
                assert_eq!(
                    receipt.new_prefill_tokens, 1,
                    "{case}: healthy candidate positions"
                );
                release_receipt(&workers, &receipt);
            }
        }
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
            tier_match(&t, StorageTier::Disk, &[(5, 10)]),
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
        let policy = KvAwarePolicy::default();
        let workers: Vec<Arc<dyn Worker>> = ["http://busy", "http://idle"]
            .iter()
            .map(|url| {
                Arc::new(BasicWorker::new((*url).into(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect();
        let ids: Vec<u32> = (0..32).collect();
        for worker in &workers {
            let target = policy.target_for_worker(worker.as_ref());
            let binding = register_full(&policy, worker.url(), target, "model", 16);
            warm_full(&binding, 0, &ids);
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
        let policy = KvAwarePolicy::default();
        let warm = register_full(&policy, "http://warm", target("http://warm", 0), "model", 2);
        policy.register_target("http://cold", target("http://cold", 0));
        let ids = [1_u32, 2, 3, 4, 5, 6];
        warm_full(&warm, 0, &ids[..4]);
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
    fn argmin_picker_spreads_exact_ties_and_keeps_the_ranking() {
        let picker = TemperaturePicker::new(0.0);
        let mut seen = [false; 3];
        for _ in 0..64 {
            let index = picker.pick(&[5.0, 2.0, 2.0]).unwrap();
            assert_ne!(index, 0, "a dearer candidate must never win a tie");
            seen[index] = true;
        }
        assert_eq!(seen, [false, true, true]);
        assert_eq!(picker.pick(&[]), None);
    }

    #[test]
    fn zero_cost_ties_spread_across_equivalent_workers() {
        let policy = KvAwarePolicy::default();
        let components = [
            CostComponents {
                ptoken: 8,
                batchsize: 1,
            },
            CostComponents {
                ptoken: 8,
                batchsize: 1,
            },
            CostComponents {
                ptoken: 8,
                batchsize: 2,
            },
        ];
        let mut seen = [false; 3];
        for _ in 0..64 {
            let index = policy
                .pick_among(true, &[0.0, 0.0, 0.0], &components)
                .unwrap();
            assert_ne!(index, 2, "a busier worker must not win a full tie");
            seen[index] = true;
        }
        assert_eq!(seen, [true, true, false]);
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
        let policy = KvAwarePolicy::default();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new("http://warm".into(), WorkerType::Regular)),
            Arc::new(BasicWorker::new("http://cold".into(), WorkerType::Regular)),
        ];
        let warm = register_full(&policy, workers[0].url(), target("warm", 0), "model", 2);
        policy.register_target(workers[1].url(), target("cold", 0));
        let prompts: Arc<[Vec<u32>]> =
            Arc::from(vec![vec![1, 2, 3, 4, 5, 6], vec![11, 12, 13, 14, 15, 16]]);
        for (sequence, ids) in prompts.iter().enumerate() {
            warm_full(&warm, sequence as u64, &ids[..4]);
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
        let policy = KvAwarePolicy::default();
        let base = register_full(&policy, "http://base", target("http://base", 0), "model", 2);
        let lora = target("http://lora", 0);
        let adapter = ("adapter", "/adapters/adapter");
        let binding = Arc::new(full_binding_with_adapter(
            lora.clone(),
            2,
            adapter.0,
            adapter.1,
        ));
        binding.begin_empty(0);
        policy.register_target_with_cache_identity(
            "http://lora",
            lora,
            CacheKey {
                model: Arc::from("model"),
                block_size: 2,
            },
            binding.clone(),
        );
        let ids = [1_u32, 2, 3, 4];
        warm_full(&base, 0, &ids);
        binding
            .apply_batch(
                0,
                &event_batch(vec![KVEvent::BlockStored(full_store(
                    &ids,
                    2,
                    Some(adapter),
                ))]),
            )
            .unwrap();
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
    fn decode_credit_requires_every_candidate_to_be_eligible() {
        for (cacheable, salt, models, expected) in [
            (true, None, vec![true, true], true),
            (true, Some(""), vec![true], true),
            (false, None, vec![true], false),
            (true, Some("tenant"), vec![true], false),
            (true, None, vec![true, false], false),
            (true, None, vec![], false),
        ] {
            assert_eq!(
                KvAwarePolicy::all_candidates_priceable(cacheable, salt, models),
                expected
            );
        }
    }

    #[test]
    fn cache_salt_disables_incompatible_placement_credit() {
        for (salt, verified, expected) in [
            (None, true, None),
            (Some(""), true, None),
            (None, false, Some("unverified_model")),
            (Some("tenant-salt"), true, Some("cache_salt")),
            (Some("tenant-salt"), false, Some("cache_salt")),
        ] {
            assert_eq!(KvAwarePolicy::credit_fallback(salt, verified), expected);
        }
        let policy = KvAwarePolicy::default();
        let cached = register_full(
            &policy,
            "http://cached",
            target("http://cached", 0),
            "model",
            2,
        );
        policy.register_target("http://uncached", target("http://uncached", 0));
        let ids = [1_u32, 2, 3, 4];
        warm_full(&cached, 0, &ids);
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
        let policy = KvAwarePolicy::default();
        let first = target("first-instance", 0);
        let first_binding = register_full(&policy, "http://first", first.clone(), "model-a", 2);
        register_full(
            &policy,
            "http://second",
            target("second-instance", 0),
            "model-b",
            4,
        );
        let ids = [1_u32, 2, 3, 4];
        warm_full(&first_binding, 0, &ids);
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
        let policy = KvAwarePolicy::default();
        let first = register_full(&policy, "http://w1", target("http://w1", 0), "model", 2);
        policy.register_target("http://w2", target("http://w2", 0));
        let ids = [1_u32, 2, 3, 4];
        warm_full(&first, 0, &ids);
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

    #[test]
    fn optional_qualification_does_not_invoke_an_ordinary_only_implementer() {
        struct OrdinaryOnly {
            selections: AtomicUsize,
        }

        impl KvAwareOps for OrdinaryOnly {
            fn estimate_prefill_work<'tokens>(
                &self,
                _inputs: &[(&dyn Worker, &'tokens [Vec<u32>], bool)],
                _hashes: &mut PromptHashes<'tokens>,
            ) -> Vec<PrefillWorkEstimate> {
                unreachable!("ordinary-only qualification must not enter PD selection")
            }

            fn estimate_decode_work<'tokens>(
                &self,
                _inputs: &[(&dyn Worker, &'tokens [Vec<u32>], bool)],
                _hashes: &mut PromptHashes<'tokens>,
            ) -> Vec<DecodeWorkEstimate> {
                unreachable!("ordinary-only qualification must not enter PD selection")
            }

            fn pick_prefill(&self, _costs: &[f64], _occupancy: &[usize]) -> Option<usize> {
                unreachable!("ordinary-only qualification must not enter PD selection")
            }

            fn pick_decode(&self, _loads: &[usize], _additional: &[u64]) -> Option<usize> {
                unreachable!("ordinary-only qualification must not enter PD selection")
            }

            fn select_decode(
                &self,
                _workers: &[Arc<dyn Worker>],
                _session: Option<&AffinityKey>,
                _lora_name: Option<&str>,
                _cache_salt: Option<&str>,
                _new_context: Option<&HashMap<String, u64>>,
            ) -> Option<StagePick> {
                unreachable!("ordinary-only qualification must not enter PD selection")
            }

            fn session_affinity_enabled(&self) -> bool {
                false
            }

            fn register_worker(
                &self,
                _worker: &dyn Worker,
                _info: Option<&WorkerKvInfo>,
                _matches: Option<Arc<dyn TieredMatchProvider>>,
            ) {
            }

            fn unregister_worker(&self, _worker_url: &str) {}

            fn has_affinity(
                &self,
                _workers: &[Arc<dyn Worker>],
                _lora_name: Option<&str>,
                _cache_salt: Option<&str>,
                _affinity_key: &AffinityKey,
            ) -> bool {
                false
            }

            fn select_worker_with_inputs(
                &self,
                _workers: &[Arc<dyn Worker>],
                _tokens: KvRequestTokens<'_>,
                lora_name: Option<&str>,
                cache_salt: Option<&str>,
                key: Option<&AffinityKey>,
                _excluded_workers: &[String],
            ) -> Option<KvSelectionReceipt> {
                assert_eq!(lora_name, Some("adapter"));
                assert_eq!(cache_salt, Some("tenant-salt"));
                assert_eq!(key, Some(&affinity_key("ordinary")));
                self.selections.fetch_add(1, Ordering::Relaxed);
                None
            }
        }

        let ordinary = OrdinaryOnly {
            selections: AtomicUsize::new(0),
        };
        let ops: &dyn KvAwareOps = &ordinary;
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(BasicWorker::new(
            "http://ordinary".into(),
            WorkerType::Regular,
        ))];
        let denied = [false];
        let key = affinity_key("ordinary");
        for eligibility in [None, Some(denied.as_slice())] {
            assert!(ops
                .select_with_history(
                    &workers,
                    KvRequestTokens::Shared(Some(&[1, 2])),
                    eligibility,
                    None,
                    SelectionIdentity {
                        lora_name: Some("adapter"),
                        cache_salt: Some("tenant-salt"),
                        affinity_key: Some(&key),
                    },
                    &[],
                )
                .is_none());
        }
        assert_eq!(ordinary.selections.load(Ordering::Relaxed), 1);
        assert_eq!(workers[0].load(), 0);
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
        affinity_from_request(
            None,
            Some(&serde_json::json!({"session_id": session})),
            "default",
        )
        .unwrap()
    }

    fn completion_history_fixture() -> (KvAwarePolicy, Vec<Arc<dyn Worker>>, WorkerTokenInputs) {
        #[derive(Debug)]
        struct FirstPicker;
        impl Picker for FirstPicker {
            fn pick(&self, costs: &[f64]) -> Option<usize> {
                (!costs.is_empty()).then_some(0)
            }
        }
        let policy = KvAwarePolicy {
            picker: Arc::new(FirstPicker),
            ..Default::default()
        };
        let workers = registered_worker_pair(&policy);
        let mut inputs = WorkerTokenInputs::new();
        for (index, worker) in workers.iter().enumerate() {
            // One incumbent per worker keeps the cost positive so FirstPicker
            // makes a deterministic baseline before the advisory tie-break.
            worker.increment_load();
            policy.register_history_worker(
                worker,
                Some(&WorkerKvInfo {
                    instance_id: worker.url().to_owned(),
                    ranks: vec![0],
                    cache_key: Some(CacheKey {
                        model: Arc::from("model"),
                        block_size: 2,
                    }),
                    ..WorkerKvInfo::default()
                }),
                Some(Arc::new(EmptyMatchProvider)),
                Some(current_registration()),
            );
            inputs.insert(
                worker.base_url().to_owned(),
                Arc::from(vec![if index == 0 {
                    vec![9, 8, 7, 6]
                } else {
                    vec![1, 2, 3, 4]
                }]),
            );
        }
        (policy, workers, inputs)
    }

    #[test]
    fn completion_history_equal_overlap_keeps_ordinary_selection_and_full_cost() {
        let (policy, workers, inputs) = completion_history_fixture();
        let request = policy
            .capture_completion_history(&workers, "model")
            .unwrap();
        let select = |excluded: &[String]| {
            policy
                .select_with_history(
                    &workers,
                    KvRequestTokens::PerWorker(&inputs),
                    None,
                    Some(&request),
                    SelectionIdentity::default(),
                    excluded,
                )
                .unwrap()
        };
        for index in 0..2 {
            let (seed, observation) = select(&[workers[1 - index].url().to_owned()]);
            assert_eq!(seed.worker_index, index);
            policy.observe_completion_milestone(workers[index].as_ref(), observation.unwrap());
            release_receipt(&workers, &seed);
        }

        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let (receipt, observation) = metrics::with_local_recorder(&recorder, || select(&[]));
        assert_eq!(receipt.worker_index, 0);
        assert_eq!(receipt.new_prefill_tokens, 4);
        assert!(observation.is_some());
        let target = &receipt.decision.target;
        assert_eq!(
            policy.load.sample(target),
            LoadSample {
                active_prefill_tokens: 4,
                active_requests: 1,
            }
        );
        assert_eq!(workers[0].load(), 2);
        release_receipt(&workers, &receipt);
        assert_eq!(policy.load.sample(target), LoadSample::default());
        assert_eq!(workers[0].load(), 1);
        assert_eq!(workers[1].load(), 1);

        let rendered = recorder.handle().render();
        assert!(
            rendered.contains("reason=\"eligible_ordinary\""),
            "{rendered}"
        );
        assert!(!rendered.contains("reason=\"override\""), "{rendered}");
        assert!(!rendered.contains("outcome=\"history_hint\""), "{rendered}");
    }

    #[test]
    fn completion_history_is_advisory_with_worker_specific_inputs() {
        let (policy, workers, inputs) = completion_history_fixture();
        let ops: &dyn KvAwareOps = &policy;
        let request = policy
            .capture_completion_history(&workers, "model")
            .unwrap();
        let tokens = KvRequestTokens::PerWorker(&inputs);
        let exclude_first = [workers[0].url().to_owned()];
        let (seed, observation) = ops
            .select_with_history(
                &workers,
                tokens,
                None,
                Some(&request),
                SelectionIdentity::default(),
                &exclude_first,
            )
            .unwrap();
        assert_eq!(seed.worker_index, 1);
        ops.observe_completion_milestone(workers[1].as_ref(), observation.unwrap());
        release_receipt(&workers, &seed);
        let (receipt, observation) = ops
            .select_with_history(
                &workers,
                tokens,
                None,
                Some(&request),
                SelectionIdentity::default(),
                &[],
            )
            .unwrap();
        assert_eq!(receipt.worker_index, 1);
        assert_eq!(receipt.new_prefill_tokens, 4);
        assert!(observation.is_some());
        release_receipt(&workers, &receipt);
        assert_eq!(
            policy.observed_completion_prefix(workers[1].as_ref(), &[9, 8, 7, 6]),
            0
        );

        let key = affinity_key("scoped-history");
        for (case, lora_name, cache_salt) in [
            ("LoRA", Some("adapter"), None),
            ("salt", None, Some("tenant-salt")),
        ] {
            // Ordinary selection establishes a scoped binding to the peer
            // that the unbound FirstPicker would not choose on its own.
            let ordinary = ops
                .select_worker_with_qualified_inputs(
                    &workers,
                    tokens,
                    lora_name,
                    cache_salt,
                    Some(&key),
                    &exclude_first,
                    None,
                )
                .unwrap();
            assert_eq!(ordinary.worker_index, 1, "{case}");
            release_receipt(&workers, &ordinary);
            let binding = policy
                .affinity
                .get(&key.scoped_digest(lora_name, cache_salt))
                .unwrap();

            let (receipt, observation) = ops
                .select_with_history(
                    &workers,
                    tokens,
                    None,
                    Some(&request),
                    SelectionIdentity {
                        lora_name,
                        cache_salt,
                        affinity_key: Some(&key),
                    },
                    &[],
                )
                .unwrap();
            assert_eq!(receipt.worker_index, ordinary.worker_index, "{case}");
            assert_eq!(receipt.decision, ordinary.decision, "{case}");
            assert_eq!(receipt.new_prefill_tokens, 4, "{case}");
            assert!(observation.is_none(), "{case}");
            let target = &receipt.decision.target;
            assert_eq!(
                policy.load.sample(target),
                LoadSample {
                    active_prefill_tokens: 4,
                    active_requests: 1,
                },
                "{case}",
            );
            assert_eq!(workers[1].load(), 2, "{case}");
            assert_eq!(binding.in_flight(), 1, "{case}");
            release_receipt(&workers, &receipt);
            assert_eq!(policy.load.sample(target), LoadSample::default(), "{case}");
            assert_eq!(workers[1].load(), 1, "{case}");
            assert_eq!(binding.in_flight(), 0, "{case}");
        }
    }

    #[test]
    fn completion_history_rechecks_every_feasible_provider_before_using_history() {
        use std::sync::atomic::AtomicBool;

        #[derive(Debug)]
        struct RetiringProvider {
            current: AtomicBool,
        }
        impl TieredMatchProvider for RetiringProvider {}
        impl RegistrationCurrent for RetiringProvider {
            fn current(&self, rank: u32) -> bool {
                rank == 0 && self.current.load(Ordering::Acquire)
            }
        }

        #[derive(Debug)]
        struct RetireOnPick {
            provider: Arc<RetiringProvider>,
            ordinary: Arc<dyn Picker>,
        }
        impl Picker for RetireOnPick {
            fn pick(&self, costs: &[f64]) -> Option<usize> {
                // Both candidates were prepared with known misses, but the
                // ordinary winner loses its source before history refines the tie.
                self.provider.current.store(false, Ordering::Release);
                self.ordinary.pick(costs)
            }
        }

        let (mut policy, workers, inputs) = completion_history_fixture();
        let provider = Arc::new(RetiringProvider {
            current: AtomicBool::new(true),
        });
        policy.register_history_worker(
            &workers[0],
            Some(&WorkerKvInfo {
                instance_id: workers[0].url().to_owned(),
                ranks: vec![0],
                cache_key: policy
                    .binding_for_worker(workers[0].as_ref())
                    .cache_key
                    .clone(),
                ..WorkerKvInfo::default()
            }),
            Some(provider.clone()),
            Some(provider.clone()),
        );
        let request = policy
            .capture_completion_history(&workers, "model")
            .unwrap();
        let select = |policy: &KvAwarePolicy, excluded: &[String]| {
            policy
                .select_with_history(
                    &workers,
                    KvRequestTokens::PerWorker(&inputs),
                    None,
                    Some(&request),
                    SelectionIdentity::default(),
                    excluded,
                )
                .unwrap()
        };
        let (receipt, late) = select(&policy, &[workers[1].url().to_owned()]);
        assert_eq!(receipt.worker_index, 0);
        let late = late.unwrap();
        release_receipt(&workers, &receipt);

        let (receipt, seed) = select(&policy, &[workers[0].url().to_owned()]);
        assert_eq!(receipt.worker_index, 1);
        policy.observe_completion_milestone(workers[1].as_ref(), seed.unwrap());
        release_receipt(&workers, &receipt);

        let (receipt, _) = select(&policy, &[]);
        assert_eq!(receipt.worker_index, 1);
        assert_eq!(receipt.new_prefill_tokens, 4);
        release_receipt(&workers, &receipt);

        policy.picker = Arc::new(RetireOnPick {
            provider,
            ordinary: policy.picker.clone(),
        });
        let (receipt, observation) = select(&policy, &[]);
        assert_eq!(receipt.worker_index, 0);
        assert_eq!(receipt.new_prefill_tokens, 4);
        assert!(observation.is_none());
        release_receipt(&workers, &receipt);

        let retired_registration = late.registration_id;
        policy.observe_completion_milestone(workers[0].as_ref(), late);
        assert!(policy
            .completion_history
            .get()
            .unwrap()
            .match_prefix(retired_registration, &[9, 8, 7, 6], 2, Instant::now())
            .is_none());
    }

    #[test]
    fn completion_history_rejects_cross_policy_observations_for_the_same_worker() {
        let (policy, workers, inputs) = completion_history_fixture();
        let request = policy
            .capture_completion_history(&workers, "model")
            .unwrap();
        let (receipt, observation) = policy
            .select_with_context(
                &workers,
                QualifiedKvInput {
                    tokens: KvRequestTokens::PerWorker(&inputs),
                    eligibility: None,
                },
                SelectionIdentity::default(),
                &[workers[0].url().to_owned()],
                Some(&request),
            )
            .unwrap();
        release_receipt(&workers, &receipt);
        // Identical local generations and the same Worker object in another
        // policy do not make that policy the origin of this capture or ticket.
        let other = KvAwarePolicy::default();
        // Advance the initial pair's generations, as in the original fixture.
        for worker in &workers {
            other.register_target(worker.url(), other.target_for_worker(worker.as_ref()));
        }
        for worker in &workers {
            let binding = policy.binding_for_worker(worker.as_ref());
            other.register_history_worker(
                worker,
                Some(&WorkerKvInfo {
                    instance_id: binding.target.instance_id.to_string(),
                    ranks: vec![0],
                    cache_key: binding.cache_key.clone(),
                    ..WorkerKvInfo::default()
                }),
                Some(Arc::new(EmptyMatchProvider)),
                Some(current_registration()),
            );
            assert_eq!(
                other.binding_for_worker(worker.as_ref()).generation,
                binding.generation,
            );
        }
        let other_capture = other.capture_completion_history(&workers, "model").unwrap();
        assert!(request
            .registration_id(
                workers[0].as_ref(),
                &other.binding_for_worker(workers[0].as_ref())
            )
            .is_none());
        assert!(other_capture
            .registration_id(
                workers[0].as_ref(),
                &policy.binding_for_worker(workers[0].as_ref())
            )
            .is_none());
        other.observe_completion_milestone(workers[1].as_ref(), observation.unwrap());
        assert_eq!(
            other.observed_completion_prefix(workers[1].as_ref(), &[1, 2, 3, 4]),
            0
        );
    }

    #[test]
    fn completion_history_requires_qualified_single_worker_input() {
        let (policy, workers, inputs) = completion_history_fixture();
        let request = policy
            .capture_completion_history(&workers, "model")
            .unwrap();
        let input = QualifiedKvInput {
            tokens: KvRequestTokens::PerWorker(&inputs),
            eligibility: None,
        };
        let (seed, observation) = policy
            .select_with_context(
                &workers,
                input,
                SelectionIdentity::default(),
                &[workers[0].url().to_owned()],
                Some(&request),
            )
            .unwrap();
        policy.observe_completion_milestone(workers[1].as_ref(), observation.unwrap());
        release_receipt(&workers, &seed);
        // Missing qualification, batch inputs, and wrong request model cannot
        // produce observations, even if their generation has stored history.
        let wrong_model = policy
            .capture_completion_history(&workers, "other-model")
            .unwrap();
        let mut batch = inputs.clone();
        batch.insert(
            workers[1].base_url().to_owned(),
            Arc::from(vec![vec![1, 2], vec![3, 4]]),
        );
        for (name, tokens, mask, captured) in [
            (
                "ineligible",
                KvRequestTokens::PerWorker(&inputs),
                Some([false, false]),
                &request,
            ),
            ("batch", KvRequestTokens::PerWorker(&batch), None, &request),
            (
                "model",
                KvRequestTokens::PerWorker(&inputs),
                None,
                &wrong_model,
            ),
        ] {
            let (receipt, observation) = policy
                .select_with_context(
                    &workers,
                    QualifiedKvInput {
                        tokens,
                        eligibility: mask.as_ref().map(|m| m.as_slice()),
                    },
                    SelectionIdentity::default(),
                    &[workers[0].url().to_owned()],
                    Some(captured),
                )
                .unwrap();
            assert_eq!(receipt.worker_index, 1, "{name}");
            assert!(observation.is_none(), "{name}");
            release_receipt(&workers, &receipt);
        }
        // A missing or empty prompt is unknown. A nonempty prompt shorter
        // than a block has known zero credit, but still cannot override cost.
        for (case, ids, charge, observed, reason) in [
            ("missing", None, 4, false, "missing_evidence"),
            ("empty", Some(vec![]), 0, false, "missing_evidence"),
            ("short", Some(vec![9]), 1, true, "no_tie"),
        ] {
            let mut per_worker = inputs.clone();
            let peer = workers[0].base_url();
            if let Some(ids) = ids {
                per_worker.insert(peer.to_owned(), Arc::from(vec![ids]));
            } else {
                per_worker.remove(peer);
            }
            let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
            let (receipt, observation) = metrics::with_local_recorder(&recorder, || {
                policy.select_with_context(
                    &workers,
                    QualifiedKvInput {
                        tokens: KvRequestTokens::PerWorker(&per_worker),
                        eligibility: None,
                    },
                    SelectionIdentity::default(),
                    &[],
                    Some(&request),
                )
            })
            .unwrap();
            assert_eq!(receipt.worker_index, 0, "{case}");
            assert_eq!(receipt.new_prefill_tokens, charge, "{case}");
            assert_eq!(observation.is_some(), observed, "{case}");
            release_receipt(&workers, &receipt);
            let rendered = recorder.handle().render();
            let mut decisions = rendered
                .lines()
                .filter(|line| line.starts_with("vllm_router_history_decisions_total{"));
            assert!(
                decisions.next().is_some_and(|line| {
                    line.contains(&format!("reason=\"{reason}\"")) && line.ends_with(" 1")
                }) && decisions.next().is_none(),
                "{case}: {rendered}",
            );
        }
    }

    #[test]
    fn completion_history_shared_grpc_input_keeps_identity_and_full_charge() {
        let policy = KvAwarePolicy::default();
        let workers: Vec<Arc<dyn Worker>> = ["grpc://a", "grpc://b"]
            .into_iter()
            .map(|url| {
                Arc::new(BasicWorker::with_connection_mode(
                    url.into(),
                    WorkerType::Regular,
                    crate::core::ConnectionMode::Grpc,
                )) as Arc<dyn Worker>
            })
            .collect();
        let matches = Arc::new(EmptyMatchProvider);
        for worker in &workers {
            worker.increment_load();
            policy.register_history_worker(
                worker,
                Some(&WorkerKvInfo {
                    instance_id: worker.url().into(),
                    ranks: vec![0],
                    cache_key: Some(CacheKey {
                        model: Arc::from("model"),
                        block_size: 16,
                    }),
                    ..Default::default()
                }),
                Some(matches.clone()),
                Some(current_registration()),
            );
        }
        let request = policy
            .capture_completion_history(&workers, "model")
            .unwrap();
        let ids = [7; 33];
        let input = QualifiedKvInput {
            tokens: KvRequestTokens::Shared(Some(&ids)),
            eligibility: None,
        };
        let (seed, observation) = policy
            .select_with_context(
                &workers,
                input,
                SelectionIdentity::default(),
                &[workers[0].url().into()],
                Some(&request),
            )
            .unwrap();
        assert_eq!(seed.new_prefill_tokens, 33);
        assert_eq!(
            policy.observed_completion_prefix(workers[1].as_ref(), &ids),
            0
        );
        policy.observe_completion_milestone(workers[1].as_ref(), observation.unwrap());
        release_receipt(&workers, &seed);
        assert_eq!(
            policy.observed_completion_prefix(workers[1].as_ref(), &ids),
            32
        );
        // The second candidate reuses the first one's placement lookup.
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let (next, observation) = metrics::with_local_recorder(&recorder, || {
            policy.select_with_context(
                &workers,
                input,
                SelectionIdentity::default(),
                &[],
                Some(&request),
            )
        })
        .unwrap();
        assert_eq!(next.worker_index, 1);
        assert_eq!(next.new_prefill_tokens, 33);
        assert!(observation.is_some());
        release_receipt(&workers, &next);
        let rendered = recorder.handle().render();
        let mut decisions = rendered
            .lines()
            .filter(|line| line.starts_with("vllm_router_history_decisions_total{"));
        assert!(
            decisions.next().is_some_and(|line| {
                (line.contains("reason=\"override\"")
                    || line.contains("reason=\"eligible_ordinary\""))
                    && line.ends_with(" 1")
            }) && decisions.next().is_none(),
            "{rendered}",
        );
        for worker in &workers {
            assert_eq!(worker.load(), 1, "background load remains");
            worker.decrement_load();
        }
    }

    #[test]
    fn completion_history_respects_binding_load_exclusions_and_credit() {
        let (policy, workers, inputs) = completion_history_fixture();
        let request = policy
            .capture_completion_history(&workers, "model")
            .unwrap();
        let input = QualifiedKvInput {
            tokens: KvRequestTokens::PerWorker(&inputs),
            eligibility: None,
        };
        let (seed, observation) = policy
            .select_with_context(
                &workers,
                input,
                SelectionIdentity::default(),
                &[workers[0].url().to_owned()],
                Some(&request),
            )
            .unwrap();
        policy.observe_completion_milestone(workers[1].as_ref(), observation.unwrap());
        release_receipt(&workers, &seed);
        let key = affinity_key("history-cannot-rebind");
        let bound = policy
            .select_worker_with_qualified_inputs(
                &workers,
                input,
                None,
                None,
                Some(&key),
                &[workers[1].url().to_owned()],
            )
            .unwrap();
        release_receipt(&workers, &bound);
        let (receipt, _) = policy
            .select_with_context(
                &workers,
                input,
                SelectionIdentity {
                    lora_name: None,
                    cache_salt: None,
                    affinity_key: Some(&key),
                },
                &[],
                Some(&request),
            )
            .unwrap();
        assert_eq!(receipt.worker_index, 0);
        release_receipt(&workers, &receipt);
        workers[1].increment_load();
        let (receipt, _) = policy
            .select_with_context(
                &workers,
                input,
                SelectionIdentity::default(),
                &[],
                Some(&request),
            )
            .unwrap();
        assert_eq!(receipt.worker_index, 0);
        release_receipt(&workers, &receipt);
        workers[1].decrement_load();
        for (excluded, expected) in [
            (vec![workers[1].url().to_owned()], 0),
            (
                workers
                    .iter()
                    .map(|worker| worker.url().to_owned())
                    .collect(),
                1,
            ),
        ] {
            let (receipt, _) = policy
                .select_with_context(
                    &workers,
                    input,
                    SelectionIdentity::default(),
                    &excluded,
                    Some(&request),
                )
                .unwrap();
            assert_eq!(receipt.worker_index, expected);
            release_receipt(&workers, &receipt);
        }
        struct Credit(RoutableTarget, StorageTier);
        impl TieredMatchProvider for Credit {
            fn matches(&self, _: &[u32], _: u32, _: Option<&str>, _: u32) -> Vec<TierMatch> {
                vec![tier_match(&self.0, self.1, &[(0, 1)])]
            }
        }
        let binding = policy.targets.get(workers[0].url()).unwrap().clone();
        // At B2 the disk tier supplies 0.6 token of credit: the bill remains
        // four, but this must still suppress the warm peer's history override.
        for (tier, expected_charge) in [(StorageTier::Device, 2), (StorageTier::Disk, 4)] {
            policy.register_history_worker(
                &workers[0],
                Some(&WorkerKvInfo {
                    instance_id: binding.target.instance_id.to_string(),
                    ranks: vec![0],
                    cache_key: binding.cache_key.clone(),
                    ..WorkerKvInfo::default()
                }),
                Some(Arc::new(Credit(binding.target.clone(), tier))),
                Some(current_registration()),
            );
            let current = policy
                .capture_completion_history(&workers, "model")
                .unwrap();
            let (receipt, _) = policy
                .select_with_context(
                    &workers,
                    input,
                    SelectionIdentity::default(),
                    &[],
                    Some(&current),
                )
                .unwrap();
            assert_eq!(receipt.worker_index, 0);
            assert_eq!(receipt.new_prefill_tokens, expected_charge);
            release_receipt(&workers, &receipt);
        }
    }

    #[test]
    fn completion_history_production_picker_stays_within_eligible_set() {
        let (mut policy, workers, inputs) = completion_history_fixture();
        policy.picker = Arc::new(TemperaturePicker::new(0.0));
        let captured = policy
            .capture_completion_history(&workers, "model")
            .unwrap();
        let input = QualifiedKvInput {
            tokens: KvRequestTokens::PerWorker(&inputs),
            eligibility: None,
        };
        for target in 0..2 {
            let (receipt, observation) = policy
                .select_with_context(
                    &workers,
                    input,
                    SelectionIdentity::default(),
                    &[workers[1 - target].url().to_owned()],
                    Some(&captured),
                )
                .unwrap();
            assert_eq!(receipt.worker_index, target);
            policy.observe_completion_milestone(workers[target].as_ref(), observation.unwrap());
            release_receipt(&workers, &receipt);
        }
        for excluded in [None, Some(0), Some(1)] {
            let excluded_urls = excluded
                .map(|i| vec![workers[i].url().to_owned()])
                .unwrap_or_default();
            let (receipt, _) = policy
                .select_with_context(
                    &workers,
                    input,
                    SelectionIdentity::default(),
                    &excluded_urls,
                    Some(&captured),
                )
                .unwrap();
            assert!(receipt.worker_index < workers.len());
            assert_ne!(Some(receipt.worker_index), excluded);
            assert_eq!(receipt.new_prefill_tokens, 4);
            release_receipt(&workers, &receipt);
        }
    }

    #[test]
    fn completion_history_requires_complete_binding_block() {
        let (policy, workers, inputs) = completion_history_fixture();
        let captured = policy
            .capture_completion_history(&workers, "model")
            .unwrap();
        let (receipt, observation) = policy
            .select_with_context(
                &workers,
                QualifiedKvInput {
                    tokens: KvRequestTokens::PerWorker(&inputs),
                    eligibility: None,
                },
                SelectionIdentity::default(),
                &[workers[0].url().to_owned()],
                Some(&captured),
            )
            .unwrap();
        policy.observe_completion_milestone(workers[1].as_ref(), observation.unwrap());
        release_receipt(&workers, &receipt);
        for (ids, expected) in [
            (vec![1], 0),
            (vec![1, 2], 2),
            (vec![1, 2, 3], 2),
            (vec![1, 2, 3, 4], 4),
        ] {
            assert_eq!(
                policy.observed_completion_prefix(workers[1].as_ref(), &ids),
                expected,
                "{ids:?}"
            );
        }
    }

    #[test]
    fn completion_history_capture_cannot_cross_registration_or_worker_replacement() {
        let (policy, workers, inputs) = completion_history_fixture();
        let captured = policy
            .capture_completion_history(&workers, "model")
            .unwrap();
        let input = QualifiedKvInput {
            tokens: KvRequestTokens::PerWorker(&inputs),
            eligibility: None,
        };
        let mut pending = Vec::new();
        for index in 0..2 {
            let (receipt, observation) = policy
                .select_with_context(
                    &workers,
                    input,
                    SelectionIdentity::default(),
                    &[workers[1 - index].url().to_owned()],
                    Some(&captured),
                )
                .unwrap();
            let binding = policy.targets.get(workers[index].url()).unwrap().clone();
            let registration_id = captured
                .registration_id(workers[index].as_ref(), &binding)
                .unwrap();
            // One old successful observation plus one still in flight.
            policy.observe_completion_milestone(
                workers[index].as_ref(),
                HistoryObservation::new(registration_id, &[1, 2, 3, 4]),
            );
            pending.push((observation.unwrap(), binding));
            release_receipt(&workers, &receipt);
        }
        let (old_observation, old_binding) = pending.remove(0);
        policy.unregister_target(workers[0].url());
        policy.register_history_worker(
            &workers[0],
            Some(&WorkerKvInfo {
                instance_id: old_binding.target.instance_id.to_string(),
                ranks: vec![0],
                cache_key: old_binding.cache_key.clone(),
                ..WorkerKvInfo::default()
            }),
            Some(Arc::new(EmptyMatchProvider)),
            Some(current_registration()),
        );
        policy.observe_completion_milestone(workers[0].as_ref(), old_observation);
        let new_binding = policy.targets.get(workers[0].url()).unwrap().clone();
        assert!(captured
            .registration_id(workers[0].as_ref(), &new_binding)
            .is_none());
        assert!(policy
            .completion_history
            .get()
            .unwrap()
            .match_prefix(
                new_binding.registration_id,
                &[1, 2, 3, 4],
                new_binding.block_size,
                Instant::now()
            )
            .is_none());
        let (receipt, observation) = policy
            .select_with_context(
                &workers,
                input,
                SelectionIdentity::default(),
                &[workers[1].url().to_owned()],
                Some(&captured),
            )
            .unwrap();
        assert_eq!(receipt.new_prefill_tokens, 4);
        assert!(observation.is_none());
        release_receipt(&workers, &receipt);
        // A different Worker object at the unchanged URL cannot use captured
        // preparation evidence; the untouched peer still retains its hint.
        let replacement: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            workers[1].url().to_owned(),
            WorkerType::Regular,
        ));
        let peer = policy.targets.get(workers[1].url()).unwrap();
        assert!(captured
            .registration_id(replacement.as_ref(), &peer)
            .is_none());
        assert!(captured
            .registration_id(workers[1].as_ref(), &peer)
            .is_some());
        assert!(policy
            .completion_history
            .get()
            .unwrap()
            .match_prefix(
                peer.registration_id,
                &[1, 2, 3, 4],
                peer.block_size,
                Instant::now()
            )
            .is_some());
        let peer_registration_id = peer.registration_id;
        drop(peer);
        let replaced_workers = vec![workers[0].clone(), replacement.clone()];
        let fresh = policy
            .capture_completion_history(&replaced_workers, "model")
            .unwrap();
        let binding = policy.targets.get(replacement.url()).unwrap();
        assert!(fresh
            .registration_id(replacement.as_ref(), &binding)
            .is_none());
        drop(binding);
        let (receipt, observation) = policy
            .select_with_context(
                &replaced_workers,
                input,
                SelectionIdentity::default(),
                &[workers[0].url().to_owned()],
                Some(&fresh),
            )
            .unwrap();
        assert_eq!(receipt.worker_index, 1);
        assert_eq!(receipt.new_prefill_tokens, 4);
        assert!(observation.is_none());
        release_receipt(&replaced_workers, &receipt);
        policy.observe_completion_milestone(
            replacement.as_ref(),
            HistoryObservation::new(peer_registration_id, &[77, 88]),
        );
        assert!(policy
            .completion_history
            .get()
            .unwrap()
            .match_prefix(peer_registration_id, &[77, 88], 2, Instant::now())
            .is_none());
        // The registry replacement attempt did not invalidate the real peer.
        let (receipt, _) = policy
            .select_with_context(
                &workers,
                input,
                SelectionIdentity::default(),
                &[],
                Some(&captured),
            )
            .unwrap();
        assert_eq!(receipt.worker_index, 1);
        release_receipt(&workers, &receipt);
    }

    #[test]
    fn affinity_preparation_hint_rejects_stale_and_out_of_scope_bindings() {
        let policy = KvAwarePolicy::default();
        let workers = registered_worker_pair(&policy);
        let key = affinity_key("session-a");
        assert!(!policy.has_affinity(&workers, None, None, &key));
        let receipt = policy
            .select_worker_with_receipt(&workers, None, None, None, Some(&key), &[])
            .unwrap();
        release_receipt(&workers, &receipt);
        assert!(policy.has_affinity(&workers, None, None, &key));
        assert!(!policy.has_affinity(&workers, Some("adapter"), None, &key));
        assert!(!policy.has_affinity(&workers, None, Some("salt"), &key));
        assert!(!policy.has_affinity(&[], None, None, &key));
        let worker = &workers[receipt.worker_index];
        worker.set_healthy(false);
        assert!(!policy.has_affinity(&workers, None, None, &key));
        worker.set_healthy(true);
        let record = policy.affinity.get(&key.scoped_digest(None, None)).unwrap();
        record.state.lock().idle_deadline = Some(Instant::now());
        assert!(!policy.has_affinity(&workers, None, None, &key));
        record.state.lock().idle_deadline = None;
        policy.register_target(worker.url(), record.target.clone());
        assert!(!policy.has_affinity(&workers, None, None, &key));
    }

    #[test]
    fn untokenized_binding_never_credits_prefill() {
        let policy = KvAwarePolicy::default();
        let workers = registered_worker_pair(&policy);
        let key = affinity_key("tools-session");

        // First turn has no exact tokens: it binds, but with no credit.
        let first = policy
            .select_worker_with_receipt(&workers, None, None, None, Some(&key), &[])
            .unwrap();
        assert_eq!(first.new_prefill_tokens, 0);
        release_receipt(&workers, &first);

        // Later turns reuse the worker, still without any prefill reduction.
        workers[first.worker_index].increment_load();
        let second = policy
            .select_worker_with_receipt(&workers, None, None, None, Some(&key), &[])
            .unwrap();
        assert_eq!(second.worker_index, first.worker_index);
        assert_eq!(second.new_prefill_tokens, 0);
        release_receipt(&workers, &second);

        // Identity isolation: a LoRA or salt changes the binding scope.
        let adapter = policy
            .select_worker_with_receipt(&workers, None, Some("adapter"), None, Some(&key), &[])
            .unwrap();
        assert!(policy
            .affinity
            .get(&key.scoped_digest(Some("adapter"), None))
            .is_some());
        release_receipt(&workers, &adapter);
        workers[first.worker_index].decrement_load();

        // With exact tokens but no verified cache, the same binding must still
        // charge the full input: affinity picks the worker, never the credit.
        let prompt: Vec<u32> = (0..64).collect();
        let tokenized = policy
            .select_worker_with_receipt(&workers, Some(&prompt), None, None, Some(&key), &[])
            .unwrap();
        assert_eq!(tokenized.worker_index, first.worker_index);
        assert_eq!(tokenized.new_prefill_tokens, prompt.len() as u64);
        release_receipt(&workers, &tokenized);
    }

    #[test]
    fn affinity_keeps_failover_binding_when_old_worker_recovers() {
        let policy = KvAwarePolicy::default();
        let workers = registered_worker_pair(&policy);
        let key = affinity_key("failover-session");
        let prompt: Vec<u32> = (0..64).collect();

        let first = policy
            .select_worker_with_receipt(&workers, Some(&prompt), None, None, Some(&key), &[])
            .unwrap();
        let original = first.worker_index;
        release_receipt(&workers, &first);
        workers[original].set_healthy(false);

        let failover = policy
            .select_worker_with_receipt(&workers, None, None, None, Some(&key), &[])
            .unwrap();
        assert_ne!(failover.worker_index, original);
        let rebound = failover.worker_index;
        release_receipt(&workers, &failover);
        workers[original].set_healthy(true);

        let recovered = policy
            .select_worker_with_receipt(&workers, Some(&prompt), None, None, Some(&key), &[])
            .unwrap();
        assert_eq!(recovered.worker_index, rebound);
        assert_eq!(recovered.new_prefill_tokens, prompt.len() as u64);
        release_receipt(&workers, &recovered);
    }

    #[test]
    fn affinity_reuses_recorded_worker_across_requests() {
        let policy = KvAwarePolicy::default();
        let workers = registered_worker_pair(&policy);
        let key = affinity_key("session-a");

        let first = policy
            .select_worker_with_receipt(&workers, None, None, None, Some(&key), &[])
            .unwrap();
        // Another in-flight request shares the binding; releasing both must
        // leave it available for the next request.
        let concurrent = policy
            .select_worker_with_receipt(&workers, None, None, None, Some(&key), &[])
            .unwrap();
        assert_eq!(concurrent.worker_index, first.worker_index);
        release_receipt(&workers, &concurrent);
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
        let targets = [target("instance-a", 0), target("instance-b", 0)];
        let urls = ["http://a", "http://b"];
        let key = affinity_key("session-a");

        // The idle pair is an exact tie, so the first placement may bind
        // either worker; keep the rest of the test relative to that choice.
        let bound = policy
            .select_worker_with_receipt(&workers, None, None, None, Some(&key), &[])
            .unwrap();
        let first = bound.worker_index;
        let second = 1 - first;
        assert_eq!(bound.decision.target, targets[first]);
        release_receipt(&workers, &bound);

        // Removal invalidates the binding; under load the next request rebinds.
        workers[first].increment_load();
        policy.unregister_target(urls[first]);
        policy.register_target(urls[first], targets[first].clone());
        let rebound = policy
            .select_worker_with_receipt(&workers, None, None, None, Some(&key), &[])
            .unwrap();
        assert_eq!(rebound.decision.target, targets[second]);
        release_receipt(&workers, &rebound);

        // Re-registering the same target bumps its registration generation,
        // so the stale record rebinds through fresh selection.
        workers[second].increment_load();
        workers[second].increment_load();
        policy.register_target(urls[second], targets[second].clone());
        let fresh = policy
            .select_worker_with_receipt(&workers, None, None, None, Some(&key), &[])
            .unwrap();
        assert_eq!(fresh.decision.target, targets[first]);
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
        let key = affinity_from_request(None, Some(&body), "model").unwrap();
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
        let key = affinity_from_request(None, Some(&canonical), "model").unwrap();
        assert_eq!(
            (key.source, key.identity.as_str()),
            ("agentic_context", "task-a:researcher")
        );

        let agent_hint = serde_json::json!({
            "agent_hint": {"session_id": "hint-session"}
        });
        let key = affinity_from_request(
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
        let key = affinity_from_request(Some(&claude), None, "model").unwrap();
        assert_eq!(
            (key.source, key.identity.as_str()),
            ("framework_header", "claude-session:researcher")
        );

        let mut codex = header_map("session-id", "codex-session");
        codex.insert("thread-id", "worker".parse().unwrap());
        let key = affinity_from_request(Some(&codex), None, "model").unwrap();
        assert_eq!(key.identity, "codex-session:worker");

        let body = serde_json::json!({"session_id": "body-session"});
        let key = affinity_from_request(
            Some(&header_map("x-session-id", "generic-session")),
            Some(&body),
            "model",
        )
        .unwrap();
        assert_eq!(key.identity, "generic-session:lead");

        let key = affinity_from_request(None, Some(&body), "model").unwrap();
        assert_eq!(
            (key.source, key.identity.as_str()),
            ("body_session", "body-session")
        );
        let params = serde_json::json!({"session_params": {"session_id": "params-session"}});
        let key = affinity_from_request(None, Some(&params), "model").unwrap();
        assert_eq!(key.identity, "params-session");
    }

    #[test]
    fn affinity_key_rejects_unstable_and_malformed_identities() {
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", "request-1".parse().unwrap());
        headers.insert("x-trace-id", "trace-1".parse().unwrap());
        assert!(affinity_from_request(Some(&headers), None, "model").is_none());

        let user_scoped = serde_json::json!({"user": "user-a", "user_id": "user-a"});
        assert!(affinity_from_request(None, Some(&user_scoped), "model").is_none());

        let malformed_hint = serde_json::json!({"agent_hint": {"session_id": []}});
        assert!(affinity_from_request(None, Some(&malformed_hint), "model").is_none());

        let malformed_context = serde_json::json!({
            "vllm_xargs": {"agentic_context": {"task_id": [], "agent_id": "worker"}}
        });
        assert!(affinity_from_request(None, Some(&malformed_context), "model").is_none());
        for params in [serde_json::json!({"session_id": []}), serde_json::json!([])] {
            let body = serde_json::json!({"session_params": params, "session_id": "fallback"});
            assert!(affinity_from_request(None, Some(&body), "model").is_none());
        }
        for params in [serde_json::json!({}), JsonValue::Null] {
            let body = serde_json::json!({"session_params": params, "session_id": "fallback"});
            assert_eq!(
                affinity_from_request(None, Some(&body), "model")
                    .unwrap()
                    .identity,
                "fallback"
            );
        }
    }

    #[test]
    fn affinity_digest_scopes_model_lora_salt_and_identity_kind() {
        let body = serde_json::json!({"session_id": "session-a"});
        let key = affinity_from_request(None, Some(&body), "model-a").unwrap();
        let header_key = affinity_from_request(
            Some(&header_map("x-session-id", "session-a")),
            None,
            "model-a",
        )
        .unwrap();
        assert_ne!(
            key.scoped_digest(None, None),
            header_key.scoped_digest(None, None)
        );

        let other_model = affinity_from_request(None, Some(&body), "model-b").unwrap();
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
    fn placement_memos_keep_distinct_providers_and_reuse_shared_arcs() {
        struct Matches {
            calls: AtomicUsize,
            hits: Vec<TierMatch>,
        }
        impl TieredMatchProvider for Matches {
            fn matches(&self, _: &[u32], _: u32, _: Option<&str>, _: u32) -> Vec<TierMatch> {
                self.calls.fetch_add(1, Ordering::Relaxed);
                self.hits.clone()
            }
        }
        let policy = KvAwarePolicy::default();
        let workers = registered_worker_pair(&policy);
        let first = Arc::new(Matches {
            calls: AtomicUsize::new(0),
            hits: Vec::new(),
        });
        let second = Arc::new(Matches {
            calls: AtomicUsize::new(0),
            hits: vec![tier_match(
                &target(workers[1].url(), 0),
                StorageTier::Device,
                &[(0, 1)],
            )],
        });
        let info = WorkerKvInfo {
            cache_key: Some(CacheKey {
                model: Arc::from("shared"),
                block_size: 2,
            }),
            ..Default::default()
        };
        for (worker, provider) in workers.iter().zip([
            first.clone() as Arc<dyn TieredMatchProvider>,
            second.clone(),
        ]) {
            policy.register_worker(worker.as_ref(), Some(&info), Some(provider));
        }
        let prompts = [vec![1, 2, 3, 4]];
        for shared in [false, true] {
            if shared {
                policy.register_worker(workers[1].as_ref(), Some(&info), Some(first.clone()));
            }
            let calls_before = [
                first.calls.load(Ordering::Relaxed),
                second.calls.load(Ordering::Relaxed),
            ];
            // An exclusion fixes the selected target, not its placement credit.
            let receipt = policy
                .select_worker_with_receipt(
                    &workers,
                    Some(&prompts[0]),
                    None,
                    None,
                    None,
                    &[workers[0].url().to_owned()],
                )
                .unwrap();
            let inputs = [
                (workers[0].as_ref(), prompts.as_slice(), true),
                (workers[1].as_ref(), prompts.as_slice(), true),
            ];
            let mut hashes = PromptHashes::new();
            let prefill = policy.estimate_prefill_work(&inputs, &mut hashes);
            let decode = policy.estimate_decode_work(&inputs, &mut hashes);
            let expected = if shared { 4 } else { 2 };
            let actual = (
                receipt.new_prefill_tokens,
                prefill
                    .iter()
                    .map(|estimate| estimate.new_prefill_tokens)
                    .collect::<Vec<_>>(),
                decode
                    .iter()
                    .map(|estimate| estimate.additional_tokens)
                    .collect::<Vec<_>>(),
            );
            release_receipt(&workers, &receipt);
            assert_eq!(
                actual,
                (expected, vec![4, expected], vec![Some(4), Some(expected)]),
                "shared={shared}"
            );
            assert_eq!(receipt.worker_index, 1);
            assert!(prefill.iter().all(|estimate| estimate.placement_verified));
            assert_eq!(
                [
                    first.calls.load(Ordering::Relaxed) - calls_before[0],
                    second.calls.load(Ordering::Relaxed) - calls_before[1],
                ],
                [3, if shared { 0 } else { 3 }],
                "each request-local memo reuses only the same provider Arc"
            );
            assert_eq!(
                policy.load.sample(&receipt.decision.target),
                LoadSample::default()
            );
            assert_eq!(workers[1].load(), 0);
        }
    }

    #[test]
    fn pd_prefill_queries_once_per_cache_identity_and_prompt() {
        struct Matches {
            calls: AtomicUsize,
            hits: Vec<TierMatch>,
        }
        impl TieredMatchProvider for Matches {
            fn matches(&self, _: &[u32], _: u32, _: Option<&str>, _: u32) -> Vec<TierMatch> {
                self.calls.fetch_add(1, Ordering::Relaxed);
                self.hits.clone()
            }
        }
        let policy = KvAwarePolicy::default();
        let workers: Vec<_> = (0..3)
            .map(|i| BasicWorker::new(format!("http://p{i}"), WorkerType::Regular))
            .collect();
        let targets: Vec<_> = workers
            .iter()
            .map(|worker| target(worker.url(), 0))
            .collect();
        let matches = Arc::new(Matches {
            calls: AtomicUsize::new(0),
            hits: vec![
                tier_match(&targets[0], StorageTier::Device, &[(0, 1)]),
                tier_match(&targets[1], StorageTier::Device, &[(0, 2)]),
            ],
        });
        for (i, worker) in workers.iter().enumerate() {
            policy.register_target_with_cache_identity(
                worker.url(),
                targets[i].clone(),
                CacheKey {
                    model: Arc::from(if i == 2 { "other" } else { "shared" }),
                    block_size: 2,
                },
                matches.clone(),
            );
        }
        let prompts = [vec![1, 2, 3, 4, 5, 6]];
        let batch = [prompts[0].clone(), prompts[0].clone()];
        let other = [vec![1, 2, 3, 4, 5, 6, 7, 8]];
        let work = policy.estimate_prefill_work(
            &[
                (&workers[0], &prompts, false),
                (&workers[0], &batch, true),
                (&workers[1], &prompts, true),
                (&workers[2], &prompts, true),
                (&workers[0], &other, true),
            ],
            &mut PromptHashes::new(),
        );
        assert_eq!(matches.calls.load(Ordering::Relaxed), 3);
        assert_eq!(
            work.iter()
                .map(|work| work.new_prefill_tokens)
                .collect::<Vec<_>>(),
            [6, 8, 2, 6, 6]
        );
        assert_eq!(
            work.iter()
                .map(|work| work.placement_verified)
                .collect::<Vec<_>>(),
            [false, true, true, true, true]
        );
    }

    #[test]
    fn churn_fallback_preserves_each_workers_known_input() {
        #[derive(Default)]
        struct Churn {
            replace: Mutex<Option<Box<dyn Fn() + Send>>>,
            queries: AtomicUsize,
        }
        impl TieredMatchProvider for Churn {
            fn matches(&self, _: &[u32], _: u32, _: Option<&str>, _: u32) -> Vec<TierMatch> {
                self.queries.fetch_add(1, Ordering::Relaxed);
                self.replace.lock().as_ref().unwrap()();
                Vec::new()
            }
        }
        let both_known: [Option<&[usize]>; 2] = [Some(&[33]), Some(&[17])];
        let missing_second = [both_known[0], None];
        let cases = [
            ("known and missing", missing_second, None, None, 33, 3),
            ("both known", both_known, None, Some(1), 17, 6),
            ("forced longer", both_known, Some(1), Some(0), 33, 6),
            ("forced shorter", both_known, Some(0), Some(1), 17, 6),
            ("forced missing", missing_second, Some(0), Some(1), 33, 3),
            (
                "empty prompt",
                [Some(&[33]), Some(&[0])],
                None,
                Some(1),
                0,
                3,
            ),
            ("all unknown", [None, None], None, None, 0, 0),
            ("two prompts", [Some(&[17, 16]); 2], None, None, 33, 6),
        ];
        for history_enabled in [false, true] {
            let policy = Arc::new(KvAwarePolicy::new(CostParams {
                router_temperature: 0.0,
                ..CostParams::default()
            }));
            let workers = registered_worker_pair(&policy);
            let matches = Arc::new(Churn::default());
            let info = WorkerKvInfo {
                ranks: vec![0],
                cache_key: Some(CacheKey {
                    model: Arc::from("model"),
                    block_size: 16,
                }),
                ..WorkerKvInfo::default()
            };
            let register =
                move |policy: &KvAwarePolicy, worker: &Arc<dyn Worker>, matches: Arc<Churn>| {
                    if history_enabled {
                        policy.register_history_worker(
                            worker,
                            Some(&info),
                            Some(matches),
                            Some(current_registration()),
                        );
                    } else {
                        policy.register_worker(worker.as_ref(), Some(&info), Some(matches));
                    }
                };
            for worker in &workers {
                register(&policy, worker, matches.clone());
            }
            let weak_policy = Arc::downgrade(&policy);
            let weak_matches = Arc::downgrade(&matches);
            let registered = workers.clone();
            // Every query replaces both registrations, forcing selection to reach
            // the final, query-free attempt rather than reserve a stale snapshot.
            *matches.replace.lock() = Some(Box::new(move || {
                let policy = weak_policy.upgrade().unwrap();
                let matches = weak_matches.upgrade().unwrap();
                for worker in &registered {
                    register(&policy, worker, matches.clone());
                }
            }));
            for (case, lengths, excluded, chosen, expected, queries) in cases {
                let case = format!("{case} (history={history_enabled})");
                let inputs: WorkerTokenInputs = workers
                    .iter()
                    .zip(lengths)
                    .filter_map(|(worker, lengths)| {
                        lengths.map(|lengths| {
                            (
                                worker.base_url().to_owned(),
                                Arc::from(
                                    lengths
                                        .iter()
                                        .map(|&length| vec![7; length])
                                        .collect::<Vec<_>>(),
                                ),
                            )
                        })
                    })
                    .collect();
                let excluded: Vec<_> = excluded
                    .map(|index: usize| workers[index].url().to_owned())
                    .into_iter()
                    .collect();
                let recorder =
                    metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
                let queries_before = matches.queries.load(Ordering::Relaxed);
                let receipt = metrics::with_local_recorder(&recorder, || {
                    let tokens = KvRequestTokens::PerWorker(&inputs);
                    if history_enabled {
                        let request = policy
                            .capture_completion_history(&workers, "model")
                            .unwrap();
                        assert_eq!(request.registrations.len(), workers.len(), "{case}");
                        let (receipt, observation) = policy
                            .select_with_history(
                                &workers,
                                tokens,
                                None,
                                Some(&request),
                                SelectionIdentity::default(),
                                &excluded,
                            )
                            .unwrap();
                        assert!(observation.is_none(), "{case}");
                        receipt
                    } else {
                        policy
                            .select_worker_with_inputs(
                                &workers, tokens, None, None, None, &excluded,
                            )
                            .unwrap()
                    }
                });
                assert_eq!(
                    matches.queries.load(Ordering::Relaxed) - queries_before,
                    queries,
                    "{case}",
                );
                if let Some(chosen) = chosen {
                    assert_eq!(receipt.worker_index, chosen, "{case}");
                }
                let worker = &workers[receipt.worker_index];
                let target = &receipt.decision.target;
                let reserved = policy.load.sample(target);
                assert_eq!(
                    receipt.new_prefill_tokens,
                    expected,
                    "{case}: {} (cost {}, reserved {reserved:?})",
                    worker.url(),
                    receipt.decision.cost,
                );
                assert_eq!(receipt.decision.components.ptoken, expected, "{case}");
                assert_eq!(receipt.decision.cost, expected as f64, "{case}");
                assert_eq!(
                    reserved,
                    LoadSample {
                        active_prefill_tokens: expected,
                        active_requests: 1,
                    },
                    "{case}",
                );
                assert_eq!(worker.load(), 1, "{case}");
                release_receipt(&workers, &receipt);
                assert_eq!(policy.load.sample(target), LoadSample::default(), "{case}");
                assert_eq!(worker.load(), 0, "{case}");

                let rendered = recorder.handle().render();
                let mut outcomes = rendered
                    .lines()
                    .filter(|line| line.starts_with("vllm_router_prefix_routing_decisions_total{"));
                let reason = if queries == 0 {
                    "missing_tokens"
                } else {
                    "registration_churn"
                };
                assert!(
                    outcomes.next().is_some_and(|line| {
                        line.contains(&format!("outcome=\"{reason}\"")) && line.ends_with(" 1")
                    }) && outcomes.next().is_none(),
                    "{case}: {rendered}",
                );
                if history_enabled && queries != 0 {
                    assert!(
                        !rendered
                            .lines()
                            .any(|line| line.starts_with("vllm_router_history_decisions_total{")),
                        "{case}: churn fallback recorded a history decision: {rendered}",
                    );
                }
            }
        }
    }

    #[test]
    fn registration_uses_the_discovered_cache_block_size_and_provider() {
        #[derive(Default)]
        struct QueryBlocks(
            AtomicUsize,
            Mutex<Option<Box<dyn FnMut() + Send>>>,
            AtomicUsize,
        );
        impl TieredMatchProvider for QueryBlocks {
            fn matches(
                &self,
                _: &[u32],
                block_size: u32,
                _: Option<&str>,
                _: u32,
            ) -> Vec<TierMatch> {
                self.0.store(block_size as usize, Ordering::Relaxed);
                self.2.fetch_add(1, Ordering::Relaxed);
                if let Some(mutate) = self.1.lock().as_mut() {
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
        assert_eq!(matches.0.load(Ordering::Relaxed), 2);

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
        // Every placement query changes the target generation. Selection must
        // terminate after three queries with full-input work and no KV credit.
        let weak_policy = Arc::downgrade(&policy);
        let weak_matches = Arc::downgrade(&matches);
        let churn_worker = workers[0].clone();
        let churn_info = info.clone();
        *matches.1.lock() = Some(Box::new(move || {
            weak_policy.upgrade().unwrap().register_worker(
                churn_worker.as_ref(),
                Some(&churn_info),
                Some(weak_matches.upgrade().unwrap()),
            );
        }));
        policy.register_worker(workers[0].as_ref(), Some(&info), Some(matches.clone()));
        let queries_before = matches.2.load(Ordering::Relaxed);
        let receipt = select().unwrap();
        assert_eq!(matches.2.load(Ordering::Relaxed) - queries_before, 3);
        assert_eq!(receipt.new_prefill_tokens, 5);
        release_receipt(&workers, &receipt);
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

    #[test]
    fn pick_decode_selects_minimum_load_and_rotates_ties() {
        let policy = KvAwarePolicy::default();
        assert_eq!(policy.pick_decode(&[4, 1], &[0, 0]), Some(1));
        assert!(policy.pick_decode(&[], &[]).is_none());
        let first = policy.pick_decode(&[0, 0], &[0, 0]).unwrap();
        let second = policy.pick_decode(&[0, 0], &[0, 0]).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn decode_ties_prefer_the_candidate_that_reuses_more() {
        let policy = KvAwarePolicy::default();
        // Same projected load: the engine holding the prefix (occupancy 25K,
        // 5K to add) must beat the empty one (0 + 30K), and it must win every
        // time rather than on a rotation.
        for _ in 0..16 {
            assert_eq!(
                policy.pick_decode(&[30_000, 30_000], &[5_000, 30_000]),
                Some(0)
            );
        }
        // Load still decides first: a candidate with a lighter projection wins
        // even if it must add more.
        assert_eq!(
            policy.pick_decode(&[20_000, 30_000], &[20_000, 5_000]),
            Some(0)
        );
    }

    #[test]
    fn pick_prefill_spreads_exact_ties_and_keeps_the_ranking() {
        let policy = KvAwarePolicy::default();
        // Equal cost and occupancy: every tied candidate must be reachable.
        let mut seen = [false; 3];
        for _ in 0..64 {
            let index = policy
                .pick_prefill(&[8_000.0, 8_000.0, 12_000.0], &[0, 0, 0])
                .unwrap();
            assert!(index < 2, "a dearer candidate must never win a tie");
            seen[index] = true;
        }
        assert_eq!(seen, [true, true, false]);

        // A cost tie still resolves by occupancy, without introducing chance.
        for _ in 0..16 {
            assert_eq!(policy.pick_prefill(&[8_000.0, 8_000.0], &[1, 0]), Some(1));
        }

        assert_eq!(policy.pick_prefill(&[], &[]), None);
        assert_eq!(policy.pick_prefill(&[1.0], &[]), None);
        assert_eq!(policy.pick_prefill(&[f64::NAN], &[0]), None);
    }

    #[test]
    fn colocated_equal_costs_spread_across_idle_workers() {
        let policy = KvAwarePolicy::default();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new("http://cold0".into(), WorkerType::Regular)),
            Arc::new(BasicWorker::new("http://cold1".into(), WorkerType::Regular)),
        ];
        policy.register_target(workers[0].url(), target("cold0", 0));
        policy.register_target(workers[1].url(), target("cold1", 0));

        let ids: Vec<u32> = (0..32).collect();
        let mut seen = [false; 2];
        for _ in 0..32 {
            let receipt = policy
                .select_worker_with_receipt(&workers, Some(&ids), None, None, None, &[])
                .unwrap();
            seen[receipt.worker_index] = true;
            release_receipt(&workers, &receipt);
        }
        assert_eq!(seen, [true, true], "equal costs must use every worker");
    }

    fn stage_policy() -> KvAwarePolicy {
        KvAwarePolicy::default().with_shared_affinity(Arc::new(KvAffinityTable::with_ttl(Some(
            Duration::from_secs(1800),
        ))))
    }

    #[test]
    fn stage_binding_stays_within_fair_share() {
        let policy = stage_policy();
        let key = affinity_key("s1");
        let workers = decode_pool(&["http://d0", "http://d1"]);

        let first = policy
            .select_decode(&workers, Some(&key), None, None, None)
            .unwrap();
        assert_eq!((first.index, first.outcome), (0, "bind"));
        assert!(first.binding.is_some());

        // Equal counts preserve locality even if the picker prefers d1.
        workers[0].increment_load();
        workers[1].increment_load();
        let hit = policy
            .select_decode(&workers, Some(&key), None, None, None)
            .unwrap();
        assert_eq!((hit.index, hit.outcome), (0, "sticky_hit"));
        first.binding.unwrap().release_ref();
        hit.binding.unwrap().release_ref();
    }

    #[test]
    fn stage_binding_rebinds_above_fair_share_and_releases_old_references() {
        let policy = stage_policy();
        let key = affinity_key("s1");
        let workers = decode_pool(&["http://d0", "http://d1", "http://d2"]);
        let first = policy
            .select_decode(&workers, Some(&key), None, None, None)
            .unwrap();
        let original = first.binding.unwrap();
        workers[0].increment_load();
        let moved = policy
            .select_decode(&workers, Some(&key), None, None, None)
            .unwrap();
        assert_eq!((moved.index, moved.outcome), (2, "rebind"));
        let replacement = moved.binding.unwrap();
        assert_eq!(original.in_flight(), 1);
        assert_eq!(replacement.in_flight(), 1);
        original.release_ref();
        assert_eq!(original.in_flight(), 0);
        assert_eq!(replacement.in_flight(), 1);
        replacement.release_ref();
        let hit = policy
            .select_decode(&workers, Some(&key), None, None, None)
            .unwrap();
        assert_eq!((hit.index, hit.outcome), (2, "sticky_hit"));
        hit.binding.unwrap().release_ref();
    }

    #[test]
    fn decode_without_ttl_uses_minimum_occupancy_without_binding() {
        let policy = KvAwarePolicy::default();
        let workers = decode_pool(&["http://d0", "http://d1"]);
        workers[0].increment_load();
        let pick = policy
            .select_decode(&workers, Some(&affinity_key("s1")), None, None, None)
            .unwrap();
        assert_eq!(pick.index, 1);
        assert!(pick.binding.is_none());
        assert!(policy.affinity.is_empty());
    }

    #[test]
    fn stage_binding_expires_after_last_reference_is_released() {
        let policy = KvAwarePolicy::default()
            .with_shared_affinity(Arc::new(KvAffinityTable::with_ttl(Some(Duration::ZERO))));
        let workers = decode_pool(&["http://d0", "http://d1"]);
        let key = affinity_key("s1");
        let first = policy
            .select_decode(&workers, Some(&key), None, None, None)
            .unwrap();
        let hit = policy
            .select_decode(&workers, Some(&key), None, None, None)
            .unwrap();
        assert_eq!(hit.outcome, "sticky_hit");
        first.binding.unwrap().release_ref();
        hit.binding.unwrap().release_ref();
        let expired = policy
            .select_decode(&workers, Some(&key), None, None, None)
            .unwrap();
        assert_eq!((expired.index, expired.outcome), (1, "bind"));
        expired.binding.unwrap().release_ref();
    }

    #[test]
    fn stage_binding_accepts_a_single_busy_worker_and_empty_pools_fail_cleanly() {
        let policy = stage_policy();
        let key = affinity_key("s1");
        let workers = decode_pool(&["http://d0"]);
        workers[0].increment_load();
        let pick = policy
            .select_decode(&workers, Some(&key), None, None, None)
            .unwrap();
        assert_eq!(pick.index, 0);
        pick.binding.unwrap().release_ref();
        assert!(policy
            .select_decode(&[], Some(&key), None, None, None)
            .is_none());
    }

    #[test]
    fn stage_binding_reselects_on_removal_and_re_registration() {
        let policy = stage_policy();
        let key = affinity_key("s1");
        let workers = decode_pool(&["http://d0", "http://d1"]);
        let pick = policy
            .select_decode(&workers, Some(&key), None, None, None)
            .unwrap();
        assert_eq!(pick.outcome, "bind");

        // The pool shrank to d1: the stale binding must rebind, not fail.
        let shrunk = decode_pool(&["http://d1"]);
        policy.unregister_target("http://d0");
        assert!(!policy.targets.contains_key("http://d0"));
        let rebound = policy
            .select_decode(&shrunk, Some(&key), None, None, None)
            .unwrap();
        assert_eq!((rebound.index, rebound.outcome), (0, "bind"));

        // A re-registered target reports a new registration epoch: the old
        // binding is stale and the stage reselects by its own score.
        let moved = decode_pool(&["http://d0", "http://d1"]);
        policy.register_target("http://d1", target("http://d1", 0));
        moved[0].increment_load();
        let rebind = policy
            .select_decode(&moved, Some(&key), None, None, None)
            .unwrap();
        assert_eq!((rebind.index, rebind.outcome), (1, "bind"));
        policy.unregister_target("http://d0");
        policy.unregister_target("http://d1");
        assert!(policy.targets.is_empty());
    }

    #[test]
    fn stage_binding_without_session_selects_per_request() {
        let policy = KvAwarePolicy::default();
        let workers = decode_pool(&["http://d0", "http://d1"]);
        workers[0].increment_load();
        let pick = policy
            .select_decode(&workers, None, None, None, None)
            .unwrap();
        assert_eq!((pick.index, pick.outcome), (1, "no_session"));
        assert!(pick.binding.is_none());
        assert!(policy.affinity.is_empty());
    }

    fn decode_pool(urls: &[&str]) -> Vec<Arc<dyn Worker>> {
        urls.iter()
            .map(|url| {
                Arc::new(BasicWorker::new((*url).to_owned(), WorkerType::Regular))
                    as Arc<dyn Worker>
            })
            .collect()
    }

    /// A Decode pool whose engines publish KV events: each worker gets a cache
    /// identity, and `held` gives the root-aligned device blocks it already has.
    fn decode_pool_with_blocks(
        policy: &KvAwarePolicy,
        urls: &[&str],
        held: &[u32],
        block_size: u32,
    ) -> Vec<Arc<dyn Worker>> {
        let workers = decode_pool(urls);
        let provider = Arc::new(StaticMatches {
            blocks: held.to_vec(),
            targets: workers
                .iter()
                .map(|worker| target(worker.url(), 0))
                .collect(),
        });
        let key = CacheKey {
            model: Arc::from("decode-pool"),
            block_size,
        };
        for worker in &workers {
            policy.register_target_with_cache_identity(
                worker.url(),
                target(worker.url(), 0),
                key.clone(),
                provider.clone(),
            );
        }
        workers
    }

    /// Answers every placement query with the same per-target intervals, which
    /// is what a fixed set of held blocks looks like to the cost model.
    struct StaticMatches {
        blocks: Vec<u32>,
        targets: Vec<RoutableTarget>,
    }

    impl TieredMatchProvider for StaticMatches {
        fn matches(&self, _: &[u32], _: u32, _: Option<&str>, _: u32) -> Vec<TierMatch> {
            self.targets
                .iter()
                .zip(&self.blocks)
                .map(|(target, blocks)| {
                    let segments = if *blocks == 0 {
                        Vec::new()
                    } else {
                        vec![(0, *blocks)]
                    };
                    tier_match(target, StorageTier::Device, &segments)
                })
                .collect()
        }
    }

    #[test]
    fn decode_credit_charges_only_the_blocks_a_candidate_does_not_hold() {
        // Two decoders at equal occupancy. `d0` holds half the prompt's blocks,
        // `d1` holds none, so only d0 can resume the prefix: it must win even
        // though plain occupancy is a tie.
        let block_size = 4u32;
        let policy = KvAwarePolicy::default();
        let workers =
            decode_pool_with_blocks(&policy, &["http://d0", "http://d1"], &[4, 0], block_size);
        let ids: Vec<u32> = (0..8 * block_size).collect();
        let prompts: Arc<[Vec<u32>]> = Arc::from(vec![ids]);
        let context: HashMap<String, u64> = workers
            .iter()
            .map(|worker| {
                let estimate = kv_aware_ops(&policy).unwrap().estimate_decode_work(
                    &[(worker.as_ref(), &prompts[..], true)],
                    &mut PromptHashes::new(),
                );
                (
                    worker.url().to_owned(),
                    estimate[0].additional_tokens.expect("verified identity"),
                )
            })
            .collect();
        assert_eq!(context["http://d0"], 16, "half the prompt is reusable");
        assert_eq!(context["http://d1"], 32, "a cold decoder pays the prompt");

        let pick = policy
            .select_decode(&workers, None, None, None, Some(&context))
            .unwrap();
        assert_eq!(pick.index, 0, "the holder must be preferred");

        // Without the credit both are tied at zero occupancy, so the tie cursor
        // rotates between them instead of always taking the holder.
        let first = policy
            .select_decode(&workers, None, None, None, None)
            .unwrap();
        let second = policy
            .select_decode(&workers, None, None, None, None)
            .unwrap();
        assert_ne!(first.index, second.index);
    }

    #[test]
    fn decode_partial_context_table_falls_back_to_occupancy() {
        // A table that omits a candidate would read that absence as zero, making
        // the unlisted worker look free. The policy must drop the whole table.
        let policy = KvAwarePolicy::default();
        let workers = decode_pool(&["http://d0", "http://d1"]);
        let partial: HashMap<String, u64> = HashMap::from([("http://d0".to_owned(), 100)]);
        let pick = policy
            .select_decode(&workers, None, None, None, Some(&partial))
            .unwrap();
        assert_eq!(
            pick.index, 0,
            "d1's missing entry is not a zero price, so idle d0 wins"
        );
    }

    #[test]
    fn decode_work_estimate_returns_nothing_without_a_verified_identity() {
        let block_size = 4u32;
        let policy = KvAwarePolicy::default();
        let workers = decode_pool(&["http://d0"]);
        let ids: Vec<u32> = (0..8 * block_size).collect();
        let prompts = vec![ids];
        let estimates = kv_aware_ops(&policy).unwrap().estimate_decode_work(
            &[(workers[0].as_ref(), prompts.as_slice(), false)],
            &mut PromptHashes::new(),
        );
        assert_eq!(estimates, vec![DecodeWorkEstimate::default()]);
    }
}
