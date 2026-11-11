//! Shadow-mode KV transfer planning for `kv_aware`.
//!
//! After a worker is picked, the planner asks whether fetching the cached
//! prefix from a busier peer with the same cache identity would have beaten
//! recomputing it. The answer is only reported: nothing moves and routing does
//! not change. The planner is a pure function of its input.
//!
//! Every peer's residency is measured from the destination's own prompt lookup
//! (`TransferPlannerInput::destination_hits`), so a peer's prefix always
//! describes tokens the destination prompt actually contains. A candidate's own
//! `matches` (computed from its own prompt) only describe the cost of routing
//! there.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use dashmap::mapref::entry::Entry;
use dashmap::DashMap;

use super::kv_aware::{request_prefill_tokens, CandidateView, CostParams};
use crate::kv_index::{CacheKey, Locality, RoutableTarget, StorageTier, TierMatch};

/// Opt-in planner configuration. `Default` is disabled. Cost constants are
/// placeholders until calibrated; in PR 1 they only shape shadow-mode plans and
/// never affect routing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransferPolicy {
    pub enabled: bool,
    /// Minimum blocks the source holds beyond the destination.
    pub min_prefix_blocks: u32,
    /// The source counts as busy when `active_requests >= this`.
    pub source_load_threshold: usize,
    /// When false, a shared cache identity is enough (shadow measurement only).
    pub require_capability: bool,
    pub prefill_us_per_token: f64,
    pub transfer_fixed_ms: f64,
    pub transfer_us_per_token: f64,
    /// Transfer must cost less than `margin` times the prefill it saves. In (0, 1].
    pub margin: f64,
}

impl Default for TransferPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            min_prefix_blocks: 64,
            source_load_threshold: 1,
            require_capability: true,
            prefill_us_per_token: 50.0,
            transfer_fixed_ms: 5.0,
            transfer_us_per_token: 5.0,
            margin: 0.8,
        }
    }
}

impl TransferPolicy {
    pub fn valid(&self) -> bool {
        [
            self.prefill_us_per_token,
            self.transfer_fixed_ms,
            self.transfer_us_per_token,
        ]
        .into_iter()
        .all(|value| value.is_finite() && value >= 0.0)
            && self.margin.is_finite()
            && self.margin > 0.0
            && self.margin <= 1.0
    }
}

/// What a worker can do for KV transfer. Declared per worker in router config
/// (`kv_transfer_capability`); engine-reported capability is later work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferCapability {
    pub can_export: bool,
    pub can_import: bool,
    /// Transport name, e.g. "nixl".
    pub connector: Arc<str>,
    pub connector_version: Arc<str>,
    /// Model, KV dtype, TP/PP shape, attention spec, LoRA/salt handling.
    pub layout_fingerprint: Arc<str>,
}

impl TransferCapability {
    /// True iff `self` can export, `destination` can import, and connector,
    /// version and layout fingerprint all match.
    pub fn can_send_to(&self, destination: &TransferCapability) -> bool {
        self.can_export
            && destination.can_import
            && self.connector == destination.connector
            && self.connector_version == destination.connector_version
            && self.layout_fingerprint == destination.layout_fingerprint
    }
}

/// Where an engine rank accepts peer KV fetches, as it reported in its
/// `peer_lookup` record. Shadow mode never uses it; a capability without an
/// address is shadow-only, and active mode will require the address on the
/// source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerEndpoint {
    pub engine_id: Arc<str>,
    pub host: Arc<str>,
    pub port: u16,
    pub tp_size: u32,
    pub dcp_size: u32,
    pub pp_size: u32,
}

/// Caps how many peer fetches may be in flight against one source at a time.
///
/// A fetch counts from the moment its [`PeerFetchPermit`] is granted until the
/// permit is released or dropped, so a request that errors out, is cancelled or
/// is dropped mid-stream cannot leak a slot.
#[derive(Debug)]
pub struct PeerFetchLimiter {
    max_inflight: usize,
    /// In-flight fetches by source worker URL; zero entries are removed.
    inflight: DashMap<Arc<str>, usize>,
}

impl PeerFetchLimiter {
    pub fn new(max_inflight_per_source: usize) -> Arc<Self> {
        Arc::new(Self {
            max_inflight: max_inflight_per_source,
            inflight: DashMap::new(),
        })
    }

    /// Take a slot against `source`, or `None` when it is at the cap.
    pub fn try_acquire(self: &Arc<Self>, source: &str) -> Option<PeerFetchPermit> {
        let source: Arc<str> = Arc::from(source);
        match self.inflight.entry(source.clone()) {
            Entry::Occupied(mut entry) => {
                if *entry.get() >= self.max_inflight {
                    return None;
                }
                *entry.get_mut() += 1;
            }
            Entry::Vacant(entry) => {
                if self.max_inflight == 0 {
                    return None;
                }
                entry.insert(1);
            }
        }
        Some(PeerFetchPermit {
            inner: Arc::new(PermitInner {
                limiter: self.clone(),
                source,
                released: AtomicBool::new(false),
            }),
        })
    }

    /// Fetches currently counted against `source`.
    pub fn inflight(&self, source: &str) -> usize {
        self.inflight.get(source).map_or(0, |count| *count)
    }

    fn release(&self, source: &Arc<str>) {
        if let Entry::Occupied(mut entry) = self.inflight.entry(source.clone()) {
            if *entry.get() <= 1 {
                entry.remove();
            } else {
                *entry.get_mut() -= 1;
            }
        }
    }
}

#[derive(Debug)]
struct PermitInner {
    limiter: Arc<PeerFetchLimiter>,
    source: Arc<str>,
    released: AtomicBool,
}

impl PermitInner {
    fn release(&self) {
        if !self.released.swap(true, Ordering::AcqRel) {
            self.limiter.release(&self.source);
        }
    }
}

impl Drop for PermitInner {
    fn drop(&mut self) {
        self.release();
    }
}

/// One in-flight slot against a source. Clones share the slot; it is freed by
/// [`release`](Self::release) or when the last clone drops, whichever is first.
#[derive(Debug, Clone)]
pub struct PeerFetchPermit {
    inner: Arc<PermitInner>,
}

impl PeerFetchPermit {
    /// Free the slot now. Idempotent.
    pub fn release(&self) {
        self.inner.release();
    }
}

/// Active mode's decision for one planned request: where the destination
/// should fetch from, and the in-flight slot that fetch holds.
#[derive(Debug, Clone)]
pub struct PeerFetchGrant {
    pub endpoint: PeerEndpoint,
    pub permit: PeerFetchPermit,
}

/// A hypothetical transfer the planner would have made. In PR 1 it is only
/// reported.
#[derive(Debug, Clone, PartialEq)]
pub struct TransferPlan {
    /// Assigned by the policy (monotonic counter); 0 inside the pure function.
    pub plan_id: u64,
    pub source: RoutableTarget,
    pub destination: RoutableTarget,
    /// `[start, end)` block range the destination lacks.
    pub blocks: (u32, u32),
    pub block_size: u32,
    /// Tokens the transfer would save the destination.
    pub credited_tokens: u64,
    pub estimated_transfer_ms: f64,
    pub estimated_saved_prefill_ms: f64,
    pub estimated_source_wait_ms: f64,
    pub estimated_destination_queue_ms: f64,
}

/// Why no plan was made. `label()` is a stable string for metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferSkip {
    Disabled,
    /// No other candidate in the destination's identity group.
    NoPeer,
    /// The destination's cache key is unknown.
    UnverifiedIdentity,
    /// The destination's own prompt lookup is unavailable (no tokens for it, a
    /// multi-prompt request, or no credit), so peer residency cannot be measured.
    UnknownDestinationInput,
    PickedHoldsLongestPrefix,
    PrefixTooShort,
    SourceNotBusy,
    SourceExcluded,
    Incompatible,
    CostRule,
    /// Used by the call site: the pick came from a session binding.
    SessionBinding,
    /// Used by the call site: the request has no token ids to match.
    MissingTokens,
}

impl TransferSkip {
    pub fn label(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::NoPeer => "no_peer",
            Self::UnverifiedIdentity => "unverified_identity",
            Self::UnknownDestinationInput => "unknown_destination_input",
            Self::PickedHoldsLongestPrefix => "picked_holds_longest_prefix",
            Self::PrefixTooShort => "prefix_too_short",
            Self::SourceNotBusy => "source_not_busy",
            Self::SourceExcluded => "source_excluded",
            Self::Incompatible => "incompatible",
            Self::CostRule => "cost_rule",
            Self::SessionBinding => "session_binding",
            Self::MissingTokens => "missing_tokens",
        }
    }
}

/// Everything the planner may read. All slices are aligned by candidate
/// position.
pub struct TransferPlannerInput<'a> {
    pub candidates: &'a [CandidateView<'a>],
    pub cache_keys: &'a [Option<&'a CacheKey>],
    pub block_sizes: &'a [u32],
    pub capabilities: &'a [Option<&'a TransferCapability>],
    pub excluded: &'a [bool],
    pub picked: usize,
    /// Unfiltered index hits (all targets) from the destination's own prompt
    /// lookup; every peer's residency is measured from these. `None` when that
    /// lookup is unavailable.
    pub destination_hits: Option<&'a [TierMatch]>,
    /// The cost parameters routing scored with (tier credits); the planner
    /// prices both the fetch and the wait path with them. `block_size` is
    /// taken from `block_sizes`.
    pub cost_params: &'a CostParams,
}

/// What the planner returns: the plan or skip, plus the cost estimate whenever
/// the cost rule was evaluated for the peer whose result is returned (the
/// planned peer, or the peer whose `CostRule` skip is reported). `None` for
/// skips that happen before the cost rule.
#[derive(Debug, Clone, PartialEq)]
pub struct TransferOutcome {
    pub result: Result<TransferPlan, TransferSkip>,
    pub estimate: Option<CostEstimate>,
}

impl TransferOutcome {
    pub(crate) fn skip(skip: TransferSkip) -> Self {
        Self {
            result: Err(skip),
            estimate: None,
        }
    }
}

pub trait TransferPlanner: Send + Sync + std::fmt::Debug {
    fn plan(&self, input: &TransferPlannerInput<'_>) -> TransferOutcome;
}

#[derive(Debug, Clone, Copy)]
pub struct CostRuleTransferPlanner {
    pub policy: TransferPolicy,
}

impl TransferPlanner for CostRuleTransferPlanner {
    fn plan(&self, input: &TransferPlannerInput<'_>) -> TransferOutcome {
        plan_transfer(input, &self.policy)
    }
}

/// Blocks a peer could export from block 0: the contiguous `Device` run,
/// continued by `HostPinned` segments that start at or before the current end.
/// `Disk`, `External` and segments not reachable from block 0 are ignored.
fn transferable_prefix_blocks(matches: &[TierMatch]) -> u32 {
    prefix_blocks(matches.iter())
}

fn prefix_blocks<'a>(matches: impl Iterator<Item = &'a TierMatch> + Clone) -> u32 {
    let mut end = matches
        .clone()
        .filter(|hit| hit.tier == StorageTier::Device)
        .flat_map(|hit| hit.segments.iter().copied())
        .filter(|(start, _)| *start == 0)
        .map(|(_, end)| end)
        .max()
        .unwrap_or(0);
    loop {
        let next = matches
            .clone()
            .filter(|hit| hit.tier == StorageTier::HostPinned)
            .flat_map(|hit| hit.segments.iter().copied())
            .filter(|&(start, seg_end)| start <= end && seg_end > end)
            .map(|(_, seg_end)| seg_end)
            .max();
        match next {
            Some(seg_end) => end = seg_end,
            None => return end,
        }
    }
}

/// Pure: no locks, no I/O, no clock. Deterministic for the same input.
pub fn plan_transfer(input: &TransferPlannerInput<'_>, policy: &TransferPolicy) -> TransferOutcome {
    if !policy.enabled || !policy.valid() {
        return TransferOutcome::skip(TransferSkip::Disabled);
    }
    // Misaligned input is a caller bug; refuse to plan rather than panic on
    // the routing path.
    let count = input.candidates.len();
    if input.picked >= count
        || input.cache_keys.len() != count
        || input.block_sizes.len() != count
        || input.capabilities.len() != count
        || input.excluded.len() != count
    {
        return TransferOutcome::skip(TransferSkip::Disabled);
    }
    let dest = input.picked;
    let Some(dest_key) = input.cache_keys[dest] else {
        return TransferOutcome::skip(TransferSkip::UnverifiedIdentity);
    };
    let block_size = input.block_sizes[dest];
    if block_size == 0 {
        return TransferOutcome::skip(TransferSkip::UnverifiedIdentity);
    }
    let Some(destination_hits) = input.destination_hits else {
        return TransferOutcome::skip(TransferSkip::UnknownDestinationInput);
    };
    let dest_prefix = transferable_prefix_blocks(input.candidates[dest].matches);

    // Rank every peer that shares the destination's identity: largest prefix
    // of the destination's prompt first, then the less loaded peer, then the
    // lower position. Each one is then checked in that order and the first
    // that clears every check is planned, so a peer that cannot serve
    // (excluded, incompatible, idle, too expensive) never hides a shorter one
    // that can.
    let mut peers: Vec<(usize, u32)> = (0..count)
        .filter(|&pos| {
            pos != dest
                && input.cache_keys[pos] == Some(dest_key)
                && input.block_sizes[pos] == block_size
        })
        .map(|pos| {
            let peer_target = input.candidates[pos].target;
            let prefix = prefix_blocks(
                destination_hits
                    .iter()
                    .filter(|hit| hit.target == *peer_target),
            );
            (pos, prefix)
        })
        .collect();
    peers.sort_by_key(|&(pos, prefix)| {
        (
            std::cmp::Reverse(prefix),
            input.candidates[pos].load.active_requests,
            pos,
        )
    });

    // When no peer qualifies, report the best-ranked peer's reason, preferring
    // one that could have served at all: a retry exclusion or a layout
    // mismatch on the longest holder says less about lost transfers than why
    // the next usable holder was rejected.
    let mut first_skip: Option<(TransferSkip, Option<CostEstimate>)> = None;
    let mut first_usable_skip: Option<(TransferSkip, Option<CostEstimate>)> = None;
    for (src, src_prefix) in peers {
        let outcome = plan_from_source(input, policy, src, src_prefix, dest_prefix, block_size);
        match outcome.result {
            Ok(_) => return outcome,
            Err(skip) => {
                first_skip.get_or_insert((skip, outcome.estimate));
                if !matches!(
                    skip,
                    TransferSkip::SourceExcluded | TransferSkip::Incompatible
                ) {
                    first_usable_skip.get_or_insert((skip, outcome.estimate));
                }
            }
        }
    }
    match first_usable_skip.or(first_skip) {
        Some((skip, estimate)) => TransferOutcome {
            result: Err(skip),
            estimate,
        },
        None => TransferOutcome::skip(TransferSkip::NoPeer),
    }
}

/// Remaining prefill for `candidate` under the router's own cost model (the
/// same tier credits and last-token rule that scored it), in tokens.
fn routing_prefill_tokens(
    candidate: &CandidateView<'_>,
    cost_params: &CostParams,
    block_size: u32,
) -> u64 {
    let mut params = *cost_params;
    params.block_size = block_size;
    request_prefill_tokens(candidate, &params)
}

/// The two paths the planner compares, in milliseconds, plus what the fetch
/// saves. Both paths price the remaining prefill with the router's estimate
/// so a plan never disagrees with the cost model that made the pick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostEstimate {
    pub transfer_ms: f64,
    /// Destination queue + transfer + destination prefill after the fetch.
    pub fetch_ms: f64,
    /// Source queue + the source's normal prefill (all its tier credits).
    pub wait_ms: f64,
    /// Destination prefill without the fetch minus with it.
    pub saved_ms: f64,
    pub dest_queue_ms: f64,
    pub source_wait_ms: f64,
}

fn estimate_costs(
    input: &TransferPlannerInput<'_>,
    policy: &TransferPolicy,
    src: usize,
    fetched_end: u32,
    credited: u64,
    block_size: u32,
) -> CostEstimate {
    let (source, destination) = (&input.candidates[src], &input.candidates[input.picked]);
    let prefill_ms = |tokens: u64| tokens as f64 * policy.prefill_us_per_token / 1000.0;
    let transfer_ms =
        policy.transfer_fixed_ms + credited as f64 * policy.transfer_us_per_token / 1000.0;
    let dest_queue_ms = prefill_ms(destination.load.active_prefill_tokens);
    let source_wait_ms = prefill_ms(source.load.active_prefill_tokens);

    // After the fetch the destination holds `[0, fetched_end)` on device, on
    // top of whatever else it already had.
    let mut fetched_matches = destination.matches.to_vec();
    fetched_matches.push(TierMatch {
        target: destination.target.clone(),
        tier: StorageTier::Device,
        segments: vec![(0, fetched_end)],
        locality: Locality::Local,
    });
    let after_fetch = CandidateView {
        matches: &fetched_matches,
        ..*destination
    };
    let rem_dest_before = routing_prefill_tokens(destination, input.cost_params, block_size);
    let rem_dest_after = routing_prefill_tokens(&after_fetch, input.cost_params, block_size);
    let rem_src = routing_prefill_tokens(source, input.cost_params, block_size);

    CostEstimate {
        transfer_ms,
        fetch_ms: dest_queue_ms + transfer_ms + prefill_ms(rem_dest_after),
        wait_ms: source_wait_ms + prefill_ms(rem_src),
        saved_ms: prefill_ms(rem_dest_before.saturating_sub(rem_dest_after)),
        dest_queue_ms,
        source_wait_ms,
    }
}

/// Every check for one candidate source, in order: can it serve at all, does
/// it add enough, is it busy, and does fetching from it beat waiting on it.
fn plan_from_source(
    input: &TransferPlannerInput<'_>,
    policy: &TransferPolicy,
    src: usize,
    src_prefix: u32,
    dest_prefix: u32,
    block_size: u32,
) -> TransferOutcome {
    let dest = input.picked;
    if input.excluded[src] {
        return TransferOutcome::skip(TransferSkip::SourceExcluded);
    }
    if policy.require_capability {
        match (input.capabilities[src], input.capabilities[dest]) {
            (Some(from), Some(to)) if from.can_send_to(to) => {}
            _ => return TransferOutcome::skip(TransferSkip::Incompatible),
        }
    }
    if src_prefix <= dest_prefix {
        return TransferOutcome::skip(TransferSkip::PickedHoldsLongestPrefix);
    }
    let missing_blocks = src_prefix - dest_prefix;
    if missing_blocks < policy.min_prefix_blocks {
        return TransferOutcome::skip(TransferSkip::PrefixTooShort);
    }
    let (source, destination) = (&input.candidates[src], &input.candidates[dest]);
    if source.load.active_requests < policy.source_load_threshold {
        return TransferOutcome::skip(TransferSkip::SourceNotBusy);
    }

    // Never credit the final prompt token, matching `credited_tokens` in
    // kv_aware.rs.
    let bs = u64::from(block_size);
    let dest_cached = u64::from(dest_prefix) * bs;
    let credited = (u64::from(missing_blocks) * bs).min(
        (destination.input_tokens as u64)
            .saturating_sub(1)
            .saturating_sub(dest_cached),
    );
    if credited == 0 {
        return TransferOutcome::skip(TransferSkip::PrefixTooShort);
    }
    let credited_blocks = credited.div_ceil(bs).min(u64::from(missing_blocks)) as u32;
    let fetched_end = dest_prefix + credited_blocks;

    let cost = estimate_costs(input, policy, src, fetched_end, credited, block_size);
    if !(cost.fetch_ms < cost.wait_ms && cost.transfer_ms < cost.saved_ms * policy.margin) {
        return TransferOutcome {
            result: Err(TransferSkip::CostRule),
            estimate: Some(cost),
        };
    }

    let plan = TransferPlan {
        plan_id: 0,
        source: source.target.clone(),
        destination: destination.target.clone(),
        blocks: (dest_prefix, fetched_end),
        block_size,
        credited_tokens: credited,
        estimated_transfer_ms: cost.transfer_ms,
        estimated_saved_prefill_ms: cost.saved_ms,
        estimated_source_wait_ms: cost.source_wait_ms,
        estimated_destination_queue_ms: cost.dest_queue_ms,
    };
    TransferOutcome {
        result: Ok(plan),
        estimate: Some(cost),
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::kv_index::HashMode;
    use crate::policies::kv_aware::{LoadSample, Scorer};

    fn target(name: &str) -> RoutableTarget {
        RoutableTarget {
            instance_id: Arc::from(name),
            dp_rank: 0,
        }
    }

    fn hit(t: &RoutableTarget, tier: StorageTier, segments: &[(u32, u32)]) -> TierMatch {
        TierMatch {
            target: t.clone(),
            tier,
            segments: segments.to_vec(),
            locality: Locality::Local,
        }
    }

    fn key(model: &str) -> CacheKey {
        CacheKey {
            model: Arc::from(model),
            hash_mode: HashMode::Sha256,
            block_size: 16,
        }
    }

    fn cap(export: bool, import: bool, fingerprint: &str) -> TransferCapability {
        TransferCapability {
            can_export: export,
            can_import: import,
            connector: Arc::from("nixl"),
            connector_version: Arc::from("1"),
            layout_fingerprint: Arc::from(fingerprint),
        }
    }

    /// Natural destination lookup: every candidate's matches together.
    fn concat(matches: &[Vec<TierMatch>]) -> Vec<TierMatch> {
        matches.iter().flatten().cloned().collect()
    }

    fn enabled() -> TransferPolicy {
        TransferPolicy {
            enabled: true,
            require_capability: false,
            ..TransferPolicy::default()
        }
    }

    fn view<'a>(
        t: &'a RoutableTarget,
        matches: &'a [TierMatch],
        active_requests: usize,
        active_prefill_tokens: u64,
    ) -> CandidateView<'a> {
        CandidateView {
            target: t,
            matches,
            load: LoadSample {
                active_prefill_tokens,
                active_requests,
            },
            input_tokens: 2000,
            prefill_tokens: None,
        }
    }

    /// Two candidates: position 0 is the idle destination (picked) holding
    /// `dest_blocks` device blocks, position 1 a busy source holding `src_matches`.
    struct Pair {
        targets: [RoutableTarget; 2],
        matches: [Vec<TierMatch>; 2],
        load: [(usize, u64); 2],
        keys: [Option<CacheKey>; 2],
        block_sizes: [u32; 2],
        caps: [Option<TransferCapability>; 2],
        excluded: [bool; 2],
        /// Overrides the default destination lookup (all matches concatenated).
        dest_hits: Option<Option<Vec<TierMatch>>>,
    }

    impl Pair {
        fn new(dest_blocks: u32, src_blocks: u32) -> Self {
            let targets = [target("dest"), target("src")];
            let dest = if dest_blocks > 0 {
                vec![hit(&targets[0], StorageTier::Device, &[(0, dest_blocks)])]
            } else {
                vec![]
            };
            let src = vec![hit(&targets[1], StorageTier::Device, &[(0, src_blocks)])];
            Self {
                targets,
                matches: [dest, src],
                load: [(0, 0), (4, 10_000)],
                keys: [Some(key("m")), Some(key("m"))],
                block_sizes: [16, 16],
                caps: [None, None],
                excluded: [false, false],
                dest_hits: None,
            }
        }

        fn plan(&self, policy: &TransferPolicy) -> Result<TransferPlan, TransferSkip> {
            self.outcome(policy).result
        }

        fn outcome(&self, policy: &TransferPolicy) -> TransferOutcome {
            let candidates = [
                view(
                    &self.targets[0],
                    &self.matches[0],
                    self.load[0].0,
                    self.load[0].1,
                ),
                view(
                    &self.targets[1],
                    &self.matches[1],
                    self.load[1].0,
                    self.load[1].1,
                ),
            ];
            let keys = [self.keys[0].as_ref(), self.keys[1].as_ref()];
            let caps = [self.caps[0].as_ref(), self.caps[1].as_ref()];
            let hits = match &self.dest_hits {
                Some(over) => over.clone(),
                None => Some(concat(&self.matches)),
            };
            plan_transfer(
                &TransferPlannerInput {
                    candidates: &candidates,
                    cache_keys: &keys,
                    block_sizes: &self.block_sizes,
                    capabilities: &caps,
                    excluded: &self.excluded,
                    picked: 0,
                    destination_hits: hits.as_deref(),
                    cost_params: &CostParams::default(),
                },
                policy,
            )
        }
    }

    #[test]
    fn own_prompt_prefix_is_not_credited_when_destination_lookup_shows_little() {
        // The source holds 100 blocks of its OWN prompt, but the destination's
        // lookup shows it holds only 10 blocks of the destination prompt.
        let mut pair = Pair::new(0, 100);
        pair.dest_hits = Some(Some(vec![hit(
            &pair.targets[1],
            StorageTier::Device,
            &[(0, 10)],
        )]));
        assert_eq!(pair.plan(&enabled()), Err(TransferSkip::PrefixTooShort));

        // Nothing at all on the destination prompt: the picked worker is as good.
        pair.dest_hits = Some(Some(vec![]));
        assert_eq!(
            pair.plan(&enabled()),
            Err(TransferSkip::PickedHoldsLongestPrefix)
        );
    }

    #[test]
    fn partial_overlap_bounds_the_credited_blocks() {
        let mut pair = Pair::new(0, 100);
        pair.dest_hits = Some(Some(vec![hit(
            &pair.targets[1],
            StorageTier::Device,
            &[(0, 70)],
        )]));
        let plan = pair.plan(&enabled()).unwrap();
        assert_eq!(plan.blocks, (0, 70));
        assert_eq!(plan.credited_tokens, 70 * 16);
    }

    #[test]
    fn unknown_destination_input_is_skipped() {
        let mut pair = Pair::new(0, 100);
        pair.dest_hits = Some(None);
        assert_eq!(
            pair.plan(&enabled()),
            Err(TransferSkip::UnknownDestinationInput)
        );
        // Identity and policy checks still come first.
        pair.keys[0] = None;
        assert_eq!(pair.plan(&enabled()), Err(TransferSkip::UnverifiedIdentity));
        assert_eq!(
            pair.plan(&TransferPolicy::default()),
            Err(TransferSkip::Disabled)
        );
    }

    #[test]
    fn default_policy_is_disabled_and_valid() {
        let policy = TransferPolicy::default();
        assert!(!policy.enabled);
        assert!(policy.valid());
        assert_eq!(Pair::new(0, 100).plan(&policy), Err(TransferSkip::Disabled));
    }

    #[test]
    fn misaligned_input_and_zero_block_size_refuse_without_panicking() {
        let pair = Pair::new(0, 100);
        let candidates = [view(&pair.targets[0], &pair.matches[0], 0, 0)];
        let keys = [pair.keys[0].as_ref()];
        let caps = [None];
        let input = |picked, block_sizes: &[u32], excluded: &[bool]| {
            plan_transfer(
                &TransferPlannerInput {
                    candidates: &candidates,
                    cache_keys: &keys,
                    block_sizes,
                    capabilities: &caps,
                    excluded,
                    picked,
                    destination_hits: Some(&[]),
                    cost_params: &CostParams::default(),
                },
                &enabled(),
            )
            .result
        };
        assert_eq!(input(1, &[16], &[false]), Err(TransferSkip::Disabled));
        assert_eq!(input(0, &[16, 16], &[false]), Err(TransferSkip::Disabled));
        assert_eq!(input(0, &[16], &[]), Err(TransferSkip::Disabled));

        let mut zero = Pair::new(0, 100);
        zero.block_sizes = [0, 0];
        assert_eq!(zero.plan(&enabled()), Err(TransferSkip::UnverifiedIdentity));
    }

    #[test]
    fn invalid_policy_is_treated_as_disabled() {
        let pair = Pair::new(0, 100);
        for bad in [
            TransferPolicy {
                margin: 0.0,
                ..enabled()
            },
            TransferPolicy {
                margin: 1.5,
                ..enabled()
            },
            TransferPolicy {
                margin: f64::NAN,
                ..enabled()
            },
            TransferPolicy {
                prefill_us_per_token: -1.0,
                ..enabled()
            },
            TransferPolicy {
                transfer_fixed_ms: f64::INFINITY,
                ..enabled()
            },
            TransferPolicy {
                transfer_us_per_token: f64::NAN,
                ..enabled()
            },
        ] {
            assert!(!bad.valid());
            assert_eq!(pair.plan(&bad), Err(TransferSkip::Disabled));
        }
        assert!(TransferPolicy {
            margin: 1.0,
            ..enabled()
        }
        .valid());
    }

    #[test]
    fn unverified_destination_is_skipped() {
        let mut pair = Pair::new(0, 100);
        pair.keys[0] = None;
        assert_eq!(pair.plan(&enabled()), Err(TransferSkip::UnverifiedIdentity));
    }

    #[test]
    fn peers_need_same_cache_key_and_block_size() {
        let mut pair = Pair::new(0, 100);
        pair.keys[1] = Some(key("other"));
        assert_eq!(pair.plan(&enabled()), Err(TransferSkip::NoPeer));

        let mut pair = Pair::new(0, 100);
        pair.keys[1] = None;
        assert_eq!(pair.plan(&enabled()), Err(TransferSkip::NoPeer));

        let mut pair = Pair::new(0, 100);
        pair.block_sizes[1] = 32;
        assert_eq!(pair.plan(&enabled()), Err(TransferSkip::NoPeer));
    }

    #[test]
    fn picked_holding_longest_prefix_is_skipped() {
        assert_eq!(
            Pair::new(100, 100).plan(&enabled()),
            Err(TransferSkip::PickedHoldsLongestPrefix)
        );
        assert_eq!(
            Pair::new(120, 100).plan(&enabled()),
            Err(TransferSkip::PickedHoldsLongestPrefix)
        );
    }

    #[test]
    fn prefix_helper_counts_device_run_and_host_continuation() {
        let t = target("w");
        assert_eq!(transferable_prefix_blocks(&[]), 0);
        assert_eq!(
            transferable_prefix_blocks(&[hit(&t, StorageTier::Device, &[(0, 5)])]),
            5
        );
        // Host-pinned continuation counted.
        assert_eq!(
            transferable_prefix_blocks(&[
                hit(&t, StorageTier::Device, &[(0, 5)]),
                hit(&t, StorageTier::HostPinned, &[(3, 9)]),
            ]),
            9
        );
        // Chain of host segments, each starting at or before the previous end.
        assert_eq!(
            transferable_prefix_blocks(&[
                hit(&t, StorageTier::Device, &[(0, 5)]),
                hit(&t, StorageTier::HostPinned, &[(5, 8), (8, 12)]),
            ]),
            12
        );
        // Host-only prefix from block 0 counts.
        assert_eq!(
            transferable_prefix_blocks(&[hit(&t, StorageTier::HostPinned, &[(0, 4)])]),
            4
        );
    }

    #[test]
    fn prefix_helper_ignores_disk_external_and_gaps() {
        let t = target("w");
        assert_eq!(
            transferable_prefix_blocks(&[
                hit(&t, StorageTier::Device, &[(0, 5)]),
                hit(&t, StorageTier::Disk, &[(5, 20)]),
                hit(&t, StorageTier::External, &[(5, 30)]),
            ]),
            5
        );
        // A gap stops the chain, even if a later segment exists.
        assert_eq!(
            transferable_prefix_blocks(&[
                hit(&t, StorageTier::Device, &[(0, 5)]),
                hit(&t, StorageTier::HostPinned, &[(6, 20)]),
            ]),
            5
        );
        // Segments not reachable from block 0 are ignored.
        assert_eq!(
            transferable_prefix_blocks(&[hit(&t, StorageTier::Device, &[(2, 9)])]),
            0
        );
    }

    #[test]
    fn host_pinned_continuation_feeds_the_plan() {
        let mut pair = Pair::new(0, 40);
        pair.matches[1].push(hit(&pair.targets[1], StorageTier::HostPinned, &[(40, 100)]));
        let plan = pair.plan(&enabled()).unwrap();
        assert_eq!(plan.blocks, (0, 100));
    }

    #[test]
    fn short_prefix_is_skipped() {
        assert_eq!(
            Pair::new(0, 63).plan(&enabled()),
            Err(TransferSkip::PrefixTooShort)
        );
        // Counted beyond the destination's own prefix.
        assert_eq!(
            Pair::new(50, 100).plan(&enabled()),
            Err(TransferSkip::PrefixTooShort)
        );
        assert!(Pair::new(0, 64).plan(&enabled()).is_ok());
    }

    #[test]
    fn excluded_source_is_skipped() {
        let mut pair = Pair::new(0, 100);
        pair.excluded[1] = true;
        assert_eq!(pair.plan(&enabled()), Err(TransferSkip::SourceExcluded));
    }

    #[test]
    fn idle_source_is_skipped() {
        let mut pair = Pair::new(0, 100);
        pair.load[1] = (0, 0);
        assert_eq!(pair.plan(&enabled()), Err(TransferSkip::SourceNotBusy));
        let policy = TransferPolicy {
            source_load_threshold: 5,
            ..enabled()
        };
        assert_eq!(
            Pair::new(0, 100).plan(&policy),
            Err(TransferSkip::SourceNotBusy)
        );
    }

    #[test]
    fn capability_is_required_by_default() {
        let policy = TransferPolicy {
            require_capability: true,
            ..enabled()
        };
        let mut pair = Pair::new(0, 100);
        assert_eq!(pair.plan(&policy), Err(TransferSkip::Incompatible));

        pair.caps = [None, Some(cap(true, true, "a"))];
        assert_eq!(pair.plan(&policy), Err(TransferSkip::Incompatible));

        pair.caps = [Some(cap(true, true, "a")), Some(cap(true, true, "b"))];
        assert_eq!(pair.plan(&policy), Err(TransferSkip::Incompatible));

        // Source cannot export / destination cannot import.
        pair.caps = [Some(cap(true, true, "a")), Some(cap(false, true, "a"))];
        assert_eq!(pair.plan(&policy), Err(TransferSkip::Incompatible));
        pair.caps = [Some(cap(true, false, "a")), Some(cap(true, true, "a"))];
        assert_eq!(pair.plan(&policy), Err(TransferSkip::Incompatible));

        pair.caps = [Some(cap(false, true, "a")), Some(cap(true, false, "a"))];
        assert!(pair.plan(&policy).is_ok());
    }

    #[test]
    fn capability_connector_and_version_must_match() {
        let a = cap(true, true, "a");
        let mut other_connector = a.clone();
        other_connector.connector = Arc::from("mooncake");
        let mut other_version = a.clone();
        other_version.connector_version = Arc::from("2");
        assert!(a.can_send_to(&a.clone()));
        assert!(!a.can_send_to(&other_connector));
        assert!(!a.can_send_to(&other_version));
    }

    #[test]
    fn identity_alone_plans_when_capability_not_required() {
        let pair = Pair::new(0, 100);
        assert!(pair.plan(&enabled()).is_ok());
    }

    #[test]
    fn final_prompt_token_is_never_credited() {
        // 125 blocks * 16 = 2000 tokens; the source holds the whole prompt.
        let plan = Pair::new(0, 125).plan(&enabled()).unwrap();
        assert_eq!(plan.credited_tokens, 1999);
        assert_eq!(plan.blocks, (0, 125));

        // Destination already holds all but the last token's worth: nothing to credit.
        let mut pair = Pair::new(124, 125);
        pair.matches[1] = vec![hit(&pair.targets[1], StorageTier::Device, &[(0, 125)])];
        let policy = TransferPolicy {
            min_prefix_blocks: 1,
            // Fixed cost would swamp a 15-token transfer.
            transfer_fixed_ms: 0.0,
            ..enabled()
        };
        // 124 * 16 = 1984 cached. 15 tokens would move, but the router never
        // credits a device block that ends at the prompt boundary, so the
        // destination still prefills 16 tokens either way: the fetch saves
        // nothing.
        assert_eq!(pair.plan(&policy), Err(TransferSkip::CostRule));
    }

    #[test]
    fn zero_creditable_tokens_is_prefix_too_short() {
        // A one-token prompt leaves nothing to credit once the last token is held back.
        let pair = Pair::new(0, 2);
        let policy = TransferPolicy {
            min_prefix_blocks: 1,
            ..enabled()
        };
        let candidates = [
            CandidateView {
                input_tokens: 1,
                ..view(&pair.targets[0], &pair.matches[0], 0, 0)
            },
            CandidateView {
                input_tokens: 1,
                ..view(&pair.targets[1], &pair.matches[1], 4, 0)
            },
        ];
        let keys = [pair.keys[0].as_ref(), pair.keys[1].as_ref()];
        let result = plan_transfer(
            &TransferPlannerInput {
                candidates: &candidates,
                cache_keys: &keys,
                block_sizes: &pair.block_sizes,
                capabilities: &[None, None],
                excluded: &pair.excluded,
                picked: 0,
                destination_hits: Some(&concat(&pair.matches)),
                cost_params: &CostParams::default(),
            },
            &policy,
        )
        .result;
        assert_eq!(result, Err(TransferSkip::PrefixTooShort));
    }

    #[test]
    fn fast_recompute_rejects_by_cost_rule() {
        let policy = TransferPolicy {
            prefill_us_per_token: 1.0,
            ..enabled()
        };
        assert_eq!(Pair::new(0, 100).plan(&policy), Err(TransferSkip::CostRule));
    }

    #[test]
    fn margin_rejects_when_transfer_is_barely_cheaper() {
        // prefill(1600) = 80ms; transfer = 13ms. A 0.1 margin demands < 8ms.
        let policy = TransferPolicy {
            margin: 0.1,
            ..enabled()
        };
        assert_eq!(Pair::new(0, 100).plan(&policy), Err(TransferSkip::CostRule));
    }

    #[test]
    fn busy_source_makes_the_transfer_worthwhile() {
        // Same prefix, but the source queue is the thing that is slow.
        let mut pair = Pair::new(0, 100);
        pair.load[1] = (4, 100_000);
        assert!(pair.plan(&enabled()).is_ok());
        // With a short source queue, recomputing on the destination is not
        // worse than waiting, but the transfer rule still gates it.
        pair.load[1] = (1, 0);
        pair.load[0] = (0, 200_000);
        assert_eq!(pair.plan(&enabled()), Err(TransferSkip::CostRule));
    }

    #[test]
    fn equal_sources_tie_break_on_load_then_position() {
        let targets = [target("dest"), target("a"), target("b"), target("c")];
        let src = |i: usize| vec![hit(&targets[i], StorageTier::Device, &[(0, 100)])];
        let matches = [vec![], src(1), src(2), src(3)];
        let k = key("m");
        let run = |loads: [usize; 4]| {
            let candidates = [
                view(&targets[0], &matches[0], loads[0], 0),
                view(&targets[1], &matches[1], loads[1], 10_000),
                view(&targets[2], &matches[2], loads[2], 10_000),
                view(&targets[3], &matches[3], loads[3], 10_000),
            ];
            let keys = [Some(&k); 4];
            let caps = [None; 4];
            plan_transfer(
                &TransferPlannerInput {
                    candidates: &candidates,
                    cache_keys: &keys,
                    block_sizes: &[16; 4],
                    capabilities: &caps,
                    excluded: &[false; 4],
                    picked: 0,
                    destination_hits: Some(&concat(&matches)),
                    cost_params: &CostParams::default(),
                },
                &enabled(),
            )
            .result
            .unwrap()
            .source
        };
        // Equal load: lowest position.
        assert_eq!(run([0, 3, 3, 3]), targets[1]);
        // Lower load beats lower position.
        assert_eq!(run([0, 5, 3, 4]), targets[2]);
        assert_eq!(run([0, 5, 4, 3]), targets[3]);
        // Deterministic across repeated calls.
        assert_eq!(run([0, 3, 3, 3]), run([0, 3, 3, 3]));
    }

    #[test]
    fn longest_prefix_source_wins_over_less_loaded_shorter_one() {
        let targets = [target("dest"), target("short"), target("long")];
        let matches = [
            vec![],
            vec![hit(&targets[1], StorageTier::Device, &[(0, 80)])],
            vec![hit(&targets[2], StorageTier::Device, &[(0, 100)])],
        ];
        let k = key("m");
        let candidates = [
            view(&targets[0], &matches[0], 0, 0),
            view(&targets[1], &matches[1], 1, 10_000),
            view(&targets[2], &matches[2], 9, 10_000),
        ];
        let plan = plan_transfer(
            &TransferPlannerInput {
                candidates: &candidates,
                cache_keys: &[Some(&k); 3],
                block_sizes: &[16; 3],
                capabilities: &[None; 3],
                excluded: &[false; 3],
                picked: 0,
                destination_hits: Some(&concat(&matches)),
                cost_params: &CostParams::default(),
            },
            &enabled(),
        )
        .result
        .unwrap();
        assert_eq!(plan.source, targets[2]);
        assert_eq!(plan.blocks, (0, 100));
    }

    #[test]
    fn plan_fields_for_a_worked_example() {
        // bs 16, 2000-token prompts. Destination idle, no cache. Source holds
        // 100 device blocks, has 4 active requests and 10_000 queued tokens.
        //   credited      = min(100 * 16, 1999 - 0)  = 1600 tokens
        //   saved prefill = 1600 * 50us              = 80 ms
        //   transfer      = 5 + 1600 * 5us           = 13 ms
        //   dest queue    = 0 ms, source wait = 10_000 * 50us = 500 ms
        //   (a) 13 + 0 + prefill(400) = 33  <  500 + prefill(400) = 520
        //   (b) 13 < 80 * 0.8 = 64
        let pair = Pair::new(0, 100);
        let plan = pair.plan(&enabled()).unwrap();
        assert_eq!(
            plan,
            TransferPlan {
                plan_id: 0,
                source: target("src"),
                destination: target("dest"),
                blocks: (0, 100),
                block_size: 16,
                credited_tokens: 1600,
                estimated_transfer_ms: 13.0,
                estimated_saved_prefill_ms: 80.0,
                estimated_source_wait_ms: 500.0,
                estimated_destination_queue_ms: 0.0,
            }
        );
    }

    #[test]
    fn destination_partial_prefix_shifts_the_block_range() {
        // Destination holds 30 blocks, source 130 (capped by the prompt: 2000
        // tokens = 125 blocks, final token excluded).
        let plan = Pair::new(30, 130).plan(&enabled()).unwrap();
        // min(100 * 16, 1999 - 480) = 1519 tokens -> ceil(1519 / 16) = 95 blocks.
        assert_eq!(plan.credited_tokens, 1519);
        assert_eq!(plan.blocks, (30, 125));
    }

    #[test]
    fn cost_rule_planner_delegates_to_plan_transfer() {
        let pair = Pair::new(0, 100);
        let planner = CostRuleTransferPlanner { policy: enabled() };
        let candidates = [
            view(&pair.targets[0], &pair.matches[0], 0, 0),
            view(&pair.targets[1], &pair.matches[1], 4, 10_000),
        ];
        let keys = [pair.keys[0].as_ref(), pair.keys[1].as_ref()];
        let hits = concat(&pair.matches);
        let input = TransferPlannerInput {
            candidates: &candidates,
            cache_keys: &keys,
            block_sizes: &pair.block_sizes,
            capabilities: &[None, None],
            excluded: &pair.excluded,
            picked: 0,
            destination_hits: Some(&hits),
            cost_params: &CostParams::default(),
        };
        assert_eq!(planner.plan(&input), plan_transfer(&input, &enabled()));
        let off = CostRuleTransferPlanner {
            policy: TransferPolicy::default(),
        };
        assert_eq!(off.plan(&input).result, Err(TransferSkip::Disabled));
    }

    /// One candidate for [`plan_many`]: matches on its own prompt (also used as
    /// the destination lookup), load, retry exclusion and capability.
    struct Peer {
        target: RoutableTarget,
        matches: Vec<TierMatch>,
        load: (usize, u64),
        excluded: bool,
        cap: Option<TransferCapability>,
    }

    fn peer(name: &str, tiers: &[(StorageTier, (u32, u32))], load: (usize, u64)) -> Peer {
        let target = target(name);
        let matches = tiers
            .iter()
            .map(|&(tier, segment)| hit(&target, tier, &[segment]))
            .collect();
        Peer {
            target,
            matches,
            load,
            excluded: false,
            cap: None,
        }
    }

    /// Position 0 is the picked destination; the rest are peers.
    fn plan_many(peers: &[Peer], policy: &TransferPolicy) -> Result<TransferPlan, TransferSkip> {
        plan_many_outcome(peers, policy).result
    }

    fn plan_many_outcome(peers: &[Peer], policy: &TransferPolicy) -> TransferOutcome {
        let candidates: Vec<_> = peers
            .iter()
            .map(|p| view(&p.target, &p.matches, p.load.0, p.load.1))
            .collect();
        let k = key("m");
        let keys = vec![Some(&k); peers.len()];
        let caps: Vec<_> = peers.iter().map(|p| p.cap.as_ref()).collect();
        let excluded: Vec<_> = peers.iter().map(|p| p.excluded).collect();
        let hits: Vec<TierMatch> = peers.iter().flat_map(|p| p.matches.clone()).collect();
        plan_transfer(
            &TransferPlannerInput {
                candidates: &candidates,
                cache_keys: &keys,
                block_sizes: &vec![16; peers.len()],
                capabilities: &caps,
                excluded: &excluded,
                picked: 0,
                destination_hits: Some(&hits),
                cost_params: &CostParams::default(),
            },
            policy,
        )
    }

    const BUSY: (usize, u64) = (4, 10_000);

    #[test]
    fn excluded_longest_holder_does_not_mask_a_shorter_eligible_peer() {
        // Review example: an excluded peer holds 100 blocks, an eligible busy
        // peer 80. 80 clears the 64-block minimum and both cost checks.
        let mut long = peer("long", &[(StorageTier::Device, (0, 100))], BUSY);
        long.excluded = true;
        let peers = [
            peer("dest", &[], (0, 0)),
            long,
            peer("short", &[(StorageTier::Device, (0, 80))], BUSY),
        ];
        let plan = plan_many(&peers, &enabled()).unwrap();
        assert_eq!(plan.source, target("short"));
        assert_eq!(plan.blocks, (0, 80));
        assert_eq!(plan.credited_tokens, 80 * 16);
    }

    #[test]
    fn incompatible_longest_holder_does_not_mask_a_compatible_peer() {
        let policy = TransferPolicy {
            require_capability: true,
            ..enabled()
        };
        let mut dest = peer("dest", &[], (0, 0));
        dest.cap = Some(cap(true, true, "a"));
        let mut long = peer("long", &[(StorageTier::Device, (0, 100))], BUSY);
        long.cap = Some(cap(true, true, "other-layout"));
        let mut short = peer("short", &[(StorageTier::Device, (0, 80))], BUSY);
        short.cap = Some(cap(true, true, "a"));
        let plan = plan_many(&[dest, long, short], &policy).unwrap();
        assert_eq!(plan.source, target("short"));
    }

    #[test]
    fn next_peer_is_tried_when_the_longest_holder_fails_a_soft_check() {
        // The 100-block holder is busy by count but has nothing queued, so
        // waiting on it (20 ms) beats fetching from it (33 ms). The 80-block
        // holder has a long queue, so fetching from it wins.
        let peers = [
            peer("dest", &[], (0, 0)),
            peer("long", &[(StorageTier::Device, (0, 100))], (1, 0)),
            peer("short", &[(StorageTier::Device, (0, 80))], BUSY),
        ];
        assert_eq!(
            plan_many(&peers, &enabled()).unwrap().source,
            target("short")
        );

        // An idle longest holder does not hide a busy shorter one either.
        let peers = [
            peer("dest", &[], (0, 0)),
            peer("long", &[(StorageTier::Device, (0, 100))], (0, 0)),
            peer("short", &[(StorageTier::Device, (0, 80))], BUSY),
        ];
        assert_eq!(
            plan_many(&peers, &enabled()).unwrap().source,
            target("short")
        );
    }

    #[test]
    fn skip_reason_prefers_the_best_usable_peer() {
        // Longest holder excluded, the next one rejected on cost: the counter
        // says why the usable peer lost, not that a retry excluded one.
        let mut long = peer("long", &[(StorageTier::Device, (0, 100))], BUSY);
        long.excluded = true;
        let peers = [
            peer("dest", &[], (0, 0)),
            long,
            peer("short", &[(StorageTier::Device, (0, 80))], (1, 0)),
        ];
        assert_eq!(plan_many(&peers, &enabled()), Err(TransferSkip::CostRule));

        // Only unusable peers: their own reason is reported.
        let mut only = peer("long", &[(StorageTier::Device, (0, 100))], BUSY);
        only.excluded = true;
        let peers = [peer("dest", &[], (0, 0)), only];
        assert_eq!(
            plan_many(&peers, &enabled()),
            Err(TransferSkip::SourceExcluded)
        );
    }

    #[test]
    fn wait_on_source_uses_the_routing_prefill_estimate() {
        // Review example: 2000-token prompt, block size 16. The source holds
        // 100 device blocks plus 25 external blocks, has 280 queued tokens and
        // four active requests; the destination is idle and empty.
        let peers = [
            peer("dest", &[], (0, 0)),
            peer(
                "src",
                &[
                    (StorageTier::Device, (0, 100)),
                    (StorageTier::External, (100, 125)),
                ],
                (4, 280),
            ),
        ];
        let candidates: Vec<_> = peers
            .iter()
            .map(|p| view(&p.target, &p.matches, p.load.0, p.load.1))
            .collect();

        // Routing credits the external hits at 0.6: the source prefills
        // 2000 - (100 + 25 * 0.6) * 16 = 160 tokens, so its cost is
        // (160 + 280) * 5 = 2200 against the idle destination's 2000 * 1.
        let mut costs = [0.0; 2];
        crate::policies::kv_aware::MultiplicativeScorer
            .score(&candidates, &CostParams::default(), &mut costs)
            .unwrap();
        assert_eq!(costs, [2000.0, 2200.0]);

        let k = key("m");
        let hits: Vec<TierMatch> = peers.iter().flat_map(|p| p.matches.clone()).collect();
        let input = TransferPlannerInput {
            candidates: &candidates,
            cache_keys: &[Some(&k); 2],
            block_sizes: &[16; 2],
            capabilities: &[None; 2],
            excluded: &[false; 2],
            picked: 0,
            destination_hits: Some(&hits),
            cost_params: &CostParams::default(),
        };
        // fetch = 13 ms transfer + prefill(400) = 33 ms
        // wait  = prefill(280) + prefill(160)   = 22 ms (34 ms if the external
        //         hits were dropped, which reported a false transfer win)
        let cost = estimate_costs(&input, &enabled(), 1, 100, 1600, 16);
        assert_eq!(cost.fetch_ms, 33.0);
        assert_eq!(cost.wait_ms, 22.0);
        assert_eq!(
            plan_transfer(&input, &enabled()).result,
            Err(TransferSkip::CostRule)
        );
        let outcome = plan_transfer(&input, &enabled());
        let estimate = outcome.estimate.expect("cost rule evaluated");
        assert_eq!(estimate.fetch_ms, 33.0);
        assert_eq!(estimate.wait_ms, 22.0);
    }

    #[test]
    fn destination_external_credit_is_not_counted_as_saved() {
        // The destination already gets 0.6 credit for 125 external blocks, so
        // routing expects it to prefill 2000 - 75 * 16 = 800 tokens. After a
        // 100-block fetch it prefills 2000 - (100 + 25 * 0.6) * 16 = 160, so
        // the fetch saves 640 tokens (32 ms), not the 1600 it moves.
        let peers = [
            peer("dest", &[(StorageTier::External, (0, 125))], (0, 0)),
            peer("src", &[(StorageTier::Device, (0, 100))], BUSY),
        ];
        let plan = plan_many(&peers, &enabled()).unwrap();
        assert_eq!(plan.credited_tokens, 1600);
        assert_eq!(plan.estimated_saved_prefill_ms, 32.0);
    }

    #[test]
    fn skip_labels_are_stable_and_distinct() {
        let all = [
            TransferSkip::Disabled,
            TransferSkip::NoPeer,
            TransferSkip::UnverifiedIdentity,
            TransferSkip::UnknownDestinationInput,
            TransferSkip::PickedHoldsLongestPrefix,
            TransferSkip::PrefixTooShort,
            TransferSkip::SourceNotBusy,
            TransferSkip::SourceExcluded,
            TransferSkip::Incompatible,
            TransferSkip::CostRule,
            TransferSkip::SessionBinding,
            TransferSkip::MissingTokens,
        ];
        let labels: std::collections::HashSet<_> = all.iter().map(|s| s.label()).collect();
        assert_eq!(labels.len(), all.len());
        assert_eq!(TransferSkip::CostRule.label(), "cost_rule");
        assert_eq!(
            TransferSkip::UnknownDestinationInput.label(),
            "unknown_destination_input"
        );
    }

    #[test]
    fn estimate_is_present_on_a_planned_outcome() {
        // Idle empty destination, source with 100 blocks, 4 active requests and
        // 10_000 queued tokens; 2000-token prompt at 50 us/token.
        // credited = 1600; transfer = 5 + 1600 * 5 / 1000 = 13 ms;
        // fetch = 0 + 13 + prefill(400) = 33 ms;
        // wait = prefill(10_000) + prefill(400) = 520 ms; saved = prefill(1600) = 80 ms.
        let outcome = Pair::new(0, 100).outcome(&enabled());
        let plan = outcome.result.expect("planned");
        let estimate = outcome.estimate.expect("estimate on a plan");
        assert_eq!(estimate.fetch_ms, 33.0);
        assert_eq!(estimate.wait_ms, 520.0);
        assert_eq!(estimate.saved_ms, 80.0);
        assert_eq!(plan.estimated_saved_prefill_ms, estimate.saved_ms);
        assert_eq!(plan.estimated_transfer_ms, estimate.transfer_ms);
    }

    #[test]
    fn estimate_is_present_on_a_cost_rule_skip() {
        let policy = TransferPolicy {
            prefill_us_per_token: 1.0,
            ..enabled()
        };
        let outcome = Pair::new(0, 100).outcome(&policy);
        assert_eq!(outcome.result, Err(TransferSkip::CostRule));
        let estimate = outcome.estimate.expect("estimate on a cost_rule skip");
        // transfer 13 ms; prefill(400) = 0.4 ms; fetch = 13.4 ms; wait = 10.4 ms.
        assert!((estimate.fetch_ms - 13.4).abs() < 1e-9);
        assert!((estimate.wait_ms - 10.4).abs() < 1e-9);

        // The reported peer's estimate: the first usable skip is a cost rule
        // for the busy-but-expensive peer below.
        let peers = [
            peer("dest", &[], (0, 0)),
            peer("short", &[(StorageTier::Device, (0, 80))], (1, 0)),
        ];
        let outcome = plan_many_outcome(&peers, &enabled());
        assert_eq!(outcome.result, Err(TransferSkip::CostRule));
        assert!(outcome.estimate.is_some());
    }

    #[test]
    fn estimate_is_absent_for_skips_before_the_cost_rule() {
        assert_eq!(Pair::new(0, 63).outcome(&enabled()).estimate, None);
        assert_eq!(Pair::new(100, 100).outcome(&enabled()).estimate, None);
        let mut idle = Pair::new(0, 100);
        idle.load[1] = (0, 0);
        let outcome = idle.outcome(&enabled());
        assert_eq!(outcome.result, Err(TransferSkip::SourceNotBusy));
        assert_eq!(outcome.estimate, None);
        let mut excluded = Pair::new(0, 100);
        excluded.excluded[1] = true;
        assert_eq!(excluded.outcome(&enabled()).estimate, None);
        let outcome = Pair::new(0, 100).outcome(&TransferPolicy::default());
        assert_eq!(outcome.result, Err(TransferSkip::Disabled));
        assert_eq!(outcome.estimate, None);

        // Reported reason is a pre-cost-rule skip even though a later peer
        // reached the cost rule: the estimate belongs to the reported peer.
        let peers = [
            peer("dest", &[], (0, 0)),
            peer("longest", &[(StorageTier::Device, (0, 100))], (0, 0)),
            peer("short", &[(StorageTier::Device, (0, 80))], (1, 0)),
        ];
        let outcome = plan_many_outcome(&peers, &enabled());
        assert_eq!(outcome.result, Err(TransferSkip::SourceNotBusy));
        assert_eq!(outcome.estimate, None);
    }

    #[test]
    fn limiter_caps_per_source_and_frees_on_release_or_drop() {
        let limiter = PeerFetchLimiter::new(2);
        let first = limiter.try_acquire("a").expect("first");
        let second = limiter.try_acquire("a").expect("second");
        assert!(limiter.try_acquire("a").is_none(), "cap reached");
        assert_eq!(limiter.inflight("a"), 2);
        // Other sources are independent.
        let other = limiter.try_acquire("b").expect("other source");

        first.release();
        first.release(); // idempotent
        assert_eq!(limiter.inflight("a"), 1);
        let third = limiter.try_acquire("a").expect("slot freed by release");

        drop(second);
        assert_eq!(limiter.inflight("a"), 1);
        drop(third);
        assert_eq!(limiter.inflight("a"), 0);
        drop(other);
        assert!(limiter.inflight.is_empty(), "idle sources are forgotten");
        // `first` was released explicitly; dropping it must not free again.
        drop(first);
        assert_eq!(limiter.inflight("a"), 0);
    }

    #[test]
    fn permit_clones_share_one_slot() {
        let limiter = PeerFetchLimiter::new(1);
        let permit = limiter.try_acquire("a").unwrap();
        let clone = permit.clone();
        drop(permit);
        assert_eq!(limiter.inflight("a"), 1, "a clone still holds it");
        drop(clone);
        assert_eq!(limiter.inflight("a"), 0);
    }

    #[test]
    fn zero_cap_admits_nothing() {
        assert!(PeerFetchLimiter::new(0).try_acquire("a").is_none());
    }
}
