//! KV transfer planning for `kv_aware`.
//!
//! After a worker is picked, the planner asks whether fetching the cached
//! prefix from a busier peer with the same cache identity would beat
//! recomputing it. The plan never changes which worker is picked. In shadow
//! mode it is only counted; in active mode the policy turns it into a peer
//! fetch. The planner is a pure function of its input.
//!
//! Every peer's residency is measured from the destination's own prompt lookup
//! (`PeerFetchPlannerInput::destination_hits`), so a peer's prefix always
//! describes tokens the destination prompt actually contains. A candidate's own
//! `matches` (computed from its own prompt) only describe the cost of routing
//! there.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use dashmap::mapref::entry::Entry;
use dashmap::DashMap;

use super::kv_aware::{request_prefill_tokens_with_device_prefix, CandidateView, CostParams};
use crate::config::types::{peer_fetch_cost_constants_valid, PeerFetchConfig, PeerFetchMode};
use crate::kv_index::{CacheKey, PeerLookupRecord, RoutableTarget, StorageTier, TierMatch};

/// Opt-in planner configuration, built from `PeerFetchConfig` via `From`.
/// Cost constants are placeholders until calibrated; they shape which transfers
/// are planned and never affect which worker is picked.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PeerFetchRules {
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

impl PeerFetchRules {
    pub fn valid(&self) -> bool {
        peer_fetch_cost_constants_valid(
            self.margin,
            self.prefill_us_per_token,
            self.transfer_fixed_ms,
            self.transfer_us_per_token,
        )
    }
}

impl From<&PeerFetchConfig> for PeerFetchRules {
    /// The planner policy for a config block. `enabled` follows `mode`.
    fn from(config: &PeerFetchConfig) -> Self {
        PeerFetchRules {
            enabled: config.mode != PeerFetchMode::Off,
            min_prefix_blocks: config.min_prefix_blocks,
            source_load_threshold: config.source_load_threshold,
            // Active transfers must never run on a shared cache identity alone,
            // so only shadow can drop the check (validation also rejects
            // `ignore_capability` in active mode).
            require_capability: config.mode == PeerFetchMode::Active || !config.ignore_capability,
            prefill_us_per_token: config.prefill_us_per_token,
            transfer_fixed_ms: config.transfer_fixed_ms,
            transfer_us_per_token: config.transfer_us_per_token,
            margin: config.margin,
        }
    }
}

/// What a worker can do for KV transfer. Declared per worker in router config
/// or reported by the engine in its `peer_lookup` record.
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

    /// The capability an engine's `peer_lookup` record grants: export and
    /// import, with the compared strings trimmed exactly as a static
    /// declaration's are, so `"nixl "` and `"nixl"` match whichever side
    /// reported them.
    pub(crate) fn from_engine_record(record: &PeerLookupRecord) -> Self {
        TransferCapability {
            can_export: true,
            can_import: true,
            connector: record.connector.trim().into(),
            connector_version: record.connector_version.trim().into(),
            layout_fingerprint: record.layout_fingerprint.trim().into(),
        }
    }
}

/// Where an engine rank accepts peer KV fetches, as it reported in its
/// `peer_lookup` record. Active mode needs the source's address to fetch; a
/// capability without one can be planned but not fetched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerEndpoint {
    pub engine_id: Arc<str>,
    pub host: Arc<str>,
    pub port: u16,
    pub tp_size: u32,
    pub dcp_size: u32,
    pub pp_size: u32,
}

/// The address half of an engine's `peer_lookup` record. These strings are
/// sent back to the engine, so they are passed through untrimmed.
impl From<&PeerLookupRecord> for PeerEndpoint {
    fn from(record: &PeerLookupRecord) -> Self {
        PeerEndpoint {
            engine_id: Arc::from(record.engine_id.as_str()),
            host: Arc::from(record.side_channel_host.as_str()),
            port: record.side_channel_port,
            tp_size: record.tp_size,
            dcp_size: record.dcp_size,
            pp_size: record.pp_size,
        }
    }
}

/// Store the peer fetch kill switch and keep the
/// `vllm_router_peer_fetch_enabled` gauge and the warn log in step with it.
/// Every writer of the flag goes through here so the gauge always matches.
pub(crate) fn store_peer_fetch_enabled(flag: &AtomicBool, enabled: bool) {
    flag.store(enabled, Ordering::Release);
    crate::metrics::RouterMetrics::set_peer_fetch_enabled(enabled);
    tracing::warn!(
        enabled,
        "peer fetch {} for all active-mode kv_aware policies",
        if enabled { "enabled" } else { "disabled" }
    );
}

/// Caps how many peer fetches may be in flight against one source at a time.
///
/// A fetch counts from the moment its [`PeerFetchPermit`] is granted until the
/// permit is released or dropped, so a request that errors out, is cancelled or
/// is dropped mid-stream cannot leak a slot.
#[derive(Debug)]
pub(crate) struct PeerFetchLimiter {
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
            limiter: self.clone(),
            source,
            released: AtomicBool::new(false),
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

/// One in-flight slot against a source. It has a single owner and is freed by
/// [`release`](Self::release) or when it drops, whichever is first.
#[derive(Debug)]
pub(crate) struct PeerFetchPermit {
    limiter: Arc<PeerFetchLimiter>,
    source: Arc<str>,
    released: AtomicBool,
}

impl PeerFetchPermit {
    /// Free the slot now. Idempotent.
    pub fn release(&self) {
        if !self.released.swap(true, Ordering::AcqRel) {
            self.limiter.release(&self.source);
        }
    }
}

impl Drop for PeerFetchPermit {
    fn drop(&mut self) {
        self.release();
    }
}

/// Active mode's decision for one planned request: where the destination
/// should fetch from, and the in-flight slot that fetch holds.
#[derive(Debug)]
pub(crate) struct PeerFetchGrant {
    pub endpoint: PeerEndpoint,
    pub permit: PeerFetchPermit,
}

/// A transfer the planner chose for one request.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PeerFetchPlan {
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
pub(crate) enum PeerFetchSkip {
    Disabled,
    /// The picked position is outside the peer list: a caller bug.
    MisalignedInput,
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
    /// Used by the call site: the client asked for `prompt_logprobs`, so the
    /// engine computes every prompt token and reads no external KV.
    PromptLogprobs,
}

impl PeerFetchSkip {
    pub fn label(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::MisalignedInput => "misaligned_input",
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
            Self::PromptLogprobs => "prompt_logprobs",
        }
    }
}

/// One selection's planner outcome: the `outcome` label on
/// `vllm_router_peer_fetch_plans_total`. Every value of that label comes
/// from here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PeerFetchPlanOutcome {
    Planned,
    Skipped(PeerFetchSkip),
}

impl PeerFetchPlanOutcome {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Planned => "planned",
            Self::Skipped(skip) => skip.label(),
        }
    }

    pub(crate) fn record(self) {
        crate::metrics::RouterMetrics::record_peer_fetch_plan(self.label());
    }
}

/// Which estimate a `vllm_router_peer_fetch_estimated_cost_ms` sample is: the
/// `path` label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PeerFetchCostPath {
    /// The destination fetches the prefix from the source (`fetch_ms`).
    Fetch,
    /// The request waits on the source instead (`wait_ms`).
    Wait,
}

impl PeerFetchCostPath {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Fetch => "fetch",
            Self::Wait => "wait",
        }
    }

    pub(crate) fn record(self, cost_ms: f64) {
        crate::metrics::RouterMetrics::record_peer_fetch_estimated_cost_ms(self.label(), cost_ms);
    }
}

/// Active-mode outcome for a planned request: the `outcome` label on
/// `vllm_router_peer_fetch_requests_total`. The policy records the four
/// outcomes that stop a fetch grant; the HTTP send path records the rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PeerFetchRequestOutcome {
    /// The body carries the peer fetch params.
    Sent,
    /// The destination is a gRPC worker; only the HTTP path adds fetch params.
    GrpcDestination,
    /// The source or destination re-registered after planning.
    RegistrationChanged,
    /// The source has no fetch address.
    NoPeerEndpoint,
    /// The source is at its in-flight fetch cap.
    InflightCap,
    /// The client already sent `kv_transfer_params`.
    ClientParamsPresent,
    /// The client asked for prompt logprobs, so the engine reads no external KV.
    PromptLogprobs,
    /// The body is not a JSON object.
    InvalidBody,
}

impl PeerFetchRequestOutcome {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Sent => "sent",
            Self::GrpcDestination => "grpc_destination",
            Self::RegistrationChanged => "registration_changed",
            Self::NoPeerEndpoint => "no_peer_endpoint",
            Self::InflightCap => "inflight_cap",
            Self::ClientParamsPresent => "client_params_present",
            Self::PromptLogprobs => "prompt_logprobs",
            Self::InvalidBody => "invalid_body",
        }
    }

    pub(crate) fn record(self) {
        crate::metrics::RouterMetrics::record_peer_fetch_request(self.label());
    }
}

/// One routing candidate as the planner sees it.
pub(crate) struct PeerInput<'a> {
    pub candidate: &'a CandidateView<'a>,
    /// `None` when the worker's cache identity is unknown. The block size is
    /// read from here.
    pub cache_key: Option<&'a CacheKey>,
    pub capability: Option<&'a TransferCapability>,
    /// Barred as a source (e.g. a retry exclusion).
    pub excluded: bool,
}

/// Everything the planner may read.
pub(crate) struct PeerFetchPlannerInput<'a> {
    pub peers: &'a [PeerInput<'a>],
    /// Index into `peers` of the destination.
    pub picked: usize,
    /// Unfiltered index hits (all targets) from the destination's own prompt
    /// lookup; every peer's residency is measured from these. `None` when that
    /// lookup is unavailable.
    pub destination_hits: Option<&'a [TierMatch]>,
    /// The cost parameters routing scored with (tier credits); the planner
    /// prices both the fetch and the wait path with them. `block_size` is
    /// taken from the destination's cache key.
    pub cost_params: &'a CostParams,
}

/// What the planner returns: the plan or skip, plus the cost estimate whenever
/// the cost rule was evaluated for the peer whose result is returned (the
/// planned peer, or the peer whose `CostRule` skip is reported). `None` for
/// skips that happen before the cost rule.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PeerFetchOutcome {
    pub result: Result<PeerFetchPlan, PeerFetchSkip>,
    pub estimate: Option<CostEstimate>,
}

impl PeerFetchOutcome {
    pub(crate) fn skip(skip: PeerFetchSkip) -> Self {
        Self {
            result: Err(skip),
            estimate: None,
        }
    }
}

pub(crate) trait PeerFetchPlanner: Send + Sync + std::fmt::Debug {
    fn plan(&self, input: &PeerFetchPlannerInput<'_>) -> PeerFetchOutcome;
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CostRulePeerFetchPlanner {
    pub policy: PeerFetchRules,
}

impl PeerFetchPlanner for CostRulePeerFetchPlanner {
    fn plan(&self, input: &PeerFetchPlannerInput<'_>) -> PeerFetchOutcome {
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
pub(crate) fn plan_transfer(
    input: &PeerFetchPlannerInput<'_>,
    policy: &PeerFetchRules,
) -> PeerFetchOutcome {
    if !policy.enabled || !policy.valid() {
        return PeerFetchOutcome::skip(PeerFetchSkip::Disabled);
    }
    // A pick outside the list is a caller bug; refuse to plan rather than
    // panic on the routing path in release builds.
    let count = input.peers.len();
    debug_assert!(
        input.picked < count,
        "picked {} out of range for {count} peers",
        input.picked
    );
    if input.picked >= count {
        return PeerFetchOutcome::skip(PeerFetchSkip::MisalignedInput);
    }
    let dest = input.picked;
    let Some(dest_key) = input.peers[dest].cache_key else {
        return PeerFetchOutcome::skip(PeerFetchSkip::UnverifiedIdentity);
    };
    let block_size = dest_key.block_size;
    if block_size == 0 {
        return PeerFetchOutcome::skip(PeerFetchSkip::UnverifiedIdentity);
    }
    let Some(destination_hits) = input.destination_hits else {
        return PeerFetchOutcome::skip(PeerFetchSkip::UnknownDestinationInput);
    };
    let dest_prefix = transferable_prefix_blocks(input.peers[dest].candidate.matches);

    // Rank every peer that shares the destination's identity: largest prefix
    // of the destination's prompt first, then the less loaded peer, then the
    // lower position. Each one is then checked in that order and the first
    // that clears every check is planned, so a peer that cannot serve
    // (excluded, incompatible, idle, too expensive) never hides a shorter one
    // that can.
    let mut peers: Vec<(usize, u32)> = (0..count)
        .filter(|&pos| pos != dest && input.peers[pos].cache_key == Some(dest_key))
        .map(|pos| {
            let peer_target = input.peers[pos].candidate.target;
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
            input.peers[pos].candidate.load.active_requests,
            pos,
        )
    });

    // When no peer qualifies, report the best-ranked peer's reason, preferring
    // one that could have served at all: a retry exclusion or a layout
    // mismatch on the longest holder says less about lost transfers than why
    // the next usable holder was rejected.
    let mut first_skip: Option<(PeerFetchSkip, Option<CostEstimate>)> = None;
    let mut first_usable_skip: Option<(PeerFetchSkip, Option<CostEstimate>)> = None;
    for (src, src_prefix) in peers {
        let outcome = plan_from_source(input, policy, src, src_prefix, dest_prefix, block_size);
        match outcome.result {
            Ok(_) => return outcome,
            Err(skip) => {
                first_skip.get_or_insert((skip, outcome.estimate));
                if !matches!(
                    skip,
                    PeerFetchSkip::SourceExcluded | PeerFetchSkip::Incompatible
                ) {
                    first_usable_skip.get_or_insert((skip, outcome.estimate));
                }
            }
        }
    }
    match first_usable_skip.or(first_skip) {
        Some((skip, estimate)) => PeerFetchOutcome {
            result: Err(skip),
            estimate,
        },
        None => PeerFetchOutcome::skip(PeerFetchSkip::NoPeer),
    }
}

/// Remaining prefill for `candidate` under the router's own cost model (the
/// same tier credits and last-token rule that scored it), in tokens.
///
/// `extra_device_prefix` counts device blocks `[0, n)` on top of the
/// candidate's matches (0 = none), so "after a fetch" needs no copy of them.
fn routing_prefill_tokens(
    candidate: &CandidateView<'_>,
    cost_params: &CostParams,
    block_size: u32,
    extra_device_prefix: u32,
) -> u64 {
    let mut params = *cost_params;
    params.block_size = block_size;
    request_prefill_tokens_with_device_prefix(candidate, &params, extra_device_prefix)
}

/// The two paths the planner compares, in milliseconds, plus what the fetch
/// saves. Both paths price the remaining prefill with the router's estimate
/// so a plan never disagrees with the cost model that made the pick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CostEstimate {
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
    input: &PeerFetchPlannerInput<'_>,
    policy: &PeerFetchRules,
    src: usize,
    fetched_end: u32,
    credited: u64,
    block_size: u32,
) -> CostEstimate {
    let (source, destination) = (
        input.peers[src].candidate,
        input.peers[input.picked].candidate,
    );
    let prefill_ms = |tokens: u64| tokens as f64 * policy.prefill_us_per_token / 1000.0;
    let transfer_ms =
        policy.transfer_fixed_ms + credited as f64 * policy.transfer_us_per_token / 1000.0;
    let dest_queue_ms = prefill_ms(destination.load.active_prefill_tokens);
    let source_wait_ms = prefill_ms(source.load.active_prefill_tokens);

    // After the fetch the destination holds `[0, fetched_end)` on device, on
    // top of whatever else it already had.
    let rem_dest_before = routing_prefill_tokens(destination, input.cost_params, block_size, 0);
    let rem_dest_after =
        routing_prefill_tokens(destination, input.cost_params, block_size, fetched_end);
    let rem_src = routing_prefill_tokens(source, input.cost_params, block_size, 0);

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
    input: &PeerFetchPlannerInput<'_>,
    policy: &PeerFetchRules,
    src: usize,
    src_prefix: u32,
    dest_prefix: u32,
    block_size: u32,
) -> PeerFetchOutcome {
    let dest = input.picked;
    if input.peers[src].excluded {
        return PeerFetchOutcome::skip(PeerFetchSkip::SourceExcluded);
    }
    if policy.require_capability {
        match (input.peers[src].capability, input.peers[dest].capability) {
            (Some(from), Some(to)) if from.can_send_to(to) => {}
            _ => return PeerFetchOutcome::skip(PeerFetchSkip::Incompatible),
        }
    }
    if src_prefix <= dest_prefix {
        return PeerFetchOutcome::skip(PeerFetchSkip::PickedHoldsLongestPrefix);
    }
    let missing_blocks = src_prefix - dest_prefix;
    if missing_blocks < policy.min_prefix_blocks {
        return PeerFetchOutcome::skip(PeerFetchSkip::PrefixTooShort);
    }
    let (source, destination) = (input.peers[src].candidate, input.peers[dest].candidate);
    if source.load.active_requests < policy.source_load_threshold {
        return PeerFetchOutcome::skip(PeerFetchSkip::SourceNotBusy);
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
        return PeerFetchOutcome::skip(PeerFetchSkip::PrefixTooShort);
    }
    let credited_blocks = credited.div_ceil(bs).min(u64::from(missing_blocks)) as u32;
    let fetched_end = dest_prefix + credited_blocks;

    let cost = estimate_costs(input, policy, src, fetched_end, credited, block_size);
    if !(cost.fetch_ms < cost.wait_ms && cost.transfer_ms < cost.saved_ms * policy.margin) {
        return PeerFetchOutcome {
            result: Err(PeerFetchSkip::CostRule),
            estimate: Some(cost),
        };
    }

    let plan = PeerFetchPlan {
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
    PeerFetchOutcome {
        result: Ok(plan),
        estimate: Some(cost),
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::config::types::{PeerFetchConfig, PeerFetchMode};
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
        }
    }

    fn key(model: &str) -> CacheKey {
        CacheKey {
            model: Arc::from(model),
            block_size: 16,
        }
    }

    fn peer_inputs<'a>(
        candidates: &'a [CandidateView<'a>],
        keys: &[Option<&'a CacheKey>],
        caps: &[Option<&'a TransferCapability>],
        excluded: &[bool],
    ) -> Vec<PeerInput<'a>> {
        assert!(candidates.len() == keys.len() && keys.len() == caps.len());
        assert!(caps.len() == excluded.len());
        candidates
            .iter()
            .enumerate()
            .map(|(i, candidate)| PeerInput {
                candidate,
                cache_key: keys[i],
                capability: caps[i],
                excluded: excluded[i],
            })
            .collect()
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

    /// A policy as the config builds it, with the given mode.
    /// Most fixtures declare no capabilities, so shadow drops the check; tests
    /// of the check itself set `require_capability` on the rules.
    fn policy_for(mode: PeerFetchMode) -> PeerFetchRules {
        PeerFetchRules::from(&PeerFetchConfig {
            mode,
            ignore_capability: mode == PeerFetchMode::Shadow,
            ..Default::default()
        })
    }

    fn enabled() -> PeerFetchRules {
        policy_for(PeerFetchMode::Shadow)
    }

    fn disabled() -> PeerFetchRules {
        policy_for(PeerFetchMode::Off)
    }

    /// Both modes check capability by default, so shadow predicts active.
    /// `ignore_capability` drops the check in shadow only: active mode keeps
    /// it even if validation was bypassed.
    #[test]
    fn capability_is_checked_unless_shadow_ignores_it() {
        let rules = |mode, ignore_capability| {
            PeerFetchRules::from(&PeerFetchConfig {
                mode,
                ignore_capability,
                ..Default::default()
            })
            .require_capability
        };
        assert!(rules(PeerFetchMode::Shadow, false));
        assert!(rules(PeerFetchMode::Active, false));
        assert!(!rules(PeerFetchMode::Shadow, true));
        assert!(rules(PeerFetchMode::Active, true));
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
                caps: [None, None],
                excluded: [false, false],
                dest_hits: None,
            }
        }

        fn plan(&self, policy: &PeerFetchRules) -> Result<PeerFetchPlan, PeerFetchSkip> {
            self.outcome(policy).result
        }

        fn outcome(&self, policy: &PeerFetchRules) -> PeerFetchOutcome {
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
            let peers = peer_inputs(&candidates, &keys, &caps, &self.excluded);
            plan_transfer(
                &PeerFetchPlannerInput {
                    peers: &peers,
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
        assert_eq!(pair.plan(&enabled()), Err(PeerFetchSkip::PrefixTooShort));

        // Nothing at all on the destination prompt: the picked worker is as good.
        pair.dest_hits = Some(Some(vec![]));
        assert_eq!(
            pair.plan(&enabled()),
            Err(PeerFetchSkip::PickedHoldsLongestPrefix)
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
            Err(PeerFetchSkip::UnknownDestinationInput)
        );
        // Identity and policy checks still come first.
        pair.keys[0] = None;
        assert_eq!(
            pair.plan(&enabled()),
            Err(PeerFetchSkip::UnverifiedIdentity)
        );
        assert_eq!(pair.plan(&disabled()), Err(PeerFetchSkip::Disabled));
    }

    #[test]
    fn default_policy_is_disabled_and_valid() {
        let policy = disabled();
        assert!(!policy.enabled);
        assert!(policy.valid());
        assert_eq!(
            Pair::new(0, 100).plan(&policy),
            Err(PeerFetchSkip::Disabled)
        );
    }

    /// Planner input with one candidate (the destination) and the given pick.
    fn plan_single_peer(picked: usize) -> PeerFetchOutcome {
        let pair = Pair::new(0, 100);
        let candidates = [view(&pair.targets[0], &pair.matches[0], 0, 0)];
        let peers = peer_inputs(&candidates, &[pair.keys[0].as_ref()], &[None], &[false]);
        plan_transfer(
            &PeerFetchPlannerInput {
                peers: &peers,
                picked,
                destination_hits: Some(&[]),
                cost_params: &CostParams::default(),
            },
            &enabled(),
        )
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "out of range")]
    fn picked_out_of_range_is_a_caller_bug() {
        plan_single_peer(1);
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn picked_out_of_range_is_refused_in_release() {
        assert_eq!(
            plan_single_peer(1).result,
            Err(PeerFetchSkip::MisalignedInput)
        );
    }

    #[test]
    fn zero_block_size_is_unverified_identity() {
        let mut zero = Pair::new(0, 100);
        let zero_key = CacheKey {
            block_size: 0,
            ..key("m")
        };
        zero.keys = [Some(zero_key.clone()), Some(zero_key)];
        assert_eq!(
            zero.plan(&enabled()),
            Err(PeerFetchSkip::UnverifiedIdentity)
        );
    }

    #[test]
    fn misaligned_input_has_a_stable_label() {
        assert_eq!(PeerFetchSkip::MisalignedInput.label(), "misaligned_input");
    }

    #[test]
    fn invalid_policy_is_treated_as_disabled() {
        let pair = Pair::new(0, 100);
        for bad in [
            PeerFetchRules {
                margin: 0.0,
                ..enabled()
            },
            PeerFetchRules {
                margin: 1.5,
                ..enabled()
            },
            PeerFetchRules {
                margin: f64::NAN,
                ..enabled()
            },
            PeerFetchRules {
                prefill_us_per_token: -1.0,
                ..enabled()
            },
            PeerFetchRules {
                transfer_fixed_ms: f64::INFINITY,
                ..enabled()
            },
            PeerFetchRules {
                transfer_us_per_token: f64::NAN,
                ..enabled()
            },
        ] {
            assert!(!bad.valid());
            assert_eq!(pair.plan(&bad), Err(PeerFetchSkip::Disabled));
        }
        assert!(PeerFetchRules {
            margin: 1.0,
            ..enabled()
        }
        .valid());
    }

    #[test]
    fn peers_need_same_cache_key_and_block_size() {
        let mut pair = Pair::new(0, 100);
        pair.keys[1] = Some(key("other"));
        assert_eq!(pair.plan(&enabled()), Err(PeerFetchSkip::NoPeer));

        let mut pair = Pair::new(0, 100);
        pair.keys[1] = None;
        assert_eq!(pair.plan(&enabled()), Err(PeerFetchSkip::NoPeer));

        let mut pair = Pair::new(0, 100);
        pair.keys[1] = Some(CacheKey {
            block_size: 32,
            ..key("m")
        });
        assert_eq!(pair.plan(&enabled()), Err(PeerFetchSkip::NoPeer));
    }

    #[test]
    fn picked_holding_longest_prefix_is_skipped() {
        assert_eq!(
            Pair::new(100, 100).plan(&enabled()),
            Err(PeerFetchSkip::PickedHoldsLongestPrefix)
        );
        assert_eq!(
            Pair::new(120, 100).plan(&enabled()),
            Err(PeerFetchSkip::PickedHoldsLongestPrefix)
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
            Err(PeerFetchSkip::PrefixTooShort)
        );
        // Counted beyond the destination's own prefix.
        assert_eq!(
            Pair::new(50, 100).plan(&enabled()),
            Err(PeerFetchSkip::PrefixTooShort)
        );
        assert!(Pair::new(0, 64).plan(&enabled()).is_ok());
    }

    #[test]
    fn excluded_source_is_skipped() {
        let mut pair = Pair::new(0, 100);
        pair.excluded[1] = true;
        assert_eq!(pair.plan(&enabled()), Err(PeerFetchSkip::SourceExcluded));
    }

    #[test]
    fn idle_source_is_skipped() {
        let mut pair = Pair::new(0, 100);
        pair.load[1] = (0, 0);
        assert_eq!(pair.plan(&enabled()), Err(PeerFetchSkip::SourceNotBusy));
        assert_eq!(PeerFetchSkip::SourceNotBusy.label(), "source_not_busy");
        let policy = PeerFetchRules {
            source_load_threshold: 5,
            ..enabled()
        };
        assert_eq!(
            Pair::new(0, 100).plan(&policy),
            Err(PeerFetchSkip::SourceNotBusy)
        );
    }

    #[test]
    fn capability_is_required_by_default() {
        let policy = PeerFetchRules {
            require_capability: true,
            ..enabled()
        };
        let mut pair = Pair::new(0, 100);
        assert_eq!(pair.plan(&policy), Err(PeerFetchSkip::Incompatible));

        pair.caps = [None, Some(cap(true, true, "a"))];
        assert_eq!(pair.plan(&policy), Err(PeerFetchSkip::Incompatible));

        pair.caps = [Some(cap(true, true, "a")), Some(cap(true, true, "b"))];
        assert_eq!(pair.plan(&policy), Err(PeerFetchSkip::Incompatible));

        // Source cannot export / destination cannot import.
        pair.caps = [Some(cap(true, true, "a")), Some(cap(false, true, "a"))];
        assert_eq!(pair.plan(&policy), Err(PeerFetchSkip::Incompatible));
        pair.caps = [Some(cap(true, false, "a")), Some(cap(true, true, "a"))];
        assert_eq!(pair.plan(&policy), Err(PeerFetchSkip::Incompatible));

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
        let policy = PeerFetchRules {
            min_prefix_blocks: 1,
            // Fixed cost would swamp a 15-token transfer.
            transfer_fixed_ms: 0.0,
            ..enabled()
        };
        // 124 * 16 = 1984 cached. 15 tokens would move, but the router never
        // credits a device block that ends at the prompt boundary, so the
        // destination still prefills 16 tokens either way: the fetch saves
        // nothing.
        assert_eq!(pair.plan(&policy), Err(PeerFetchSkip::CostRule));
    }

    #[test]
    fn zero_creditable_tokens_is_prefix_too_short() {
        // A one-token prompt leaves nothing to credit once the last token is held back.
        let pair = Pair::new(0, 2);
        let policy = PeerFetchRules {
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
        let peers = peer_inputs(&candidates, &keys, &[None, None], &pair.excluded);
        let result = plan_transfer(
            &PeerFetchPlannerInput {
                peers: &peers,
                picked: 0,
                destination_hits: Some(&concat(&pair.matches)),
                cost_params: &CostParams::default(),
            },
            &policy,
        )
        .result;
        assert_eq!(result, Err(PeerFetchSkip::PrefixTooShort));
    }

    #[test]
    fn fast_recompute_rejects_by_cost_rule() {
        let policy = PeerFetchRules {
            prefill_us_per_token: 1.0,
            ..enabled()
        };
        assert_eq!(
            Pair::new(0, 100).plan(&policy),
            Err(PeerFetchSkip::CostRule)
        );
    }

    #[test]
    fn margin_rejects_when_transfer_is_barely_cheaper() {
        // prefill(1600) = 80ms; transfer = 13ms. A 0.1 margin demands < 8ms.
        let policy = PeerFetchRules {
            margin: 0.1,
            ..enabled()
        };
        assert_eq!(
            Pair::new(0, 100).plan(&policy),
            Err(PeerFetchSkip::CostRule)
        );
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
        assert_eq!(pair.plan(&enabled()), Err(PeerFetchSkip::CostRule));
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
            let peers = peer_inputs(&candidates, &keys, &caps, &[false; 4]);
            plan_transfer(
                &PeerFetchPlannerInput {
                    peers: &peers,
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
        let peers = peer_inputs(&candidates, &[Some(&k); 3], &[None; 3], &[false; 3]);
        let plan = plan_transfer(
            &PeerFetchPlannerInput {
                peers: &peers,
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
            PeerFetchPlan {
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
    fn plan_many(peers: &[Peer], policy: &PeerFetchRules) -> Result<PeerFetchPlan, PeerFetchSkip> {
        plan_many_outcome(peers, policy).result
    }

    fn plan_many_outcome(peers: &[Peer], policy: &PeerFetchRules) -> PeerFetchOutcome {
        let candidates: Vec<_> = peers
            .iter()
            .map(|p| view(&p.target, &p.matches, p.load.0, p.load.1))
            .collect();
        let k = key("m");
        let keys = vec![Some(&k); peers.len()];
        let caps: Vec<_> = peers.iter().map(|p| p.cap.as_ref()).collect();
        let excluded: Vec<_> = peers.iter().map(|p| p.excluded).collect();
        let hits: Vec<TierMatch> = peers.iter().flat_map(|p| p.matches.clone()).collect();
        let inputs = peer_inputs(&candidates, &keys, &caps, &excluded);
        plan_transfer(
            &PeerFetchPlannerInput {
                peers: &inputs,
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
        let policy = PeerFetchRules {
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
        assert_eq!(plan_many(&peers, &enabled()), Err(PeerFetchSkip::CostRule));

        // Only unusable peers: their own reason is reported.
        let mut only = peer("long", &[(StorageTier::Device, (0, 100))], BUSY);
        only.excluded = true;
        let peers = [peer("dest", &[], (0, 0)), only];
        assert_eq!(
            plan_many(&peers, &enabled()),
            Err(PeerFetchSkip::SourceExcluded)
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
        let peer_list = peer_inputs(&candidates, &[Some(&k); 2], &[None; 2], &[false; 2]);
        let input = PeerFetchPlannerInput {
            peers: &peer_list,
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
            Err(PeerFetchSkip::CostRule)
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
    fn engine_and_static_capabilities_trim_alike() {
        let record = |connector: &str| PeerLookupRecord {
            engine_id: " engine-a ".into(),
            side_channel_host: "10.0.0.1".into(),
            side_channel_port: 5600,
            tp_size: 1,
            dcp_size: 1,
            pp_size: 1,
            connector: connector.into(),
            connector_version: "1 ".into(),
            layout_fingerprint: " fp".into(),
        };
        let declared = |connector: &str| {
            TransferCapability::try_from(&crate::config::types::StaticTransferCapability {
                can_export: true,
                can_import: true,
                connector: connector.into(),
                connector_version: "1".into(),
                layout_fingerprint: "fp".into(),
            })
            .unwrap()
        };

        // A stray space matches in either direction.
        let engine = TransferCapability::from_engine_record(&record("nixl "));
        assert!(engine.can_send_to(&declared("nixl")));
        assert!(declared("nixl ").can_send_to(&engine));
        assert_eq!(engine, declared("nixl"));

        // The address is sent back to the engine as reported.
        assert_eq!(
            &*PeerEndpoint::from(&record("nixl")).engine_id,
            " engine-a "
        );
    }

    #[test]
    fn plan_outcome_labels_are_stable_and_distinct() {
        // Every `plans_total` outcome; dashboards and the docs tables match on
        // these strings.
        let labels: Vec<_> = [
            PeerFetchPlanOutcome::Planned,
            PeerFetchPlanOutcome::Skipped(PeerFetchSkip::Disabled),
            PeerFetchPlanOutcome::Skipped(PeerFetchSkip::MisalignedInput),
            PeerFetchPlanOutcome::Skipped(PeerFetchSkip::NoPeer),
            PeerFetchPlanOutcome::Skipped(PeerFetchSkip::UnverifiedIdentity),
            PeerFetchPlanOutcome::Skipped(PeerFetchSkip::UnknownDestinationInput),
            PeerFetchPlanOutcome::Skipped(PeerFetchSkip::PickedHoldsLongestPrefix),
            PeerFetchPlanOutcome::Skipped(PeerFetchSkip::PrefixTooShort),
            PeerFetchPlanOutcome::Skipped(PeerFetchSkip::SourceNotBusy),
            PeerFetchPlanOutcome::Skipped(PeerFetchSkip::SourceExcluded),
            PeerFetchPlanOutcome::Skipped(PeerFetchSkip::Incompatible),
            PeerFetchPlanOutcome::Skipped(PeerFetchSkip::CostRule),
            PeerFetchPlanOutcome::Skipped(PeerFetchSkip::SessionBinding),
            PeerFetchPlanOutcome::Skipped(PeerFetchSkip::MissingTokens),
            PeerFetchPlanOutcome::Skipped(PeerFetchSkip::PromptLogprobs),
        ]
        .into_iter()
        .map(PeerFetchPlanOutcome::label)
        .collect();
        assert_eq!(
            labels,
            [
                "planned",
                "disabled",
                "misaligned_input",
                "no_peer",
                "unverified_identity",
                "unknown_destination_input",
                "picked_holds_longest_prefix",
                "prefix_too_short",
                "source_not_busy",
                "source_excluded",
                "incompatible",
                "cost_rule",
                "session_binding",
                "missing_tokens",
                "prompt_logprobs",
            ]
        );
        let distinct: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(distinct.len(), labels.len());
    }

    #[test]
    fn cost_path_labels_are_stable() {
        assert_eq!(PeerFetchCostPath::Fetch.label(), "fetch");
        assert_eq!(PeerFetchCostPath::Wait.label(), "wait");
    }

    #[test]
    fn request_outcome_labels_are_stable_and_distinct() {
        // Dashboards and the docs table match on these strings.
        let labels: Vec<_> = [
            PeerFetchRequestOutcome::Sent,
            PeerFetchRequestOutcome::GrpcDestination,
            PeerFetchRequestOutcome::RegistrationChanged,
            PeerFetchRequestOutcome::NoPeerEndpoint,
            PeerFetchRequestOutcome::InflightCap,
            PeerFetchRequestOutcome::ClientParamsPresent,
            PeerFetchRequestOutcome::PromptLogprobs,
            PeerFetchRequestOutcome::InvalidBody,
        ]
        .into_iter()
        .map(PeerFetchRequestOutcome::label)
        .collect();
        assert_eq!(
            labels,
            [
                "sent",
                "grpc_destination",
                "registration_changed",
                "no_peer_endpoint",
                "inflight_cap",
                "client_params_present",
                "prompt_logprobs",
                "invalid_body",
            ]
        );
    }

    #[test]
    fn estimate_is_present_on_a_cost_rule_skip() {
        let policy = PeerFetchRules {
            prefill_us_per_token: 1.0,
            ..enabled()
        };
        let outcome = Pair::new(0, 100).outcome(&policy);
        assert_eq!(outcome.result, Err(PeerFetchSkip::CostRule));
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
        assert_eq!(outcome.result, Err(PeerFetchSkip::CostRule));
        assert!(outcome.estimate.is_some());
    }

    #[test]
    fn estimate_is_absent_for_skips_before_the_cost_rule() {
        assert_eq!(Pair::new(0, 63).outcome(&enabled()).estimate, None);
        assert_eq!(Pair::new(100, 100).outcome(&enabled()).estimate, None);
        let mut idle = Pair::new(0, 100);
        idle.load[1] = (0, 0);
        let outcome = idle.outcome(&enabled());
        assert_eq!(outcome.result, Err(PeerFetchSkip::SourceNotBusy));
        assert_eq!(outcome.estimate, None);
        let mut excluded = Pair::new(0, 100);
        excluded.excluded[1] = true;
        assert_eq!(excluded.outcome(&enabled()).estimate, None);
        let outcome = Pair::new(0, 100).outcome(&disabled());
        assert_eq!(outcome.result, Err(PeerFetchSkip::Disabled));
        assert_eq!(outcome.estimate, None);

        // Reported reason is a pre-cost-rule skip even though a later peer
        // reached the cost rule: the estimate belongs to the reported peer.
        let peers = [
            peer("dest", &[], (0, 0)),
            peer("longest", &[(StorageTier::Device, (0, 100))], (0, 0)),
            peer("short", &[(StorageTier::Device, (0, 80))], (1, 0)),
        ];
        let outcome = plan_many_outcome(&peers, &enabled());
        assert_eq!(outcome.result, Err(PeerFetchSkip::SourceNotBusy));
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
    fn zero_cap_admits_nothing() {
        assert!(PeerFetchLimiter::new(0).try_acquire("a").is_none());
    }

    #[test]
    fn limiter_is_atomic_under_contention() {
        use std::sync::atomic::AtomicUsize;
        use std::thread;
        use std::time::Duration;

        let limiter = PeerFetchLimiter::new(4);
        let holders = Arc::new(AtomicUsize::new(0)); // threads holding a permit
        let max_seen = Arc::new(AtomicUsize::new(0));
        thread::scope(|scope| {
            for _ in 0..16 {
                let (limiter, holders, max_seen) = (
                    Arc::clone(&limiter),
                    Arc::clone(&holders),
                    Arc::clone(&max_seen),
                );
                scope.spawn(move || {
                    for _ in 0..4_000 {
                        if let Some(permit) = limiter.try_acquire("a") {
                            let inside = holders.fetch_add(1, Ordering::SeqCst) + 1;
                            max_seen.fetch_max(inside, Ordering::SeqCst);
                            thread::sleep(Duration::from_micros(5));
                            holders.fetch_sub(1, Ordering::SeqCst);
                            drop(permit);
                        }
                    }
                });
            }
        });
        assert_eq!(limiter.inflight("a"), 0);
        assert!(max_seen.load(Ordering::SeqCst) <= 4);
    }
}
