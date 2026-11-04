//! Shadow-mode KV transfer planning for `kv_aware`.
//!
//! After a worker is picked, the planner asks whether fetching the cached
//! prefix from a busier peer with the same cache identity would have beaten
//! recomputing it. The answer is only reported: nothing moves and routing does
//! not change. The planner is a pure function of its input.

use std::sync::Arc;

use super::kv_aware::CandidateView;
use crate::kv_index::{CacheKey, RoutableTarget, StorageTier, TierMatch};

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

/// What a worker declares it can do for KV transfer. No production caller
/// registers these yet (the handshake is later work).
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
}

pub trait TransferPlanner: Send + Sync + std::fmt::Debug {
    fn plan(&self, input: &TransferPlannerInput<'_>) -> Result<TransferPlan, TransferSkip>;
}

#[derive(Debug, Clone, Copy)]
pub struct CostRuleTransferPlanner {
    pub policy: TransferPolicy,
}

impl TransferPlanner for CostRuleTransferPlanner {
    fn plan(&self, input: &TransferPlannerInput<'_>) -> Result<TransferPlan, TransferSkip> {
        plan_transfer(input, &self.policy)
    }
}

/// Blocks a peer could export from block 0: the contiguous `Device` run,
/// continued by `HostPinned` segments that start at or before the current end.
/// `Disk`, `External` and segments not reachable from block 0 are ignored.
fn transferable_prefix_blocks(matches: &[TierMatch]) -> u32 {
    let mut end = matches
        .iter()
        .filter(|hit| hit.tier == StorageTier::Device)
        .flat_map(|hit| hit.segments.iter().copied())
        .filter(|(start, _)| *start == 0)
        .map(|(_, end)| end)
        .max()
        .unwrap_or(0);
    loop {
        let next = matches
            .iter()
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
pub fn plan_transfer(
    input: &TransferPlannerInput<'_>,
    policy: &TransferPolicy,
) -> Result<TransferPlan, TransferSkip> {
    if !policy.enabled || !policy.valid() {
        return Err(TransferSkip::Disabled);
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
        return Err(TransferSkip::Disabled);
    }
    let dest = input.picked;
    let Some(dest_key) = input.cache_keys[dest] else {
        return Err(TransferSkip::UnverifiedIdentity);
    };
    let block_size = input.block_sizes[dest];
    if block_size == 0 {
        return Err(TransferSkip::UnverifiedIdentity);
    }
    let dest_prefix = transferable_prefix_blocks(input.candidates[dest].matches);

    // Largest prefix wins; ties go to the less loaded peer, then the lower
    // position (strict comparison keeps the first one seen).
    let mut best: Option<(usize, u32)> = None;
    for pos in 0..input.candidates.len() {
        if pos == dest
            || input.cache_keys[pos] != Some(dest_key)
            || input.block_sizes[pos] != block_size
        {
            continue;
        }
        let prefix = transferable_prefix_blocks(input.candidates[pos].matches);
        let better = match best {
            None => true,
            Some((best_pos, best_prefix)) => {
                prefix > best_prefix
                    || (prefix == best_prefix
                        && input.candidates[pos].load.active_requests
                            < input.candidates[best_pos].load.active_requests)
            }
        };
        if better {
            best = Some((pos, prefix));
        }
    }
    let Some((src, src_prefix)) = best else {
        return Err(TransferSkip::NoPeer);
    };
    if src_prefix <= dest_prefix {
        return Err(TransferSkip::PickedHoldsLongestPrefix);
    }
    let missing_blocks = src_prefix - dest_prefix;
    if missing_blocks < policy.min_prefix_blocks {
        return Err(TransferSkip::PrefixTooShort);
    }
    if input.excluded[src] {
        return Err(TransferSkip::SourceExcluded);
    }
    let (source, destination) = (&input.candidates[src], &input.candidates[dest]);
    if source.load.active_requests < policy.source_load_threshold {
        return Err(TransferSkip::SourceNotBusy);
    }
    if policy.require_capability {
        match (input.capabilities[src], input.capabilities[dest]) {
            (Some(from), Some(to)) if from.can_send_to(to) => {}
            _ => return Err(TransferSkip::Incompatible),
        }
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
        return Err(TransferSkip::PrefixTooShort);
    }

    let prefill_ms = |tokens: u64| tokens as f64 * policy.prefill_us_per_token / 1000.0;
    let transfer_ms =
        policy.transfer_fixed_ms + credited as f64 * policy.transfer_us_per_token / 1000.0;
    let dest_queue_ms = prefill_ms(destination.load.active_prefill_tokens);
    let source_wait_ms = prefill_ms(source.load.active_prefill_tokens);
    let rem_dest = (destination.input_tokens as u64)
        .saturating_sub(dest_cached)
        .saturating_sub(credited);
    let rem_src = (source.input_tokens as u64)
        .saturating_sub(u64::from(src_prefix) * bs)
        .max(1);
    let saved_ms = prefill_ms(credited);
    let fetch_beats_waiting =
        transfer_ms + dest_queue_ms + prefill_ms(rem_dest) < source_wait_ms + prefill_ms(rem_src);
    if !(fetch_beats_waiting && transfer_ms < saved_ms * policy.margin) {
        return Err(TransferSkip::CostRule);
    }

    let credited_blocks = credited.div_ceil(bs).min(u64::from(missing_blocks)) as u32;
    Ok(TransferPlan {
        plan_id: 0,
        source: source.target.clone(),
        destination: destination.target.clone(),
        blocks: (dest_prefix, dest_prefix + credited_blocks),
        block_size,
        credited_tokens: credited,
        estimated_transfer_ms: transfer_ms,
        estimated_saved_prefill_ms: saved_ms,
        estimated_source_wait_ms: source_wait_ms,
        estimated_destination_queue_ms: dest_queue_ms,
    })
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::kv_index::{HashMode, Locality};
    use crate::policies::kv_aware::LoadSample;

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
            }
        }

        fn plan(&self, policy: &TransferPolicy) -> Result<TransferPlan, TransferSkip> {
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
            plan_transfer(
                &TransferPlannerInput {
                    candidates: &candidates,
                    cache_keys: &keys,
                    block_sizes: &self.block_sizes,
                    capabilities: &caps,
                    excluded: &self.excluded,
                    picked: 0,
                },
                policy,
            )
        }
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
                },
                &enabled(),
            )
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
        // 124 * 16 = 1984 cached, 1999 - 1984 = 15 creditable, one block missing.
        let plan = pair.plan(&policy).unwrap();
        assert_eq!(plan.credited_tokens, 15);
        assert_eq!(plan.blocks, (124, 125));
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
            },
            &policy,
        );
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
                },
                &enabled(),
            )
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
            },
            &enabled(),
        )
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
        let input = TransferPlannerInput {
            candidates: &candidates,
            cache_keys: &keys,
            block_sizes: &pair.block_sizes,
            capabilities: &[None, None],
            excluded: &pair.excluded,
            picked: 0,
        };
        assert_eq!(planner.plan(&input), plan_transfer(&input, &enabled()));
        let off = CostRuleTransferPlanner {
            policy: TransferPolicy::default(),
        };
        assert_eq!(off.plan(&input), Err(TransferSkip::Disabled));
    }

    #[test]
    fn skip_labels_are_stable_and_distinct() {
        let all = [
            TransferSkip::Disabled,
            TransferSkip::NoPeer,
            TransferSkip::UnverifiedIdentity,
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
    }
}
