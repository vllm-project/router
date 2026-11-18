//! Bind event sources to their layout, shared index and subscription lease.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

/// All ranks of a worker share `active`; owners of one canonical publisher and
/// exact topic share `strict_rank`. Neither identifies a remote engine lifetime.
/// Lock order is supervisor workers -> strict rank -> indexer. Keep the final
/// rank decision and index mutation together, never holding this lock over await.
#[derive(Clone)]
pub(in crate::kv_index) struct SourceLease {
    pub(in crate::kv_index) strict_rank: Arc<Mutex<bool>>,
    pub(in crate::kv_index) active: Arc<AtomicBool>,
}

impl SourceLease {
    pub(in crate::kv_index) fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }

    pub(in crate::kv_index) fn with_active<T>(
        &self,
        operation: impl FnOnce(bool) -> T,
    ) -> Option<T> {
        let strict = self.strict_rank.lock();
        self.is_active().then(|| operation(*strict))
    }
}

use crate::kv_hash::{BlockKeyGenerator, SequenceHash};
use crate::kv_index::index::source::{StreamControl, StreamPhase, StreamState};
use crate::kv_index::index::IndexBatch;
use crate::kv_index::layout::{CacheGroup, GroupRule};
use crate::kv_index::SourceError;
use crate::kv_index::{KvBlockIndexer, ResidencyOwner, RoutableTarget, StorageTier};
use std::collections::HashSet;

#[cfg(all(test, feature = "cpu", feature = "ready"))]
std::thread_local! {
    pub(in crate::kv_index) static AFTER_SOURCE_SNAPSHOT: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        std::cell::RefCell::new(None);
}

/// A bounded index for one verified identity, worker, rank and incarnation.
/// Stores and removals across all groups share one lock and one sequence.
pub struct SourceBinding {
    pub(in crate::kv_index) target: RoutableTarget,
    pub(in crate::kv_index) groups: Vec<CacheGroup>,
    pub(in crate::kv_index) keys: BlockKeyGenerator,
    /// Layout-declared restore-time recomputation window (vLLM
    /// `prefix_replay_tokens`). It applies to the whole layout: a group that
    /// publishes no events can still make this window mandatory.
    pub(in crate::kv_index) replay_tokens: u32,
    /// LoRA adapters this worker serves, as `(name, path)`. The engine seeds an
    /// adapter's chain with both, so a layout that does not declare an adapter
    /// cannot credit it, and one that declares the wrong path credits nothing.
    pub(in crate::kv_index) adapters: Vec<(String, String)>,
    pub(in crate::kv_index) max_entries: usize,
    pub(in crate::kv_index) partial_hash_hits: bool,
    /// The same domain index used by full-attention workers. Layout and stream
    /// state live here; cache residency lives only in this shared index.
    pub(in crate::kv_index) index: Arc<KvBlockIndexer>,
    pub(in crate::kv_index) owner: ResidencyOwner,
    pub(in crate::kv_index) control: Arc<StreamControl>,
    pub(in crate::kv_index) binding: Option<crate::kv_index::source::binding::SourceLease>,
}

impl SourceBinding {
    /// Test constructor keyed by a `PYTHONHASHSEED` instead of a root digest.
    #[cfg_attr(coverage_nightly, coverage(off))]
    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    pub fn new(
        target: RoutableTarget,
        groups: Vec<CacheGroup>,
        hash_block_size: u32,
        hash_seed: u32,
        max_entries: usize,
    ) -> Result<Self, SourceError> {
        if hash_block_size == 0 {
            return Err(SourceError::InvalidLayout);
        }
        Self::with_keys(
            target,
            groups,
            max_entries,
            BlockKeyGenerator::new(hash_block_size as usize, u64::from(hash_seed)),
        )
    }

    /// The root digest and group layout must come from the same verified engine.
    pub fn new_with_root_hash(
        target: RoutableTarget,
        groups: Vec<CacheGroup>,
        hash_block_size: u32,
        root_hash: SequenceHash,
        max_entries: usize,
    ) -> Result<Self, SourceError> {
        if hash_block_size == 0 {
            return Err(SourceError::InvalidLayout);
        }
        Self::with_keys(
            target,
            groups,
            max_entries,
            BlockKeyGenerator::from_root_hash(hash_block_size as usize, root_hash),
        )
    }

    fn with_keys(
        target: RoutableTarget,
        groups: Vec<CacheGroup>,
        max_entries: usize,
        keys: BlockKeyGenerator,
    ) -> Result<Self, SourceError> {
        let mut ids = HashSet::new();
        if groups.is_empty()
            || groups.len() > 64
            || max_entries == 0
            || groups.iter().any(|group| {
                group.block_size == 0
                    || !(group.block_size as usize).is_multiple_of(keys.block_size())
                    || !ids.insert(group.group_idx)
                    // Compressed state covers `tokens_per_state` tokens, so the
                    // group boundary has to be a multiple of that ratio.
                    || match group.kind {
                        GroupRule::SlidingWindow { window_tokens } => window_tokens == 0,
                        GroupRule::CompressedFullAttention { tokens_per_state } => {
                            tokens_per_state == 0
                                || !group.block_size.is_multiple_of(tokens_per_state)
                        }
                        _ => false,
                    }
            })
        {
            return Err(SourceError::InvalidLayout);
        }
        let incarnation = crate::kv_index::next_owner_incarnation();
        let owner = ResidencyOwner::Worker {
            source: target.instance_id.clone(),
            dp_rank: target.dp_rank,
            incarnation,
        };
        Ok(Self {
            target,
            groups,
            keys,
            replay_tokens: 0,
            adapters: Vec::new(),
            max_entries,
            partial_hash_hits: false,
            index: Arc::new(KvBlockIndexer::new()),
            owner,
            control: Arc::new(StreamControl::default()),
            binding: None,
        })
    }

    /// Bind an admitted layout to its supervisor's cache domain before replay.
    pub(crate) fn with_shared_index(mut self, index: Arc<KvBlockIndexer>) -> Self {
        debug_assert!(self.index.with_source_snapshot(&self.target, |view| {
            view.stream(&self.owner)
                .is_none_or(|state| state.next_seq.is_none())
        }));
        self.index = index;
        self
    }

    pub(in crate::kv_index) fn with_source_binding(
        mut self,
        binding: crate::kv_index::source::binding::SourceLease,
    ) -> Self {
        self.binding = Some(binding);
        self
    }

    pub(in crate::kv_index) fn with_active_binding<R>(
        &self,
        operation: impl FnOnce() -> R,
    ) -> Option<R> {
        match &self.binding {
            Some(binding) => binding.with_active(|_| operation()),
            None => Some(operation()),
        }
    }

    pub(in crate::kv_index) fn with_stream_batch<R>(
        &self,
        operation: impl FnOnce(&mut StreamState, &mut IndexBatch<'_>) -> R,
    ) -> R {
        self.index
            .with_stream_batch(&self.target, &self.owner, &self.control, operation)
    }

    /// Declare the layout's restore-time recomputation window
    /// (`prefix_replay_tokens`); it lowers the credit for the whole layout.
    pub fn with_replay_tokens(mut self, replay_tokens: u32) -> Self {
        self.replay_tokens = replay_tokens;
        self
    }

    /// Enable only when the source contract identifies hash-sized stores as
    /// fine-grained prefix entries, rather than secondary pages.
    pub fn with_partial_hash_hits(mut self, enabled: bool) -> Self {
        self.partial_hash_hits = enabled;
        self
    }

    /// Declare the LoRA adapters this worker serves. Adapter pages are a
    /// separate chain keyed by `(name, path)`; without the declaration their
    /// pages are skipped rather than credited to the base model.
    pub fn with_adapters(mut self, adapters: Vec<(String, String)>) -> Self {
        self.adapters = adapters;
        self
    }

    pub(in crate::kv_index) fn adapter_path(&self, name: &str) -> Option<&str> {
        self.adapters
            .iter()
            .find(|(declared, _)| declared == name)
            .map(|(_, path)| path.as_str())
    }

    /// Begin tracking from an externally established empty-cache barrier.
    /// This is not a snapshot or replay implementation. Never call it merely
    /// because a subscription started or a gap occurred.
    #[cfg(test)]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(crate) fn begin_empty(&self, next_seq: u64) {
        self.with_active_binding(|| {
            self.with_stream_batch(|state, tx| {
                self.reset_stream(state, tx, Some(next_seq), true);
            });
        });
    }

    /// Start a full replay from an externally verified empty origin. Queries
    /// remain cold until the caller verifies a continuous replay watermark.
    #[cfg(test)]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(in crate::kv_index::source) fn begin_replay(&self, next_seq: u64) {
        self.with_active_binding(|| {
            self.with_stream_batch(|state, tx| {
                self.reset_stream(state, tx, Some(next_seq), false);
            });
        });
    }

    pub(in crate::kv_index::source) fn reset_stream(
        &self,
        state: &mut StreamState,
        tx: &mut IndexBatch<'_>,
        next_seq: Option<u64>,
        visible: bool,
    ) {
        tx.clear_owner_exact(&self.owner);
        state.reset(next_seq, visible);
        if visible {
            tx.resume_source(&self.target.instance_id, self.target.dp_rank);
        }
    }

    #[cfg(test)]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(in crate::kv_index::source) fn finish_replay(
        &self,
        expected_next_seq: u64,
    ) -> Result<(), SourceError> {
        self.with_active_binding(|| {
            self.with_stream_batch(|state, tx| self.publish_stream(state, tx, expected_next_seq))
        })
        .unwrap_or(Err(SourceError::Untrusted))
    }

    pub(in crate::kv_index::source) fn publish_stream(
        &self,
        state: &mut StreamState,
        tx: &mut IndexBatch<'_>,
        expected_next_seq: u64,
    ) -> Result<(), SourceError> {
        if state.visible() || state.next_seq != Some(expected_next_seq) {
            return Self::fail(state, tx, &self.owner, SourceError::Sequence);
        }
        tx.resume_source(&self.target.instance_id, self.target.dp_rank);
        state.publish();
        state.phase = StreamPhase::Live;
        Ok(())
    }

    /// Discard all claims on a gap, restart, lease retirement or lost event.
    pub(in crate::kv_index::source) fn invalidate(&self) {
        self.with_stream_batch(|state, tx| {
            self.reset_stream(state, tx, None, false);
            state.lease_id = 0;
        });
    }

    /// Calculate one consistent local snapshot. Zero means a miss or an
    /// untrusted source.
    pub fn reusable_tokens(&self, tokens: &[u32], max_reusable_tokens: usize) -> u64 {
        self.reusable_tokens_in(None, tokens, max_reusable_tokens)
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[path = "binding_tests.rs"]
pub(crate) mod tests;

use crate::kv_index::source::endpoints::EventSource;
use crate::kv_index::{AdapterCapability, CacheCapabilities, RankCapability};

/// Adapter declarations are the only way to credit adapter pages, so a
/// malformed declaration must fail the layout instead of binding a chain that
/// is not the one the engine stores.
fn validate_adapters(
    adapters: &[AdapterCapability],
) -> Result<Vec<(String, String)>, &'static str> {
    let mut names = HashSet::new();
    if adapters.len() > 64
        || adapters.iter().any(|adapter| {
            adapter.name.is_empty()
                || adapter.path.is_empty()
                || !names.insert(adapter.name.as_str())
        })
    {
        return Err("invalid adapter declaration");
    }
    Ok(adapters
        .iter()
        .map(|adapter| (adapter.name.clone(), adapter.path.clone()))
        .collect())
}

impl CacheCapabilities {
    /// The capability declared for one DP rank. Ranks are independent
    /// instances: a rank that is absent here simply gets no hybrid credit and
    /// must not invalidate the other ranks of the same worker.
    pub fn rank_capability(&self, rank: u32) -> Option<&RankCapability> {
        self.ranks.iter().find(|entry| entry.rank == rank)
    }

    /// Construct an index for one rank and its event source. Unknown layouts
    /// get no credit.
    pub fn index_for_source(
        &self,
        source: &EventSource,
        target: RoutableTarget,
        max_entries: usize,
    ) -> Result<SourceBinding, &'static str> {
        let mut ranks = HashSet::new();
        if self.schema_version != 1
            || self.ranks.iter().any(|entry| !ranks.insert(entry.rank))
            || target.dp_rank != source.dp_rank
            || target.instance_id != source.source
            || source.topic.is_empty()
            || source.replay_endpoint.as_deref().is_none_or(str::is_empty)
        {
            return Err("invalid capability or source binding");
        }
        let rank = self
            .rank_capability(source.dp_rank)
            .ok_or("missing rank capability")?;
        if rank.model.is_empty()
            || rank.weight_version.is_empty()
            || !rank.instance_topic
            || rank.publisher_topic.as_deref() != Some(source.topic.as_str())
            || rank.hash_algorithm != "sha256_cbor"
            || !rank.full_hashes
            || !rank.prefix_caching
            || rank.hash_lookahead_tokens != 0
            || rank.speculative_tokens != 0
        {
            return Err("unsupported engine identity or hash contract");
        }
        let root = rank
            .root_hash
            .as_deref()
            .and_then(parse_hash)
            .ok_or("invalid root hash")?;
        let hash_block_size = rank.hash_block_size.ok_or("missing hash block size")?;
        let mut groups = Vec::with_capacity(rank.groups.len());
        let mut replay_tokens = 0;
        for group in &rank.groups {
            // vLLM rewinds a hit by one replay window shared by every replay
            // group. Disagreeing windows have no single restore point, and a
            // replay window only exists for sliding-window layouts.
            if group.prefix_replay_tokens > 0 {
                if !matches!(group.kind.as_str(), "sliding_window" | "sliding_window_mla") {
                    return Err("unsupported group restore contract");
                }
                if replay_tokens != 0 && replay_tokens != group.prefix_replay_tokens {
                    return Err("unsupported group restore contract");
                }
                replay_tokens = group.prefix_replay_tokens;
            }
            // Scratch groups such as the Qwen4 QSA key ring hold per-request
            // state, never shareable state, and emit no block events. They are
            // excluded from the index only when their shape proves that.
            if !group.prefix_cacheable {
                if group.prefix_replay_tokens > 0
                    && matches!(
                        (group.kind.as_str(), group.spec_type.as_str()),
                        ("sliding_window", "SlidingWindowSpec")
                            | ("sliding_window_mla", "SlidingWindowMLASpec")
                            | ("mla_attention", "MLAAttentionSpec")
                    )
                {
                    // Bounded replay: the engine rebuilds this group's trailing
                    // window from the other groups' state. Its own blocks are
                    // not required to be resident, but the window must still be
                    // charged to the layout.
                    if group.sliding_window.is_none_or(|window| {
                        window == 0 || u64::from(window) != u64::from(group.prefix_replay_tokens)
                    }) {
                        return Err("unsupported bounded replay group");
                    }
                    continue;
                }
                if group.spec_type != "CircularBufferSpec"
                    || group.tokens_per_state.as_deref() != Some("1")
                    || group.block_size == 0
                {
                    return Err("unsupported scratch group");
                }
                continue;
            }
            let kind = match (group.kind.as_str(), group.spec_type.as_str()) {
                ("full_attention", "FullAttentionSpec") if group.sliding_window.is_none() => {
                    GroupRule::FullAttention
                }
                ("mla_attention", "MLAAttentionSpec")
                    if group.sliding_window.is_none()
                        && group
                            .cache_role
                            .as_deref()
                            .is_none_or(|role| role == "sparse") =>
                {
                    // A compressed group names the tokens one state covers.
                    // Without that declaration, restore from the root like a
                    // full-attention block instead of rejecting the layout.
                    match group.tokens_per_state.as_deref() {
                        // Declared but malformed: the engine would not honor
                        // this shape, so the layout is refused.
                        Some(declared) => {
                            let tokens_per_state = declared
                                .parse::<u32>()
                                .ok()
                                .filter(|value| {
                                    *value > 0 && group.block_size.is_multiple_of(*value)
                                })
                                .ok_or("unsupported compressed attention group")?;
                            GroupRule::CompressedFullAttention { tokens_per_state }
                        }
                        // No declaration: the page restores from the root.
                        None => GroupRule::FullAttention,
                    }
                }
                ("sliding_window", "SlidingWindowSpec")
                | ("sliding_window_mla", "SlidingWindowMLASpec") => {
                    let window = group
                        .sliding_window
                        .filter(|window| *window > 0)
                        .ok_or("unsupported sliding window group")?;
                    GroupRule::SlidingWindow {
                        window_tokens: u64::from(window),
                    }
                }
                ("mamba", "MambaSpec")
                    if group.mamba_cache_mode.as_deref() == Some("align")
                        && group.prefill_checkpoint_alignment.is_none_or(|alignment| {
                            alignment > 0 && group.block_size.is_multiple_of(alignment)
                        }) =>
                {
                    GroupRule::Checkpoint
                }
                _ => return Err("unsupported group kind"),
            };
            groups.push(CacheGroup {
                group_idx: group.group_idx,
                block_size: group.block_size,
                kind,
            });
        }
        // The declared layout, not just the indexed subset, must describe a
        // hybrid cache: several groups, at least one attention group, and some
        // state that a single attention hit cannot prove (a checkpoint group,
        // a sliding-window group, or a declared replay window). One attention
        // group is enough when the layout says it is the whole layout: a model
        // whose sparse state never publishes block events (V4.1) has no group
        // whose checkpoint the credit could contradict.
        let has_attention = groups.iter().any(|group| {
            matches!(
                group.kind,
                GroupRule::FullAttention
                    | GroupRule::CompressedFullAttention { .. }
                    | GroupRule::SlidingWindow { .. }
            )
        });
        let has_state = groups.iter().any(|group| {
            matches!(
                group.kind,
                GroupRule::Checkpoint | GroupRule::SlidingWindow { .. }
            )
        });
        let declared_single_attention =
            groups.len() == 1 && rank.declared_groups == Some(1) && has_attention;
        if groups.is_empty()
            || !has_attention
            || !(has_state || replay_tokens > 0 || declared_single_attention)
        {
            return Err("not a supported mixed layout");
        }
        let adapters = validate_adapters(&self.adapters)?;
        Ok(
            SourceBinding::new_with_root_hash(target, groups, hash_block_size, root, max_entries)
                .map_err(|_| "invalid group layout")?
                .with_replay_tokens(replay_tokens)
                .with_adapters(adapters)
                .with_partial_hash_hits(rank.partial_hash_hits),
        )
    }
}

fn parse_hash(value: &str) -> Option<[u8; 32]> {
    if value.len() != 64 {
        return None;
    }
    fn nibble(value: u8) -> Option<u8> {
        match value {
            b'0'..=b'9' => Some(value - b'0'),
            b'a'..=b'f' => Some(value - b'a' + 10),
            b'A'..=b'F' => Some(value - b'A' + 10),
            _ => None,
        }
    }
    let mut root = [0; 32];
    for (index, out) in root.iter_mut().enumerate() {
        let pair = &value.as_bytes()[index * 2..index * 2 + 2];
        *out = (nibble(pair[0])? << 4) | nibble(pair[1])?;
    }
    Some(root)
}

impl SourceBinding {
    /// The credited prefix and the tier it is credited at. A page held in host
    /// memory or on storage is still restorable, but the policy weighs it by the
    /// transfer credit of its tier, so the boundary is reported at the slowest
    /// tier that any group needs to satisfy it.
    pub(in crate::kv_index) fn credit_parts(
        &self,
        namespace: Option<&str>,
        tokens: &[u32],
        max_reusable_tokens: usize,
    ) -> (u64, Option<StorageTier>) {
        if self
            .binding
            .as_ref()
            .is_some_and(|binding| !binding.is_active())
        {
            return (0, None);
        }
        if self.control.generation() == 0 {
            return (0, None);
        }
        let hashes = match namespace {
            None => self.keys.generate_block_keys(tokens),
            Some(name) => match self.adapter_path(name) {
                Some(path) => self
                    .keys
                    .generate_block_keys_for_adapter(tokens, name, path),
                None => return (0, None),
            },
        };
        let prepared = crate::kv_index::index::prepare_sparse_keys(namespace, &hashes);
        // Copy only the required tier evidence under one source view.
        // Boundary calculation uses these owned values and does not block
        // writers for other workers while it scans the request.
        let snapshot = self.index.with_source_snapshot(&self.target, |view| {
            let state = view.stream(&self.owner)?;
            if state.next_seq.is_none() || !state.visible() {
                return None;
            }
            let claims: Vec<_> = self
                .groups
                .iter()
                .map(|group| view.page_tiers(&self.owner, group.group_idx, &prepared))
                .collect();
            Some((
                claims,
                view.source_version(&self.target),
                state.reset_generation,
            ))
        });
        let Some((claims, version, reset_generation)) = snapshot else {
            return (0, None);
        };
        #[cfg(all(test, feature = "cpu", feature = "ready"))]
        if let Some(hook) = AFTER_SOURCE_SNAPSHOT.with(|slot| slot.borrow_mut().take()) {
            hook();
        }
        let (credited, tier) = crate::kv_index::layout::credit_from_pages(
            &self.target,
            &self.groups,
            &claims,
            self.keys.block_size(),
            tokens.len().saturating_sub(1).min(max_reusable_tokens) as u64,
            self.replay_tokens,
        );
        if tier.is_none() || u32::try_from(credited / self.keys.block_size() as u64).is_err() {
            return (0, None);
        }
        if self.control.generation() != reset_generation
            || !self.index.source_version_is_current(&version)
            || self
                .binding
                .as_ref()
                .is_some_and(|binding| !binding.is_active())
        {
            return (0, None);
        }
        (credited, tier)
    }
    pub(in crate::kv_index) fn reusable_tokens_in(
        &self,
        namespace: Option<&str>,
        tokens: &[u32],
        max_reusable_tokens: usize,
    ) -> u64 {
        self.credit_parts(namespace, tokens, max_reusable_tokens).0
    }
}
